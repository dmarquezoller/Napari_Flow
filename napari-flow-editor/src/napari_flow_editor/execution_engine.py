import importlib
import numpy as np
import traceback
import json
import hashlib
import sys
import time
import threading
from qtpy.QtCore import QObject, Signal
from .flow_nodes.decorator import push_dispatch_context, pop_dispatch_context


def infer_layout_kind(axes: str) -> str:
    """
    Normalize an axes string (e.g. "TYX") into a semantic layout class.
    """
    if not axes:
        return "unknown_layout"

    a = str(axes).upper()
    mapping = {
        "YX": "2d_image",
        "YXC": "2d_image_channels",
        "ZYX": "3d_image",
        "ZYXC": "3d_image_channels",
        "TYX": "3d_timeline",
        "TYXC": "3d_timeline_channels",
        "TZYX": "4d_timeline",
        "TZYXC": "4d_timeline_channels",
    }
    return mapping.get(a, "unknown_layout")


class ExecutionWorker(QObject):
    # Signals
    log_signal = Signal(str)
    node_status_signal = Signal(str, str)     # (UID, "green"/"yellow"/"red")
    result_signal = Signal(str, str, object)  # (Title, OutputName, Data)
    finished_signal = Signal()
    error_signal = Signal(str)

    # Interactive node signals  (engine ⇄ main-thread dialog)
    interaction_request_signal = Signal(str, dict)   # (node_uid, interactive_config)
    # Loop control signal (engine ⇄ main-thread UI stop button)
    loop_control_state_signal = Signal(bool, str)    # (active, loop_node_uid)
    # Exec trace signal: transition from node via named exec output.
    exec_transition_signal = Signal(str, str)        # (from_node_uid, output_socket_name)
    # The main thread calls ``provide_interaction_result`` which sets the
    # threading.Event so the worker thread can continue.

    def __init__(self, scene, viewer):
        super().__init__()
        self.scene = scene
        self.viewer = viewer

        # Interaction synchronisation primitives
        self._interaction_event = threading.Event()
        self._interaction_result = None   # set by main thread
        self._loop_stop_requested = False
        self._stop_sentinel = object()

    # Called from the **main thread** (via signal/slot) to unblock the worker.
    def provide_interaction_result(self, data):
        """
        The main-thread dialog calls this once the user clicks Run (or Cancel).
        ``data`` is either the drawn shapes list, or ``None`` for cancel.
        """
        self._interaction_result = data
        self._interaction_event.set()

    def _request_interaction(self, node_uid, interactive_config):
        """
        Emit a signal to the main thread requesting user interaction,
        then block until the main thread calls ``provide_interaction_result``.
        Returns the interaction data (e.g. list of shape arrays) or None.
        """
        self._interaction_event.clear()
        self._interaction_result = None
        self.interaction_request_signal.emit(node_uid, interactive_config)
        # Block worker thread until the user finishes drawing + clicks Run
        self._interaction_event.wait()
        return self._interaction_result

    def request_loop_stop(self):
        """Called from the main thread to stop an active 'Until confirm' loop."""
        self._loop_stop_requested = True
        # If worker is blocked waiting for an interactive node,
        # unblock it immediately so the loop can stop now.
        self._interaction_result = self._stop_sentinel
        self._interaction_event.set()

    @staticmethod
    def _looks_like_multiscale_data(data):
        t = type(data)
        name = getattr(t, "__name__", "")
        module = str(getattr(t, "__module__", "") or "").lower()
        if name == "MultiScaleData" or "multiscale" in module:
            return True

        # Fallback for wrappers that do not expose a canonical class/module
        # but still provide napari-like multiscale metadata.
        shapes = getattr(data, "shapes", None)
        if shapes is not None:
            try:
                return len(shapes) > 1
            except Exception:
                return False
        return False

    @classmethod
    def _normalize_layer_data(cls, data):
        """
        Normalize viewer layer payloads to match dispatch expectations.

        For multiscale viewer wrappers, convert to a plain list of levels so
        downstream nodes (e.g. gaussian via dispatch) always take the pyramid
        path instead of trying to run skimage directly on the wrapper object.
        """
        if cls._looks_like_multiscale_data(data):
            try:
                return list(data)
            except Exception:
                return data
        return data

    def _reset_node_cache(self, node):
        node.last_signature = None
        node.cached_results = {}

    def _execute_single_node(self, node, library_def, force_recompute=False):
        try:
            current_signature = self.calculate_signature(node)
            if (not force_recompute) and current_signature == node.last_signature:
                self.log_signal.emit(f"Skipping: {node.title} (Cached)")
                self.node_status_signal.emit(node.uid, "green")
                return

            self.node_status_signal.emit(node.uid, "yellow")
            results = self.execute_node_logic(node, library_def)
            node.cached_results = results
            node.last_signature = current_signature
            self.node_status_signal.emit(node.uid, "green")
        except InterruptedError:
            # Expected path when a loop stop interrupts interactive waiting.
            self.node_status_signal.emit(node.uid, "green")
            raise
        except Exception:
            self.node_status_signal.emit(node.uid, "red")
            raise

    def _next_logic_node(self, node, output_name=None):
        next_node, _ = self._next_logic_step(node, output_name)
        return next_node

    def _next_logic_step(self, node, output_name=None):
        outputs = getattr(node, "logic_outputs", [])
        if not outputs:
            return None, None

        target_socket = None
        if output_name is not None:
            target_socket = next((s for s in outputs if s.name == output_name), None)
        if target_socket is None:
            target_socket = outputs[0]

        if not target_socket.connected_edges:
            return None, target_socket.name
        edge = target_socket.connected_edges[0]
        if edge.end_socket is None:
            return None, target_socket.name
        return edge.end_socket.node, target_socket.name

    def _is_data_source_node(self, node):
        """Pure data provider: no required data inputs and no exec input."""
        return (
            len(getattr(node, "inputs", [])) == 0
            and len(getattr(node, "logic_inputs", [])) == 0
        )

    def _execute_exec_path(self, start_node, library_def, force_recompute=False):
        current = start_node
        visited = set()

        while current is not None:
            if self._loop_stop_requested and force_recompute:
                break
            if current.uid in visited:
                raise RuntimeError(
                    f"Exec cycle detected near '{current.title}'. "
                    "Only loop-control nodes should express repetition."
                )
            visited.add(current.uid)

            if current.node_type == "loop_control":
                current = self._execute_loop_node(current, library_def)
                continue

            if current.node_type == "begin":
                next_node, out_name = self._next_logic_step(current)
                if next_node is not None and out_name is not None:
                    self.exec_transition_signal.emit(current.uid, out_name)
                current = next_node
                continue

            try:
                self._execute_single_node(
                    current, library_def, force_recompute=force_recompute
                )
            except InterruptedError:
                if self._loop_stop_requested:
                    break
                raise

            next_node, out_name = self._next_logic_step(current)
            if next_node is not None and out_name is not None:
                self.exec_transition_signal.emit(current.uid, out_name)
            current = next_node

    def _collect_linear_exec_nodes(self, start_node, stop_uids=None, max_steps=2048):
        """
        Collect nodes on the default exec path starting at ``start_node``.
        Used for UI status resets between loop iterations.
        """
        nodes = []
        visited = set()
        current = start_node
        stop_uids = set(stop_uids or [])
        steps = 0

        while current is not None and steps < max_steps:
            if current.uid in stop_uids or current.uid in visited:
                break
            nodes.append(current)
            visited.add(current.uid)
            steps += 1

            # Prefer explicit loop body path inside nested loops.
            if current.node_type == "loop_control":
                current, _ = self._next_logic_step(current, "loop_body")
            else:
                current, _ = self._next_logic_step(current)

        return nodes

    def _set_nodes_gray(self, nodes):
        for node in nodes:
            try:
                node.status = "gray"
            except Exception:
                pass
            self.node_status_signal.emit(node.uid, "gray")

    def _execute_loop_node(self, loop_node, library_def):
        params = getattr(loop_node, "parameters", {}) or {}
        mode = params.get("mode", "N times")
        try:
            iterations = int(params.get("iterations", 1))
        except Exception:
            iterations = 1
        iterations = max(1, iterations)

        body_start, _ = self._next_logic_step(loop_node, "loop_body")
        completed_start, _ = self._next_logic_step(loop_node, "completed")

        if body_start is None:
            self.log_signal.emit(
                f"⚠️ Loop '{loop_node.title}' has no Loop Body connection."
            )
            return completed_start

        if mode == "Until confirm":
            self._loop_stop_requested = False
            self.loop_control_state_signal.emit(True, loop_node.uid)
            body_nodes = self._collect_linear_exec_nodes(
                body_start, stop_uids={loop_node.uid}
            )
            try:
                iteration = 0
                while True:
                    if self._loop_stop_requested:
                        break
                    iteration += 1
                    self._set_nodes_gray(body_nodes)
                    self.log_signal.emit(
                        f"🔁 Loop '{loop_node.title}' iteration {iteration} (Until confirm)"
                    )
                    self.exec_transition_signal.emit(loop_node.uid, "loop_body")
                    self._execute_exec_path(
                        body_start, library_def, force_recompute=True
                    )
                    if self._loop_stop_requested:
                        break
            finally:
                self.loop_control_state_signal.emit(False, loop_node.uid)
                self._loop_stop_requested = False
        else:
            body_nodes = self._collect_linear_exec_nodes(
                body_start, stop_uids={loop_node.uid}
            )
            for i in range(iterations):
                self._set_nodes_gray(body_nodes)
                self.log_signal.emit(
                    f"🔁 Loop '{loop_node.title}' iteration {i + 1}/{iterations}"
                )
                self.exec_transition_signal.emit(loop_node.uid, "loop_body")
                self._execute_exec_path(body_start, library_def, force_recompute=True)

        if completed_start is not None:
            self.exec_transition_signal.emit(loop_node.uid, "completed")
        return completed_start

    def run(self):
        try:
            self.log_signal.emit("--- Starting Smart Execution ---")
            
            from .napari_plugin_v2 import Node, NODE_LIBRARY
            nodes = [item for item in self.scene.items() if isinstance(item, Node)]
            
            if not nodes:
                self.log_signal.emit("Pipeline is empty.")
                self.finished_signal.emit()
                return

            begin_nodes = [n for n in nodes if getattr(n, "node_type", "") == "begin"]
            if not begin_nodes:
                raise RuntimeError(
                    "No Begin node found. Add a Begin node and connect its exec output."
                )
            if len(begin_nodes) > 1:
                raise RuntimeError(
                    "Multiple Begin nodes found. Use a single Begin node for execution."
                )

            begin_node = begin_nodes[0]
            first_exec, begin_out = self._next_logic_step(begin_node)
            if first_exec is None:
                self.log_signal.emit("Begin node is not connected to any exec thread.")
            else:
                if begin_out is not None:
                    self.exec_transition_signal.emit(begin_node.uid, begin_out)
                self._execute_exec_path(first_exec, NODE_LIBRARY)

            self.log_signal.emit("--- Execution Finished ---")

        except Exception as e:
            import traceback
            full_error = traceback.format_exc()
            self.log_signal.emit(f"CRITICAL ERROR:\n{full_error}")
            self.error_signal.emit(str(e))
        finally:
            self.finished_signal.emit()

    def calculate_signature(self, node):
        try:
            param_str = json.dumps(getattr(node, "parameters", {}), sort_keys=True, default=str)
            input_sigs = []
            for socket in node.inputs:
                if socket.connected_edges:
                    edge = socket.connected_edges[0]
                    parent = edge.start_socket.node
                    sig = parent.last_signature if parent.last_signature else "dirty"
                    input_sigs.append(sig)
                else:
                    input_sigs.append("none")
            
            combined = param_str + "".join(input_sigs)
            return hashlib.md5(combined.encode('utf-8')).hexdigest()
        except:
            return "dirty"
        

    def resolve_input_node(self, node, func_params):
        if node.node_type != "get_layer":
            return None

        target_name = func_params.get("layer_name")
        if target_name not in self.viewer.layers:
            raise ValueError(f"Layer '{target_name}' not found.")

        layer = self.viewer.layers[target_name]

        # Build metadata from the richest available source:
        # 1) layer.as_layer_data_tuple() visual kwargs (colormap, gamma, etc.)
        # 2) layer.metadata custom user metadata (overrides/additions)
        layer_meta = {}
        layer_data = getattr(layer, "data", None)

        if hasattr(layer, "as_layer_data_tuple"):
            try:
                ldt = layer.as_layer_data_tuple()
                if (
                    isinstance(ldt, tuple)
                    and len(ldt) >= 2
                    and isinstance(ldt[1], dict)
                ):
                    layer_data = ldt[0]
                    layer_meta.update(dict(ldt[1]))
            except Exception:
                # Fallback to direct layer attributes below
                pass

        if hasattr(layer, "metadata") and isinstance(layer.metadata, dict):
            layer_meta.update(layer.metadata.copy())

        # Defensive fallback: copy common display attrs if not present.
        for attr in (
            "name",
            "axis_labels",
            "colormap",
            "contrast_limits",
            "gamma",
            "rgb",
            "interpolation2d",
            "interpolation3d",
            "opacity",
            "blending",
            "visible",
            "scale",
            "translate",
            "rotate",
            "shear",
            "affine",
        ):
            if attr not in layer_meta and hasattr(layer, attr):
                try:
                    layer_meta[attr] = getattr(layer, attr)
                except Exception:
                    pass

        def infer_ndim(data_obj):
            obj = data_obj
            if isinstance(obj, list) and obj and hasattr(obj[0], "shape"):
                obj = obj[0]
            if hasattr(obj, "shape"):
                try:
                    return len(obj.shape)
                except Exception:
                    return None
            return None

        def is_default_yx_axis_row(row_dict):
            if not isinstance(row_dict, dict):
                return False
            return (
                row_dict.get("d0", "-") == "Y"
                and row_dict.get("d1", "-") == "X"
                and row_dict.get("d2", "-") == "-"
                and row_dict.get("d3", "-") == "-"
                and row_dict.get("d4", "-") == "-"
            )

        # Axis map -> axes string
        axis_map = func_params.get("axis_map", [])
        if axis_map:
            row = axis_map[0]
            axes = [row.get(f"d{i}") for i in range(5) if row.get(f"d{i}", "-") != "-"]
            axes_str = "".join(axes)
            if axes_str:
                data_ndim = infer_ndim(layer_data)
                existing_axes = str(layer_meta.get("axes", "") or "")
                existing_axes_ndim = len(existing_axes) if existing_axes else None
                has_valid_existing_axes = (
                    data_ndim is not None
                    and existing_axes_ndim == data_ndim
                )
                is_default_row = is_default_yx_axis_row(row)
                explicit_matches_ndim = (
                    data_ndim is None or len(axes_str) == data_ndim
                )

                # Case 1: preserve valid source axes when axis-map is still default.
                if has_valid_existing_axes and is_default_row:
                    pass
                # Case 2: explicit (non-default) axis-map -> trust user.
                # For default rows, require ndim match to avoid accidental
                # clobbering. For explicit user mappings, preserve as entered.
                elif (not is_default_row) or explicit_matches_ndim:
                    layer_meta["axes"] = axes_str
                    layer_meta["layout_kind"] = infer_layout_kind(axes_str)
                    layer_meta.setdefault(
                        "axis_labels", tuple(a.lower() for a in axes_str)
                    )
                # Case 3: missing source axes + default YX on 3D/4D -> infer safe defaults.
                elif is_default_row and not has_valid_existing_axes:
                    inferred = None
                    if data_ndim == 3:
                        # Default to timeline stack to avoid unintended blur across
                        # axis-0 when users keep default axis-map.
                        inferred = "TYX"
                    elif data_ndim == 4:
                        inferred = "TZYX"
                    elif data_ndim == 5:
                        inferred = "TZYXC"

                    if inferred is not None:
                        layer_meta["axes"] = inferred
                        layer_meta["layout_kind"] = infer_layout_kind(inferred)
                        layer_meta.setdefault(
                            "axis_labels", tuple(a.lower() for a in inferred)
                        )

        layer_meta["source_layer"] = target_name

        normalized_data = self._normalize_layer_data(layer_data)
        return {"data_out": (normalized_data, layer_meta)}

    def execute_node_logic(self, node, library_def):
        self.log_signal.emit(f"Executing: {node.title}...")
        
        current_metadata = {} 
        func_inputs = {}
        
        # --- A. Inputs (YOUR ORIGINAL LOGIC) ---
        for socket in node.inputs:
            if socket.connected_edges:
                edge = socket.connected_edges[0]
                source_node = edge.start_socket.node
                source_socket_name = edge.start_socket.name

                if self._is_data_source_node(source_node):
                    source_is_dirty = (
                        getattr(source_node, "status", "") == "gray"
                        or getattr(source_node, "last_signature", None) is None
                        or source_socket_name not in source_node.cached_results
                    )
                    if source_is_dirty:
                        # Allow pure data providers (e.g. Get Layer) outside the
                        # exec thread; compute them lazily when first requested
                        # OR when they were invalidated by parameter changes.
                        self._execute_single_node(
                            source_node, library_def, force_recompute=True
                        )

                if source_socket_name in source_node.cached_results:
                    data_package = source_node.cached_results[source_socket_name]

                    # Unpack (Data, Meta) Envelope
                    if isinstance(data_package, tuple) and len(data_package) == 2 and isinstance(data_package[1], dict):
                        data_only = data_package[0]
                        incoming_meta = data_package[1]
                        current_metadata.update(incoming_meta)
                        func_inputs[socket.name] = data_only
                    else:
                        func_inputs[socket.name] = data_package
                else:
                    raise ValueError(f"Missing data from upstream node: {source_node.title}")

        # --- B. Parameters ---
        func_params = getattr(node, 'params', {}).copy()
        if not func_params and hasattr(node, 'parameters'):
             func_params = node.parameters.copy()
        
        # --- SPECIAL CASE: GET LAYER (RESTORED EXACTLY AS WAS) ---
        # This accesses self.viewer directly, just like your old code.

        input_result = self.resolve_input_node(node, func_params)
        if input_result is not None:
            return input_result
        # --- C. Import & Run Normal Nodes ---
        def_data = library_def.get(node.node_type, {})
        
        if "executable" in def_data:
            func = def_data["executable"]
        elif "execution_path" in def_data:
            exec_path = def_data["execution_path"] 
            module_name, func_name = exec_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
        elif hasattr(node, 'func'):
            func = node.func
        else:
            raise ValueError(f"No executable found for {node.title}")

        # Clean Params
        clean_params = {}
        for k, v in func_params.items():
            if isinstance(v, dict) and "value" in v:
                clean_params[k] = v["value"]
            else:
                clean_params[k] = v

        # --- INTERACTIVE NODE HANDLING ---
        # If the node's library entry has an ``interactive`` config, request
        # user interaction from the main thread before running the function.
        interactive_config = def_data.get("interactive")
        if interactive_config:
            self.log_signal.emit(
                f"⏳ Waiting for user interaction on '{node.title}'..."
            )
            interaction_data = self._request_interaction(node.uid, interactive_config)
            if interaction_data is self._stop_sentinel:
                raise InterruptedError("Loop stop requested during interaction.")
            if interaction_data is None:
                if self._loop_stop_requested:
                    raise InterruptedError("Loop stop requested during interaction.")
                raise RuntimeError(
                    f"Interaction cancelled for '{node.title}'."
                )
            clean_params["interaction"] = interaction_data

        # RUN
        args = {**func_inputs, **clean_params}
        dispatch_token = push_dispatch_context(
            {
                "metadata": current_metadata.copy(),
                "node_uid": node.uid,
                "node_type": node.node_type,
            }
        )
        try:
            result = func(**args)
        finally:
            pop_dispatch_context(dispatch_token)

        
        # --- D. Format Results ---
        # Check library def first, then fallback to node sockets
        output_names = def_data.get("outputs", [])
        if not output_names and hasattr(node, 'outputs'):
             output_names = list(node.outputs.keys()) if isinstance(node.outputs, dict) else [s.name for s in node.outputs]

        node_outputs = {}


        def is_layer_data_tuple(x):
            # (data, meta, layer_type)
            return (
                isinstance(x, tuple)
                and len(x) == 3
                and isinstance(x[1], dict)
                and isinstance(x[2], str)
            )

        def normalize_layer_data_tuple(ldt):
            """Merge engine-collected metadata into napari LayerDataTuple meta."""
            data, meta, layer_type = ldt
            merged = {}
            merged.update(current_metadata)
            merged.update(meta or {})
            return (data, merged, layer_type)

        # Determine if this node is a passthrough/input node that should NOT rename
        node_category = def_data.get("category", "").lower()
        is_input_node = node_category in ("input",)

        def _ensure_processed_name(meta):
            """
            If the metadata still carries the original source layer name,
            append '(Processed)' so we never overwrite the input layer.
            Skip renaming for Input-category nodes (e.g. Select Layer) that
            are just passing data through.
            """
            if is_input_node:
                return meta
            name = meta.get("name", "")
            if name and "(Processed)" not in name:
                meta["name"] = f"{name} (Processed)"
            return meta

        def _sanitize_inherited_meta(meta):
            """
            Remove metadata keys that are only valid for the *source* layer
            and would be wrong for a processed result (e.g. contrast_limits
            computed on different data, or multiscales descriptors).
            We keep display settings (contrast_limits, colormap, gamma, etc.)
            so processed outputs preserve source appearance by default.
            """
            for key in ("multiscales",):
                meta.pop(key, None)
            return meta

        def _auto_contrast_limits(data):
            """
            Estimate display contrast limits from output data.
            Uses a bounded sample so very large/dask arrays remain cheap.
            Returns [min, max] or None.
            """
            try:
                arr = data
                # Multiscale: estimate from the highest-resolution level.
                if isinstance(arr, list) and len(arr) > 0:
                    arr = arr[0]

                if not hasattr(arr, "shape"):
                    return None

                shape = tuple(int(s) for s in arr.shape)
                if len(shape) == 0:
                    return None

                n_dim = len(shape)
                slicer = []
                for axis, size in enumerate(shape):
                    if axis >= n_dim - 2:
                        # Keep a centered spatial window.
                        width = min(size, 512)
                        start = max(0, (size - width) // 2)
                        slicer.append(slice(start, start + width))
                    else:
                        # Collapse large non-spatial dims to the center.
                        if size <= 4:
                            slicer.append(slice(0, size))
                        else:
                            center = size // 2
                            slicer.append(slice(center, center + 1))

                sampled = arr[tuple(slicer)]

                # Lazily backed data (dask): compute sampled window only.
                if hasattr(sampled, "compute"):
                    sampled = sampled.compute()

                sampled = np.asarray(sampled)
                if sampled.size == 0 or not np.issubdtype(sampled.dtype, np.number):
                    return None

                finite = sampled[np.isfinite(sampled)]
                if finite.size == 0:
                    return None

                lo = float(np.percentile(finite, 1.0))
                hi = float(np.percentile(finite, 99.5))
                if hi <= lo:
                    lo = float(finite.min())
                    hi = float(finite.max())
                    if hi <= lo:
                        hi = lo + 1e-6
                return [lo, hi]
            except Exception:
                return None

        def wrap_result(res):
            # 1) If node returned a single LayerDataTuple: keep it as LayerDataTuple
            if is_layer_data_tuple(res):
                return normalize_layer_data_tuple(res)

            # 2) If node returned multiple layers: keep list of LayerDataTuples
            if isinstance(res, list) and len(res) > 0 and is_layer_data_tuple(res[0]):
                return [normalize_layer_data_tuple(x) for x in res]

            # 3) Your (data, meta) envelope: merge meta
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
                merged = current_metadata.copy()
                merged.update(res[1])
                # If the node's own meta explicitly provides a "name", respect it.
                # Otherwise ensure we don't overwrite the source layer.
                if "name" not in res[1]:
                    _ensure_processed_name(merged)
                _sanitize_inherited_meta(merged)
                # Keep inherited contrast_limits when present; only auto-estimate
                # if neither inherited nor node-provided limits exist.
                if "contrast_limits" not in merged:
                    auto_limits = _auto_contrast_limits(res[0])
                    if auto_limits is not None:
                        merged["contrast_limits"] = auto_limits
                return (res[0], merged)

            # 4) Default: wrap as (data, meta)
            # This is the path taken when a node (e.g. gaussian_blur via dispatch)
            # returns raw data (array or list of arrays) without its own metadata.
            # We must rename so the output doesn't clobber the input layer.
            meta = current_metadata.copy()
            _ensure_processed_name(meta)
            _sanitize_inherited_meta(meta)
            # Keep inherited contrast_limits when available; fallback to
            # auto-estimate only when nothing is provided upstream.
            if "contrast_limits" not in meta:
                auto_limits = _auto_contrast_limits(res)
                if auto_limits is not None:
                    meta["contrast_limits"] = auto_limits
            return (res, meta)

        if isinstance(result, tuple) and len(output_names) > 1 and not is_layer_data_tuple(result):
            for i, name in enumerate(output_names):
                if i < len(result): node_outputs[name] = wrap_result(result[i])
        elif isinstance(result, dict):
             for k, v in result.items():
                 node_outputs[k] = wrap_result(v)
        else:
             # FIX FOR SAVE IMAGE (Sink Nodes)
             # Only try to assign output if the node actually HAS outputs
             if output_names: 
                 node_outputs[output_names[0]] = wrap_result(result)
             elif hasattr(node, 'outputs') and len(node.outputs) > 0:
                 first_out = list(node.outputs)[0] if isinstance(node.outputs, list) else list(node.outputs.keys())[0]
                 out_name = first_out.name if hasattr(first_out, 'name') else first_out
                 node_outputs[out_name] = wrap_result(result)
             else:
                 # Pass for nodes with no outputs (like Save Image)
                 pass

        # Emit Results
        for out_name, out_data in node_outputs.items():
            self.result_signal.emit(node.title, out_name, out_data)
            
        return node_outputs

    def topological_sort(self, nodes):
        visited = set()
        stack = []
        def visit(n):
            if n in visited: return
            visited.add(n)
            for input_socket in n.inputs:
                if input_socket.connected_edges:
                    visit(input_socket.connected_edges[0].start_socket.node)
            stack.append(n)
        for node in nodes: visit(node)
        return stack
