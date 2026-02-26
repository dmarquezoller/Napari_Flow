import importlib
import numpy as np
import traceback
import json
import hashlib
import sys
import time
import threading
from qtpy.QtCore import QObject, Signal

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

    def _reset_node_cache(self, node):
        node.last_signature = None
        node.cached_results = {}

    def _execute_single_node(self, node, library_def):
        try:
            current_signature = self.calculate_signature(node)
            if current_signature == node.last_signature:
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

    def _build_loop_group(self, loop_node, sorted_nodes):
        start_nodes = []
        end_nodes = []

        for socket in getattr(loop_node, "logic_outputs", []):
            for edge in socket.connected_edges:
                if edge.end_socket:
                    n = edge.end_socket.node
                    if n is not loop_node:
                        start_nodes.append(n)

        for socket in getattr(loop_node, "logic_inputs", []):
            for edge in socket.connected_edges:
                if edge.start_socket:
                    n = edge.start_socket.node
                    if n is not loop_node:
                        end_nodes.append(n)

        # Preserve order while de-duplicating
        start_nodes = list(dict.fromkeys(start_nodes))
        end_nodes = list(dict.fromkeys(end_nodes))
        if not start_nodes or not end_nodes:
            return None

        # Nodes in the loop body are those reachable from starts (forward)
        # and also able to reach ends (backward), using DATA edges only.
        forward = set()
        stack = list(start_nodes)
        while stack:
            node = stack.pop()
            if node in forward or getattr(node, "node_type", "") == "loop_control":
                continue
            forward.add(node)
            for out_socket in getattr(node, "outputs", []):
                for edge in out_socket.connected_edges:
                    if edge.end_socket:
                        nxt = edge.end_socket.node
                        if nxt not in forward:
                            stack.append(nxt)

        backward = set()
        stack = list(end_nodes)
        while stack:
            node = stack.pop()
            if node in backward or getattr(node, "node_type", "") == "loop_control":
                continue
            backward.add(node)
            for in_socket in getattr(node, "inputs", []):
                if in_socket.connected_edges:
                    prev = in_socket.connected_edges[0].start_socket.node
                    if prev not in backward:
                        stack.append(prev)

        body_nodes = (forward & backward) | set(start_nodes) | set(end_nodes)
        body_order = [n for n in sorted_nodes if n in body_nodes and n is not loop_node]
        if not body_order:
            return None

        params = getattr(loop_node, "parameters", {}) or {}
        mode = params.get("mode", "N times")
        try:
            iterations = int(params.get("iterations", 1))
        except Exception:
            iterations = 1
        iterations = max(1, iterations)

        return {
            "loop_node": loop_node,
            "mode": mode,
            "iterations": iterations,
            "body_order": body_order,
        }

    def detect_loop_groups(self, sorted_nodes):
        groups = []
        for node in sorted_nodes:
            if getattr(node, "node_type", "") != "loop_control":
                continue
            group = self._build_loop_group(node, sorted_nodes)
            if group is None:
                self.log_signal.emit(
                    f"⚠️ Loop '{node.title}' is missing logic links; skipping loop behavior."
                )
                continue
            groups.append(group)
        return groups

    def _execute_loop_group(self, group, library_def):
        loop_node = group["loop_node"]
        body_order = group["body_order"]
        mode = group["mode"]

        if mode == "Until confirm":
            self._loop_stop_requested = False
            self.loop_control_state_signal.emit(True, loop_node.uid)
            try:
                iteration = 0
                while True:
                    if self._loop_stop_requested:
                        break
                    iteration += 1
                    self.log_signal.emit(
                        f"🔁 Loop '{loop_node.title}' iteration {iteration} (Until confirm)"
                    )
                    if iteration > 1:
                        for node in body_order:
                            self._reset_node_cache(node)

                    for node in body_order:
                        if self._loop_stop_requested:
                            break
                        try:
                            self._execute_single_node(node, library_def)
                        except InterruptedError:
                            self._loop_stop_requested = True
                            break
                        if self._loop_stop_requested:
                            break

                    if self._loop_stop_requested:
                        break
            finally:
                self.loop_control_state_signal.emit(False, loop_node.uid)
                self._loop_stop_requested = False
        else:
            iterations = group["iterations"]
            for iteration in range(iterations):
                self.log_signal.emit(
                    f"🔁 Loop '{loop_node.title}' iteration {iteration + 1}/{iterations}"
                )
                if iteration > 0:
                    for node in body_order:
                        self._reset_node_cache(node)

                for node in body_order:
                    self._execute_single_node(node, library_def)

    def run(self):
        try:
            self.log_signal.emit("--- Starting Smart Execution ---")
            
            from .napari_plugin_v2 import Node, NODE_LIBRARY
            nodes = [item for item in self.scene.items() if isinstance(item, Node)]
            
            if not nodes:
                self.log_signal.emit("Pipeline is empty.")
                self.finished_signal.emit()
                return

            sorted_nodes = self.topological_sort(nodes)
            loop_groups = self.detect_loop_groups(sorted_nodes)

            group_by_node_uid = {}
            for group in loop_groups:
                gid = group["loop_node"].uid
                for body_node in group["body_order"]:
                    group_by_node_uid[body_node.uid] = gid

            executed_groups = set()
            groups_by_uid = {g["loop_node"].uid: g for g in loop_groups}

            for node in sorted_nodes:
                if getattr(node, "node_type", "") == "loop_control":
                    continue

                group_uid = group_by_node_uid.get(node.uid)
                if group_uid is not None:
                    if group_uid in executed_groups:
                        continue
                    self._execute_loop_group(groups_by_uid[group_uid], NODE_LIBRARY)
                    executed_groups.add(group_uid)
                    continue

                self._execute_single_node(node, NODE_LIBRARY)

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

        # Copy metadata
        layer_meta = layer.metadata.copy() if hasattr(layer, "metadata") else {}

        # Axis map -> axes string
        axis_map = func_params.get("axis_map", [])
        if axis_map:
            row = axis_map[0]
            axes = [row.get(f"d{i}") for i in range(5) if row.get(f"d{i}", "-") != "-"]
            layer_meta["axes"] = "".join(axes)

        layer_meta["source_layer"] = target_name

        return {"data_out": (layer.data, layer_meta)}

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
                
                if source_socket_name in source_node.cached_results:
                    data_package = source_node.cached_results[source_socket_name]
                    print("🔎 INPUT from", source_node.title, "socket", source_socket_name, "->", type(data_package))
                    if isinstance(data_package, list):
                        print("   list len:", len(data_package), "first:", type(data_package[0]), "tuplelen:", len(data_package[0]) if isinstance(data_package[0], tuple) else None)
                    elif isinstance(data_package, tuple):
                        print("   tuple len:", len(data_package), "types:", [type(x) for x in data_package])

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
        result = func(**args)

        
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
            Let napari auto-detect these for the new layer.
            We keep colormap so the processed output preserves the source
            channel's color (e.g. green, magenta tint from OME-Zarr).
            """
            for key in ("contrast_limits", "multiscales"):
                meta.pop(key, None)
            return meta

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
                # Strip stale contrast_limits from inherited metadata; node's own
                # meta (res[1]) takes precedence if it supplies new ones.
                if "contrast_limits" not in res[1]:
                    _sanitize_inherited_meta(merged)
                return (res[0], merged)

            # 4) Default: wrap as (data, meta)
            # This is the path taken when a node (e.g. gaussian_blur via dispatch)
            # returns raw data (array or list of arrays) without its own metadata.
            # We must rename so the output doesn't clobber the input layer.
            meta = current_metadata.copy()
            _ensure_processed_name(meta)
            _sanitize_inherited_meta(meta)
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
            if node.title == "Gaussian Blur":
                # DEBUG
                print("🧪 GAUSS EMIT out_name=", out_name, "type(out_data)=", type(out_data))
                if isinstance(out_data, tuple):
                    print("   tuple len=", len(out_data), "types=", [type(x) for x in out_data])
                    if len(out_data) == 2 and isinstance(out_data[1], dict):
                        print("   meta name=", out_data[1].get("name"), "multiscale=", out_data[1].get("multiscale"))
                if isinstance(out_data, list):
                    print("   list len=", len(out_data), "first type=", type(out_data[0]) if out_data else None)
                ####
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
