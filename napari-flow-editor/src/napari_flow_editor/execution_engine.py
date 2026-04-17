import importlib
import csv
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
    # Step-mode state snapshot emitted after each RUN NEXT click.
    step_state_signal = Signal(object)
    # The main thread calls ``provide_interaction_result`` which sets the
    # threading.Event so the worker thread can continue.

    def __init__(self, scene, viewer, run_mode="full", step_state=None):
        super().__init__()
        self.scene = scene
        self.viewer = viewer
        self.run_mode = str(run_mode or "full").strip().lower()

        # Interaction synchronisation primitives
        self._interaction_event = threading.Event()
        self._interaction_result = None   # set by main thread
        self._loop_stop_requested = False
        self._stop_sentinel = object()
        self._batch_row = None
        self._batch_index = None
        self._batch_total = 0
        self._step_state = self._sanitize_step_state(step_state)

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

    @staticmethod
    def _is_empty_csv_value(value):
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        return False

    @staticmethod
    def _coerce_csv_value(value, param_def):
        if value is None:
            return None
        param_type = str((param_def or {}).get("type", "string")).strip().lower()
        raw = value
        if isinstance(raw, str):
            raw = raw.strip()
        if param_type == "int":
            return int(float(raw))
        if param_type == "float":
            return float(raw)
        if param_type == "bool":
            if isinstance(raw, bool):
                return raw
            token = str(raw).strip().lower()
            if token in ("1", "true", "yes", "y", "on"):
                return True
            if token in ("0", "false", "no", "n", "off"):
                return False
            raise ValueError(f"Cannot parse boolean value '{value}'.")
        # enum/string/path/table fall back to raw textual value.
        return raw

    def _apply_dynamic_param_bindings(self, node, func_params, library_def):
        if not isinstance(self._batch_row, dict):
            return func_params

        bindings = getattr(node, "dynamic_param_bindings", {}) or {}
        if not isinstance(bindings, dict) or not bindings:
            return func_params

        params_def = (
            (library_def.get(getattr(node, "node_type", ""), {}) or {}).get(
                "parameters", {}
            )
            or {}
        )
        for param_name, column_name in bindings.items():
            col = str(column_name).strip()
            if not col:
                continue
            if col not in self._batch_row:
                raise ValueError(
                    f"Batch column '{col}' not found for "
                    f"node '{node.title}' parameter '{param_name}'."
                )
            raw = self._batch_row.get(col)
            if self._is_empty_csv_value(raw):
                # Empty cell -> keep node's current/static parameter value.
                continue
            param_def = params_def.get(param_name, {})
            try:
                func_params[param_name] = self._coerce_csv_value(raw, param_def)
            except Exception as exc:
                raise ValueError(
                    f"Invalid batch value '{raw}' for "
                    f"node '{node.title}' parameter '{param_name}': {exc}"
                ) from exc

        return func_params

    @staticmethod
    def _detect_csv_dialect(sample_text):
        try:
            return csv.Sniffer().sniff(sample_text, delimiters=",;\t|")
        except Exception:
            return csv.excel

    def _read_batch_rows(self, csv_path):
        rows = []
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
            sample = fh.read(4096)
            fh.seek(0)
            dialect = self._detect_csv_dialect(sample)
            reader = csv.DictReader(fh, dialect=dialect)
            fieldnames = [str(x).strip() for x in (reader.fieldnames or []) if str(x).strip()]
            if not fieldnames:
                raise ValueError("CSV header is empty.")

            for row in reader:
                normalized = {}
                for key, val in (row or {}).items():
                    k = str(key).strip()
                    if not k:
                        continue
                    normalized[k] = val
                if not normalized:
                    continue
                if all(self._is_empty_csv_value(v) for v in normalized.values()):
                    continue
                rows.append(normalized)

        if not rows:
            raise ValueError("CSV has no data rows.")
        return rows

    def _set_batch_context(self, row, index, total):
        self._batch_row = dict(row or {})
        self._batch_index = int(index)
        self._batch_total = int(total)

    def _clear_batch_context(self):
        self._batch_row = None
        self._batch_index = None
        self._batch_total = 0

    @staticmethod
    def _sanitize_step_state(step_state):
        if not isinstance(step_state, dict):
            return None

        loop_stack = []
        for raw in step_state.get("loop_stack", []) or []:
            if not isinstance(raw, dict):
                continue
            frame = {
                "loop_uid": str(raw.get("loop_uid", "")).strip(),
                "mode": str(raw.get("mode", "N times")),
                "iterations": max(1, int(raw.get("iterations", 1) or 1)),
                "iteration": max(0, int(raw.get("iteration", 0) or 0)),
                "body_uid": str(raw.get("body_uid", "")).strip() or None,
                "completed_uid": str(raw.get("completed_uid", "")).strip() or None,
            }
            if frame["loop_uid"]:
                loop_stack.append(frame)

        state = {
            "begin_uid": str(step_state.get("begin_uid", "")).strip() or None,
            "current_uid": str(step_state.get("current_uid", "")).strip() or None,
            "current_force_recompute": bool(
                step_state.get("current_force_recompute", False)
            ),
            "finished": bool(step_state.get("finished", False)),
            "loop_stack": loop_stack,
            "stop_loop_requested": bool(step_state.get("stop_loop_requested", False)),
        }
        if not state["begin_uid"]:
            return None
        return state

    @staticmethod
    def _serialize_step_state(step_state):
        if not isinstance(step_state, dict):
            return None
        return json.loads(json.dumps(step_state))

    @staticmethod
    def _node_by_uid(uid_map, uid):
        if not uid:
            return None
        return uid_map.get(uid)

    def _active_loop_frame(self, step_state):
        stack = (step_state or {}).get("loop_stack", [])
        if not stack:
            return None
        return stack[-1]

    def _make_initial_step_state(self, begin_node):
        first_exec, begin_out = self._next_logic_step(begin_node)
        if begin_out is not None and first_exec is not None:
            self.exec_transition_signal.emit(begin_node.uid, begin_out)

        if first_exec is None:
            return {
                "begin_uid": begin_node.uid,
                "current_uid": None,
                "current_force_recompute": False,
                "finished": True,
                "loop_stack": [],
                "stop_loop_requested": False,
            }

        return {
            "begin_uid": begin_node.uid,
            "current_uid": first_exec.uid,
            "current_force_recompute": False,
            "finished": False,
            "loop_stack": [],
            "stop_loop_requested": False,
        }

    def _reset_all_nodes_for_batch(self, nodes):
        for node in nodes:
            try:
                self._reset_node_cache(node)
                node.status = "gray"
                self.node_status_signal.emit(node.uid, "gray")
            except Exception:
                pass

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

            if current.node_type in ("begin", "begin_batch"):
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

    def _prepare_step_state(self, begin_node, uid_map):
        state = self._sanitize_step_state(self._step_state)
        if state is None or state.get("begin_uid") != begin_node.uid:
            state = self._make_initial_step_state(begin_node)
            self._loop_stop_requested = False
            return state

        # If graph changed and the stored pointer no longer exists, restart.
        current_uid = state.get("current_uid")
        if current_uid and current_uid not in uid_map:
            state = self._make_initial_step_state(begin_node)
            self._loop_stop_requested = False
            return state

        # Drop stale loop frames (e.g. loop node removed).
        cleaned_stack = []
        for frame in state.get("loop_stack", []):
            if frame.get("loop_uid") in uid_map:
                cleaned_stack.append(frame)
        state["loop_stack"] = cleaned_stack
        self._loop_stop_requested = bool(state.pop("stop_loop_requested", False))

        return state

    def _handle_loop_control_step(self, loop_node, step_state, uid_map):
        stack = step_state.setdefault("loop_stack", [])
        frame = stack[-1] if stack and stack[-1].get("loop_uid") == loop_node.uid else None

        if frame is None:
            params = getattr(loop_node, "parameters", {}) or {}
            mode = str(params.get("mode", "N times"))
            try:
                iterations = int(params.get("iterations", 1))
            except Exception:
                iterations = 1
            iterations = max(1, iterations)

            body_start, _ = self._next_logic_step(loop_node, "loop_body")
            completed_start, _ = self._next_logic_step(loop_node, "completed")
            frame = {
                "loop_uid": loop_node.uid,
                "mode": mode,
                "iterations": iterations,
                "iteration": 0,
                "body_uid": getattr(body_start, "uid", None),
                "completed_uid": getattr(completed_start, "uid", None),
            }
            stack.append(frame)
            if mode == "Until confirm":
                self._loop_stop_requested = False
                self.loop_control_state_signal.emit(True, loop_node.uid)

        body_uid = frame.get("body_uid")
        completed_uid = frame.get("completed_uid")
        mode = frame.get("mode", "N times")

        if not body_uid:
            self.log_signal.emit(
                f"⚠️ Loop '{loop_node.title}' has no Loop Body connection."
            )
            if mode == "Until confirm":
                self.loop_control_state_signal.emit(False, loop_node.uid)
                self._loop_stop_requested = False
            stack.pop()
            if completed_uid:
                self.exec_transition_signal.emit(loop_node.uid, "completed")
            step_state["current_uid"] = completed_uid
            step_state["current_force_recompute"] = False
            if completed_uid is None and not stack:
                step_state["finished"] = True
            return

        if mode == "Until confirm":
            if self._loop_stop_requested:
                self.loop_control_state_signal.emit(False, loop_node.uid)
                self._loop_stop_requested = False
                stack.pop()
                if completed_uid:
                    self.exec_transition_signal.emit(loop_node.uid, "completed")
                step_state["current_uid"] = completed_uid
                step_state["current_force_recompute"] = False
                if completed_uid is None and not stack:
                    step_state["finished"] = True
                return

            frame["iteration"] = int(frame.get("iteration", 0)) + 1
            body_node = self._node_by_uid(uid_map, body_uid)
            body_nodes = self._collect_linear_exec_nodes(
                body_node, stop_uids={loop_node.uid}
            ) if body_node is not None else []
            self._set_nodes_gray(body_nodes)
            self.log_signal.emit(
                f"🔁 Loop '{loop_node.title}' iteration {frame['iteration']} (Until confirm)"
            )
            self.exec_transition_signal.emit(loop_node.uid, "loop_body")
            step_state["current_uid"] = body_uid
            step_state["current_force_recompute"] = True
            step_state["finished"] = False
            return

        # N times
        if int(frame.get("iteration", 0)) >= int(frame.get("iterations", 1)):
            stack.pop()
            if completed_uid:
                self.exec_transition_signal.emit(loop_node.uid, "completed")
            step_state["current_uid"] = completed_uid
            step_state["current_force_recompute"] = False
            if completed_uid is None and not stack:
                step_state["finished"] = True
            return

        frame["iteration"] = int(frame.get("iteration", 0)) + 1
        body_node = self._node_by_uid(uid_map, body_uid)
        body_nodes = self._collect_linear_exec_nodes(
            body_node, stop_uids={loop_node.uid}
        ) if body_node is not None else []
        self._set_nodes_gray(body_nodes)
        self.log_signal.emit(
            f"🔁 Loop '{loop_node.title}' iteration {frame['iteration']}/{frame['iterations']}"
        )
        self.exec_transition_signal.emit(loop_node.uid, "loop_body")
        step_state["current_uid"] = body_uid
        step_state["current_force_recompute"] = True
        step_state["finished"] = False

    def _execute_single_step(self, step_state, uid_map, library_def):
        if step_state.get("finished", False):
            return False

        executed_any = False
        guard = 0

        while guard < 4096:
            guard += 1

            current_uid = step_state.get("current_uid")
            if not current_uid:
                frame = self._active_loop_frame(step_state)
                if frame is not None:
                    step_state["current_uid"] = frame.get("loop_uid")
                    step_state["current_force_recompute"] = False
                    continue
                step_state["finished"] = True
                return executed_any

            current = self._node_by_uid(uid_map, current_uid)
            if current is None:
                step_state["finished"] = True
                return executed_any

            if current.node_type in ("begin", "begin_batch"):
                next_node, out_name = self._next_logic_step(current)
                if next_node is not None and out_name is not None:
                    self.exec_transition_signal.emit(current.uid, out_name)
                step_state["current_uid"] = getattr(next_node, "uid", None)
                step_state["current_force_recompute"] = False
                if next_node is None and not step_state.get("loop_stack"):
                    step_state["finished"] = True
                    return executed_any
                continue

            if current.node_type == "loop_control":
                self._handle_loop_control_step(current, step_state, uid_map)
                if step_state.get("finished", False):
                    return executed_any
                continue

            force_recompute = bool(step_state.get("current_force_recompute", False))
            try:
                self._execute_single_node(
                    current, library_def, force_recompute=force_recompute
                )
            except InterruptedError:
                if self._loop_stop_requested:
                    frame = self._active_loop_frame(step_state)
                    if frame is not None:
                        step_state["current_uid"] = frame.get("loop_uid")
                        step_state["current_force_recompute"] = False
                        continue
                    step_state["finished"] = True
                    return executed_any
                raise

            next_node, out_name = self._next_logic_step(current)
            if next_node is not None and out_name is not None:
                self.exec_transition_signal.emit(current.uid, out_name)
                step_state["current_uid"] = next_node.uid
                # Keep force_recompute while traversing loop body.
                step_state["current_force_recompute"] = force_recompute
                step_state["finished"] = False
            else:
                frame = self._active_loop_frame(step_state)
                if frame is not None:
                    step_state["current_uid"] = frame.get("loop_uid")
                    step_state["current_force_recompute"] = False
                    step_state["finished"] = False
                else:
                    step_state["current_uid"] = None
                    step_state["current_force_recompute"] = False
                    step_state["finished"] = True

            executed_any = True
            return executed_any

        raise RuntimeError("Step execution exceeded guard limit (possible cycle).")

    def run_next_step(self):
        from .napari_plugin_v2 import Node, NODE_LIBRARY

        nodes = [item for item in self.scene.items() if isinstance(item, Node)]
        if not nodes:
            self.log_signal.emit("Pipeline is empty.")
            self._step_state = None
            self.step_state_signal.emit(None)
            return

        begin_nodes = [n for n in nodes if getattr(n, "node_type", "") in ("begin", "begin_batch")]
        if not begin_nodes:
            raise RuntimeError(
                "No Begin node found. Add a Begin node and connect its exec output."
            )
        if len(begin_nodes) > 1:
            raise RuntimeError(
                "Multiple Begin nodes found. Use a single Begin node for step execution."
            )

        begin_node = begin_nodes[0]
        if getattr(begin_node, "node_type", "") == "begin_batch":
            raise RuntimeError(
                "Run Next does not support Begin Batch. Use RUN PIPELINE for batch execution."
            )

        uid_map = {n.uid: n for n in nodes}
        step_state = self._prepare_step_state(begin_node, uid_map)
        executed = self._execute_single_step(step_state, uid_map, NODE_LIBRARY)

        if step_state.get("finished", False):
            if not executed:
                self.log_signal.emit("Step mode: pipeline is complete.")
            # Ensure loop-stop UI is closed if an until-confirm loop ended.
            frame = self._active_loop_frame(step_state)
            if frame and frame.get("mode") == "Until confirm":
                self.loop_control_state_signal.emit(False, frame.get("loop_uid", ""))
            self._loop_stop_requested = False
            self._step_state = None
            self.step_state_signal.emit(None)
            return

        self._step_state = step_state
        self.step_state_signal.emit(self._serialize_step_state(step_state))

    def run(self):
        try:
            if self.run_mode == "step":
                self.run_next_step()
                return

            self.log_signal.emit("--- Starting Smart Execution ---")
            
            from .napari_plugin_v2 import Node, NODE_LIBRARY
            nodes = [item for item in self.scene.items() if isinstance(item, Node)]
            
            if not nodes:
                self.log_signal.emit("Pipeline is empty.")
                self.finished_signal.emit()
                return

            begin_nodes = [
                n
                for n in nodes
                if getattr(n, "node_type", "") in ("begin", "begin_batch")
            ]
            if not begin_nodes:
                raise RuntimeError(
                    "No Begin node found. Add a Begin or Begin Batch node and connect its exec output."
                )
            if len(begin_nodes) > 1:
                raise RuntimeError(
                    "Multiple Begin nodes found. Use a single Begin/Begin Batch node for execution."
                )

            begin_node = begin_nodes[0]
            first_exec, begin_out = self._next_logic_step(begin_node)
            begin_type = getattr(begin_node, "node_type", "")
            begin_label = "Begin Batch" if begin_type == "begin_batch" else "Begin"

            if first_exec is None:
                self.log_signal.emit(
                    f"{begin_label} node is not connected to any exec thread."
                )
            else:
                if begin_type == "begin_batch":
                    csv_path = str(
                        (getattr(begin_node, "parameters", {}) or {}).get("csv_path", "")
                    ).strip()
                    if not csv_path:
                        raise RuntimeError(
                            "Begin Batch requires a CSV path."
                        )
                    batch_rows = self._read_batch_rows(csv_path)
                    total_rows = len(batch_rows)
                    self.log_signal.emit(
                        f"📦 Begin Batch: loaded {total_rows} row(s) from {csv_path}"
                    )
                    try:
                        for i, row in enumerate(batch_rows, start=1):
                            self._set_batch_context(row=row, index=i, total=total_rows)
                            self._reset_all_nodes_for_batch(nodes)
                            self.log_signal.emit(f"📄 Batch row {i}/{total_rows}")
                            if begin_out is not None:
                                self.exec_transition_signal.emit(begin_node.uid, begin_out)
                            self._execute_exec_path(
                                first_exec, NODE_LIBRARY, force_recompute=True
                            )
                    finally:
                        self._clear_batch_context()
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
        func_params = self._apply_dynamic_param_bindings(node, func_params, library_def)
        
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
            prepared_interactive_config = dict(interactive_config)
            interaction_type = str(
                prepared_interactive_config.get("interaction_type", "shapes")
            ).strip().lower()

            if interaction_type == "layer_choice":
                layers_input = func_inputs.get("layers")
                choices = []
                if isinstance(layers_input, list):
                    for i, item in enumerate(layers_input):
                        if (
                            isinstance(item, tuple)
                            and len(item) == 3
                            and isinstance(item[1], dict)
                        ):
                            _, meta, lt = item
                            layer_name = str(meta.get("name", f"layer_{i}"))
                            layer_type = str(lt)
                            choices.append(
                                {
                                    "name": layer_name,
                                    "layer_type": layer_type,
                                    "label": f"{layer_name} ({layer_type})",
                                    "index": i,
                                }
                            )

                if not choices:
                    raise ValueError(
                        "Select Layer: no incoming layers available for selection."
                    )

                current_name = str(clean_params.get("layer_name", "")).strip()
                current_type = str(clean_params.get("layer_type", "")).strip()
                default_index = 0
                for i, ch in enumerate(choices):
                    if (
                        ch["name"] == current_name
                        and (not current_type or ch["layer_type"] == current_type)
                    ):
                        default_index = i
                        break
                else:
                    for i, ch in enumerate(choices):
                        if ch["name"] == current_name:
                            default_index = i
                            break

                prepared_interactive_config["choices"] = choices
                prepared_interactive_config["default_index"] = default_index
            elif interaction_type == "video_render":
                # Video rendering is performed on the main thread (UI-side),
                # but it must use the current node parameter values.
                prepared_interactive_config["video_params"] = dict(clean_params)

            self.log_signal.emit(
                f"⏳ Waiting for user interaction on '{node.title}'..."
            )
            interaction_data = self._request_interaction(
                node.uid, prepared_interactive_config
            )
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
        is_input_node = node_category in ("input", "inputs")

        def _assign_node_output_name(meta, output_name_hint=None):
            """
            Ensure processed node outputs use stable node-scoped layer names
            so intermediate layers do not overwrite each other.

            Input/passthrough nodes keep original source names.
            """
            if is_input_node:
                return meta

            inherited_name = str(current_metadata.get("name", "") or "").strip()
            current_name = str(meta.get("name", "") or "").strip()
            inherited_processed = (
                f"{inherited_name} (Processed)" if inherited_name else ""
            )

            # Rename when missing name OR when the name is clearly inherited
            # from upstream metadata (which would cause layer overwrite).
            should_rename = (
                not current_name
                or current_name == inherited_name
                or (inherited_processed and current_name == inherited_processed)
            )
            if not should_rename:
                return meta

            if len(output_names) > 1 and output_name_hint:
                suffix = str(output_name_hint).replace("_", " ").strip()
                meta["name"] = f"{node.title} - {suffix}"
            else:
                meta["name"] = f"{node.title} Output"
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

        def wrap_result(res, output_name_hint=None):
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
                _assign_node_output_name(merged, output_name_hint=output_name_hint)
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
            _assign_node_output_name(meta, output_name_hint=output_name_hint)
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
                if i < len(result):
                    node_outputs[name] = wrap_result(result[i], output_name_hint=name)
        elif isinstance(result, dict):
             for k, v in result.items():
                 node_outputs[k] = wrap_result(v, output_name_hint=k)
        else:
             # FIX FOR SAVE IMAGE (Sink Nodes)
             # Only try to assign output if the node actually HAS outputs
             if output_names: 
                 node_outputs[output_names[0]] = wrap_result(
                     result, output_name_hint=output_names[0]
                 )
             elif hasattr(node, 'outputs') and len(node.outputs) > 0:
                 first_out = list(node.outputs)[0] if isinstance(node.outputs, list) else list(node.outputs.keys())[0]
                 out_name = first_out.name if hasattr(first_out, 'name') else first_out
                 node_outputs[out_name] = wrap_result(
                     result, output_name_hint=out_name
                 )
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
