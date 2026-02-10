import importlib
import numpy as np
import traceback
import json
import hashlib
from qtpy.QtCore import QObject, Signal

class ExecutionWorker(QObject):
    # Signals
    log_signal = Signal(str)
    node_status_signal = Signal(str, str)     # New: sends (UID, "green"/"yellow"/"red")
    result_signal = Signal(str, str, object)  # (Title, OutputName, Data)
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, scene, viewer):
        super().__init__()
        self.scene = scene
        self.viewer = viewer

    def run(self):
        try:
            self.log_signal.emit("--- Starting Smart Execution ---")
            
            # Import Node class locally to identify items
            from .napari_plugin_v2 import Node, NODE_LIBRARY
            nodes = [item for item in self.scene.items() if isinstance(item, Node)]
            
            if not nodes:
                self.log_signal.emit("Pipeline is empty.")
                self.finished_signal.emit()
                return

            sorted_nodes = self.topological_sort(nodes)
            
            for node in sorted_nodes:
                try:
                    # 1. Calculate the 'Signature' (Inputs + Params)
                    # This tells us exactly if the node's job has changed
                    current_signature = self.calculate_signature(node)
                    
                    # 2. Check Cache (Memoization)
                    # If the node is already Green AND the signature matches, we can skip!
                    if node.status == "green" and current_signature == node.last_signature:
                        self.log_signal.emit(f"Skipping: {node.title} (Cached)")
                        continue # SKIP EXECUTION!
                    
                    # 3. If not cached, RUN
                    self.node_status_signal.emit(node.uid, "yellow") # Turn Yellow (Running)
                    
                    # Execute logic (Now returns results instead of storing internally)
                    results = self.execute_node_logic(node, NODE_LIBRARY)
                    
                    # 4. Save State (Update Memory)
                    node.cached_results = results
                    node.last_signature = current_signature
                    self.node_status_signal.emit(node.uid, "green") # Turn Green (Done)
                    
                except Exception as e:
                    self.node_status_signal.emit(node.uid, "red") # Turn Red (Error)
                    raise e # Stop the pipeline

            self.log_signal.emit("--- Execution Finished ---")

        except Exception as e:
            full_error = traceback.format_exc()
            self.log_signal.emit(f"CRITICAL ERROR:\n{full_error}")
            self.error_signal.emit(str(e))
        finally:
            self.finished_signal.emit()

    def calculate_signature(self, node):
        """
        Generates a unique hash string for the node's current state.
        If this hash matches the previous run, we know the result will be identical.
        """
        # A. Hash Parameters
        # We sort keys to ensure {"a":1, "b":2} gives same hash as {"b":2, "a":1}
        param_str = json.dumps(node.parameters, sort_keys=True, default=str)
        
        # B. Hash Input Sources
        input_sigs = []
        for socket in node.inputs:
            if socket.connected_edges:
                edge = socket.connected_edges[0]
                parent = edge.start_socket.node
                
                # We combine our parameters with the SIGNATURE of the parent.
                # If parent re-ran (new signature), our input signature changes too.
                # If parent is 'gray', it means it hasn't run, so we are 'dirty'.
                sig = parent.last_signature if parent.last_signature else "dirty"
                input_sigs.append(sig)
                
        # C. Combine and Hash
        combined = param_str + "".join(input_sigs)
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

    def execute_node_logic(self, node, library_def):
        """
        Performs the actual import and execution.
        Returns the dictionary of outputs.
        """
        self.log_signal.emit(f"Executing: {node.title}...")
        
        # --- A. Inputs (Grab from CACHE) ---
        func_inputs = {}
        for socket in node.inputs:
            if socket.connected_edges:
                edge = socket.connected_edges[0]
                source_node = edge.start_socket.node
                source_socket_name = edge.start_socket.name
                
                # CRITICAL CHANGE: Grab from source_node.cached_results directly
                if source_socket_name in source_node.cached_results:
                    data = source_node.cached_results[source_socket_name]
                    func_inputs[socket.name] = data
                else:
                    raise ValueError(f"Missing data from upstream node: {source_node.title}")

        # --- B. Parameters & Get Layer ---
        func_params = node.parameters.copy()
        
        if node.node_type == "get_layer":
            target_name = func_params.get("layer_name")
            # We assume reading data is thread-safe enough for read-only access
            if target_name in self.viewer.layers:
                data = self.viewer.layers[target_name].data
                return {"data_out": data} # Return immediately
            else:
                raise ValueError(f"Layer '{target_name}' not found.")

        # --- C. Import & Run ---
        if node.node_type not in library_def:
             raise ValueError(f"Unknown node type: {node.node_type}")
        def_data = library_def[node.node_type]
        
        # --- NEW: Input Validation ---
        validate_rules = def_data.get("validate_inputs", {})
        if validate_rules:
            for input_name, rules in validate_rules.items():
                data = func_inputs.get(input_name)
                
                # Check if required
                if rules.get("required") and data is None:
                    raise ValueError(f"❌ Validation failed for '{node.title}': input '{input_name}' is required but not connected")
                
                # Only validate further if data exists
                if data is not None:
                    # Check ndim
                    if "ndim" in rules and hasattr(data, "ndim"):
                        if data.ndim not in rules["ndim"]:
                            raise ValueError(
                                f"❌ Validation failed for '{node.title}': input '{input_name}' must be "
                                f"{rules['ndim']}D (got {data.ndim}D)"
                            )
                    
                    # Check dtype
                    if "dtype" in rules and hasattr(data, "dtype"):
                        # Normalize dtype to string for comparison
                        data_dtype_str = str(data.dtype)
                        allowed_dtypes = [str(d) for d in rules["dtype"]]
                        
                        # Use exact match to avoid false positives (e.g., "int8" matching "uint8")
                        if data_dtype_str not in allowed_dtypes:
                            raise ValueError(
                                f"❌ Validation failed for '{node.title}': input '{input_name}' dtype must be "
                                f"one of {rules['dtype']} (got {data.dtype})"
                            )
        
        if "executable" in def_data:
            func = def_data["executable"]
        else:
            exec_path = def_data["execution_path"] 
            module_name, func_name = exec_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)

        args = {**func_inputs, **func_params}
        result = func(**args)
        
        # --- D. Format Results ---
        output_names = def_data.get("outputs", ["out"])
        node_outputs = {}
        if isinstance(result, tuple):
             for i, name in enumerate(output_names):
                if i < len(result): node_outputs[name] = result[i]
        else:
             if output_names: node_outputs[output_names[0]] = result
        
        # --- NEW: Output Metadata Wrapping ---
        # Apply metadata to each output individually (after splitting tuple results)
        output_meta = def_data.get("output_meta")
        if output_meta:
            meta_with_sentinel = {**output_meta, "__napari_meta__": True}
            for out_name, out_data in node_outputs.items():
                # Check if this output is already wrapped with metadata
                is_envelope = (
                    isinstance(out_data, tuple) and 
                    len(out_data) == 2 and 
                    isinstance(out_data[1], dict) and
                    out_data[1].get("__napari_meta__") is True
                )
                if not is_envelope:
                    # Wrap this individual output with metadata
                    node_outputs[out_name] = (out_data, meta_with_sentinel)
              
        # Emit to GUI for display
        for out_name, out_data in node_outputs.items():
            self.result_signal.emit(node.title, out_name, out_data)
            
        return node_outputs

    def topological_sort(self, nodes):
        # Standard recursive sort
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