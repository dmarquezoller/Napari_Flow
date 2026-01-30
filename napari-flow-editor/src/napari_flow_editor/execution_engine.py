import importlib
import numpy as np
import traceback
import json
import hashlib
from qtpy.QtCore import QObject, Signal
import time

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
                    time.sleep(0.05) 
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
        Returns the dictionary of outputs as (Data, Metadata) tuples.
        """
        self.log_signal.emit(f"Executing: {node.title}...")
        
        # This dictionary will accumulate metadata from all parents
        current_metadata = {} 

        # --- A. Inputs (Grab from CACHE & Inherit Metadata) ---
        func_inputs = {}
        for socket in node.inputs:
            if socket.connected_edges:
                edge = socket.connected_edges[0]
                source_node = edge.start_socket.node
                source_socket_name = edge.start_socket.name
                
                if source_socket_name in source_node.cached_results:
                    # Get the package
                    data_package = source_node.cached_results[source_socket_name]
                    
                    # 1. Check if it is an Envelope (Data, Meta)
                    if isinstance(data_package, tuple) and len(data_package) == 2 and isinstance(data_package[1], dict):
                        # UNPACK: Separate data for calculation, keep meta for history
                        data_only = data_package[0]
                        incoming_meta = data_package[1]
                        
                        # Inherit the metadata
                        current_metadata.update(incoming_meta)
                        func_inputs[socket.name] = data_only
                        
                        # Debug
                        # print(f"DEBUG: Inherited metadata from {source_node.title}: {incoming_meta.keys()}")
                    else:
                        # Fallback for raw data (no metadata yet)
                        func_inputs[socket.name] = data_package
                else:
                    raise ValueError(f"Missing data from upstream node: {source_node.title}")

        # --- B. Parameters & Get Layer ---
        func_params = node.parameters.copy()
        
        if node.node_type == "get_layer":
            target_name = func_params.get("layer_name")
            if target_name in self.viewer.layers:
                layer = self.viewer.layers[target_name]
                
                # 1. Grab existing metadata safely
                # We copy it so we don't accidentally modify the real layer later
                layer_meta = layer.metadata.copy() if hasattr(layer, 'metadata') else {}
                
                # 2. Add Axes from Table (if exists)
                axis_map = func_params.get("axis_map", [])
                if axis_map:
                    row = axis_map[0]
                    axes = [row.get(f"d{i}") for i in range(5) if row.get(f"d{i}", "-") != "-"]
                    layer.metadata["axes"] = axes   

                # 3. Store the Source Name (Useful for tracking)
                layer_meta = layer.metadata.copy() if hasattr(layer, 'metadata') else {}
                layer_meta["source_layer"] = target_name

                # RETURN THE ENVELOPE: (Array, Dictionary)
                return {"data_out": (layer.data, layer_meta)}
            else:
                raise ValueError(f"Layer '{target_name}' not found.")

        # --- C. Import & Run ---
        if node.node_type not in library_def:
             raise ValueError(f"Unknown node type: {node.node_type}")
        def_data = library_def[node.node_type]
        
        if "executable" in def_data:
            func = def_data["executable"]
        else:
            exec_path = def_data["execution_path"] 
            module_name, func_name = exec_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)

        # Run the function with RAW arrays (extracted in Step A)
        args = {**func_inputs, **func_params}
        result = func(**args)
        
        # --- D. Format Results & Emit ---
        output_names = def_data.get("outputs", ["out"])
        node_outputs = {}

        # Helper: Wraps a result array into an Envelope (Result, Inherited_Metadata)
        def wrap_result(res):
            # If function returned its own metadata (rare), merge it
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
                merged = current_metadata.copy()
                merged.update(res[1])
                return (res[0], merged)
            # Otherwise, just attach the inherited metadata
            return (res, current_metadata)

        # 1. Map results to output names
        if isinstance(result, tuple) and len(output_names) > 1:
             for i, name in enumerate(output_names):
                if i < len(result): node_outputs[name] = wrap_result(result[i])
        elif isinstance(result, dict):
             for k, v in result.items():
                 node_outputs[k] = wrap_result(v)
        else:
             if output_names: node_outputs[output_names[0]] = wrap_result(result)
             
        # 2. Emit to GUI
        # We send the WHOLE ENVELOPE (Data, Meta) to the GUI
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