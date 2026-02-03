import importlib
import numpy as np
import traceback
import json
import hashlib
import sys
import time
from qtpy.QtCore import QObject, Signal

class ExecutionWorker(QObject):
    # Signals
    log_signal = Signal(str)
    node_status_signal = Signal(str, str)     # (UID, "green"/"yellow"/"red")
    result_signal = Signal(str, str, object)  # (Title, OutputName, Data)
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, scene, viewer, loop_config=None):
        super().__init__()
        self.scene = scene
        self.viewer = viewer
        self.loop_config = loop_config 

    def run(self):
        try:
            self.log_signal.emit("--- Starting Execution ---")
            
            from .napari_plugin_v2 import Node, NODE_LIBRARY
            nodes = [item for item in self.scene.items() if isinstance(item, Node)]
            
            if not nodes:
                self.log_signal.emit("Pipeline is empty.")
                self.finished_signal.emit()
                return

            sorted_nodes = self.topological_sort(nodes)
            
            # --- LOOP CONFIG ---
            loop_uids = set()
            iterations = 1
            if self.loop_config:
                loop_uids = set(self.loop_config["nodes"])
                iterations = self.loop_config["iterations"]
                self.log_signal.emit(f"Loop detected ({iterations} iters) on {len(loop_uids)} nodes.")

            # --- EXECUTION LOOP ---
            for current_iter in range(iterations):
                
                if iterations > 1:
                    self.log_signal.emit(f"=== Iteration {current_iter + 1}/{iterations} ===")

                for node in sorted_nodes:
                    # Skip non-loop nodes in subsequent iterations
                    if current_iter > 0 and node.uid not in loop_uids:
                        continue

                    try:
                        self.node_status_signal.emit(node.uid, "yellow")

                        # 1. Check Cache (Standard Logic)
                        current_signature = self.calculate_signature(node)
                        is_loop_node = node.uid in loop_uids
                        
                        if not is_loop_node and node.status == "green" and current_signature == node.last_signature:
                            self.log_signal.emit(f"Skipping: {node.title} (Cached)")
                            continue 
                        
                        # 2. EXECUTE (Using YOUR logic)
                        results = self.execute_node_logic(node, NODE_LIBRARY)
                        
                        # 3. Update State
                        node.cached_results = results
                        node.last_signature = current_signature
                        self.node_status_signal.emit(node.uid, "green")
                        
                        # Small pause for UI updates
                        if iterations > 1: time.sleep(0.05)

                    except Exception as e:
                        # Print error to terminal
                        print("\n" + "="*40)
                        print(f"!!! CRASH IN NODE: {node.title} !!!")
                        traceback.print_exc()
                        print("="*40 + "\n")
                        
                        self.node_status_signal.emit(node.uid, "red")
                        self.error_signal.emit(f"Error in {node.title}: {str(e)}")
                        return 

                # Reset visual state for loop nodes
                if current_iter < iterations - 1:
                    time.sleep(0.1)
                    for node in sorted_nodes:
                        if node.uid in loop_uids:
                            self.node_status_signal.emit(node.uid, "grey")

            self.log_signal.emit("--- Execution Finished ---")
            self.finished_signal.emit()

        except Exception as e:
            full_error = traceback.format_exc()
            self.log_signal.emit(f"CRITICAL ERROR:\n{full_error}")
            self.error_signal.emit(str(e))

    def calculate_signature(self, node):
        try:
            param_str = json.dumps(node.params, sort_keys=True, default=str)
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
        if hasattr(node, 'node_type') and node.node_type == "get_layer":
            target_name = func_params.get("layer_name")
            if target_name in self.viewer.layers:
                layer = self.viewer.layers[target_name]
                
                # Copy metadata
                layer_meta = layer.metadata.copy() if hasattr(layer, 'metadata') else {}
                
                # Axis Map Logic
                axis_map = func_params.get("axis_map", [])
                if axis_map:
                    row = axis_map[0]
                    axes = [row.get(f"d{i}") for i in range(5) if row.get(f"d{i}", "-") != "-"]
                    layer_meta["axes"] = "".join(axes) # Ensure string

                layer_meta["source_layer"] = target_name
                
                # Return the Envelope
                return {"data_out": (layer.data, layer_meta)}
            else:
                 # If layer missing, we can try waiting or fail gracefully
                 raise ValueError(f"Layer '{target_name}' not found.")

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

        # RUN
        args = {**func_inputs, **clean_params}
        result = func(**args)
        
        # --- D. Format Results ---
        # Check library def first, then fallback to node sockets
        output_names = def_data.get("outputs", [])
        if not output_names and hasattr(node, 'outputs'):
             output_names = list(node.outputs.keys()) if isinstance(node.outputs, dict) else [s.name for s in node.outputs]

        node_outputs = {}

        def wrap_result(res):
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
                merged = current_metadata.copy()
                merged.update(res[1])
                return (res[0], merged)
            return (res, current_metadata)

        if isinstance(result, tuple) and len(output_names) > 1:
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