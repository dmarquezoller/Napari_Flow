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
    interactive_request_signal = Signal(str, dict)  # (node_uid, interactive_config)
    interactive_response_signal = Signal(str, object)  # (node_uid, geometry_data)

    def __init__(self, scene, viewer, loop_config=None):
        super().__init__()
        self.scene = scene
        self.viewer = viewer
        self.loop_config = loop_config
        self.interactive_responses = {}  # Store interactive responses by node_uid 

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
            
            # Determine loop parameters
            if self.loop_config:
                loop_uids = set(self.loop_config.get("nodes", []))
                iterations = self.loop_config.get("iterations", 1)
            else:
                loop_uids = set()
                iterations = 1
            
            for iteration in range(iterations):
                # Log iteration if looping
                if iterations > 1:
                    self.log_signal.emit(f"🔁 Loop Iteration {iteration + 1}/{iterations}")
                
                # On iterations 2+, wipe cache of loop nodes so they re-execute
                if iteration > 0:
                    for node in sorted_nodes:
                        if node.uid in loop_uids:
                            node.last_signature = None
                            node.cached_results = {}
                
                # Execute all nodes in topological order
                for node in sorted_nodes:
                    try:
                        # 1. Calculate Signature
                        current_signature = self.calculate_signature(node)
                        
                        # 2. Check Cache (THE FIX)
                        # We strictly check the signature. 
                        # We do NOT check 'node.status' because the UI might have reset it to Gray.
                        # We do NOT check 'cached_results' too strictly to avoid false negatives.
                        if current_signature == node.last_signature:
                            self.log_signal.emit(f"Skipping: {node.title} (Cached)")
                            
                            # IMPORTANT: Force the UI to turn Green. 
                            # This fixes the issue where cached nodes looked "Pending/Gray".
                            self.node_status_signal.emit(node.uid, "green")
                            
                            continue # SKIP EXECUTION (Keep existing results)
                        
                        # 3. Execution (If we get here, cache missed)
                        self.node_status_signal.emit(node.uid, "yellow") # Turn Yellow
                        
                        # Run Logic
                        results = self.execute_node_logic(node, NODE_LIBRARY)
                        
                        # 4. Update Cache
                        node.cached_results = results
                        node.last_signature = current_signature
                        self.node_status_signal.emit(node.uid, "green") # Turn Green
                        
                    except Exception as e:
                        self.node_status_signal.emit(node.uid, "red")
                        # Log the specific error for easier debugging
                        print(f"Error in node {node.title}: {e}") 
                        raise e 

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

    def validate_node_inputs(self, func_inputs, validate_config):
        """
        Validate inputs before execution based on validate_inputs config.
        
        Args:
            func_inputs: Dict of input values
            validate_config: Dict of validation rules
            
        Raises:
            ValueError: If validation fails
        """
        for input_name, rules in validate_config.items():
            value = func_inputs.get(input_name)
            
            # Check required
            if rules.get("required", False) and value is None:
                raise ValueError(f"Required input '{input_name}' is missing")
            
            if value is not None:
                # Check dtype
                if "dtype" in rules:
                    allowed_dtypes = rules["dtype"]
                    if hasattr(value, "dtype"):
                        dtype_str = str(value.dtype)
                        # Check if any allowed dtype matches
                        if not any(dt in dtype_str for dt in allowed_dtypes):
                            raise ValueError(
                                f"Input '{input_name}' has dtype {value.dtype}, "
                                f"expected one of {allowed_dtypes}"
                            )
                
                # Check ndim
                if "ndim" in rules:
                    allowed_ndims = rules["ndim"]
                    if hasattr(value, "ndim"):
                        if value.ndim not in allowed_ndims:
                            raise ValueError(
                                f"Input '{input_name}' has {value.ndim} dimensions, "
                                f"expected one of {allowed_ndims}"
                            )

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
        
        # --- D. Input Validation ---
        if "validate_inputs" in def_data:
            self.validate_node_inputs(func_inputs, def_data["validate_inputs"])
        
        # --- E. Interactive Mode ---
        if "interactive" in def_data:
            interactive_config = def_data["interactive"]
            arg_name = interactive_config.get("arg_name", "roi_geometry")
            
            # Check if we already have a response for this node
            if node.uid not in self.interactive_responses:
                self.log_signal.emit(f"⏸️  Waiting for interactive input: {interactive_config.get('prompt', 'Draw on image')}")
                # Emit signal to UI and wait for response
                self.interactive_request_signal.emit(node.uid, interactive_config)
                # In a real implementation, we'd need to wait here
                # For now, we'll check if response is available
                # This is a simplified version - full implementation would use QEventLoop
            
            # Inject geometry into function inputs
            if node.uid in self.interactive_responses:
                func_inputs[arg_name] = self.interactive_responses[node.uid]

        # RUN
        args = {**func_inputs, **clean_params}
        result = func(**args)
        
        # --- F. Format Results ---
        # Check library def first, then fallback to node sockets
        output_names = def_data.get("outputs", [])
        if not output_names and hasattr(node, 'outputs'):
             output_names = list(node.outputs.keys()) if isinstance(node.outputs, dict) else [s.name for s in node.outputs]

        node_outputs = {}

        def wrap_result(res):
            """Wrap result with metadata, applying output_meta if defined."""
            # Start with current_metadata from inputs
            final_meta = current_metadata.copy()
            
            # If result already has metadata tuple, extract it
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
                data = res[0]
                final_meta.update(res[1])
            else:
                data = res
            
            # Apply output_meta from decorator if present
            if "output_meta" in def_data:
                output_meta_config = def_data["output_meta"]
                final_meta.update(output_meta_config)
                
                # Handle name_suffix
                if "name_suffix" in output_meta_config:
                    old_name = final_meta.get("name", node.title)
                    if output_meta_config["name_suffix"] not in old_name:
                        final_meta["name"] = f"{old_name} {output_meta_config['name_suffix']}"
            
            return (data, final_meta)

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