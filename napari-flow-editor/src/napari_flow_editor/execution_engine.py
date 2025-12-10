import importlib
import numpy as np
import traceback
from qtpy.QtCore import QObject, Signal

class ExecutionWorker(QObject):
    # Signals to talk to the Main Thread
    log_signal = Signal(str)                         # Send text logs
    result_signal = Signal(str, str, object)         # Send Result: (Node Title, Output Name, Data)
    finished_signal = Signal()                       # Done
    error_signal = Signal(str)                       # Error message

    def __init__(self, scene, viewer):
        super().__init__()
        self.scene = scene
        self.viewer = viewer # We only use this for READING "Get Layer" data (usually safe)
        self.results = {}

    def run(self):
        """Background execution loop."""
        try:
            self.log_signal.emit("--- Starting Execution ---")
            
            from .napari_plugin_v2 import Node, NODE_LIBRARY
            nodes = [item for item in self.scene.items() if isinstance(item, Node)]
            
            if not nodes:
                self.log_signal.emit("Pipeline is empty.")
                self.finished_signal.emit()
                return

            sorted_nodes = self.topological_sort(nodes)
            self.results.clear()
            
            for node in sorted_nodes:
                # We pass the library dict to avoid import issues
                self.execute_node(node, NODE_LIBRARY)
                
            self.log_signal.emit("--- Execution Finished ---")
            
        except Exception as e:
            # Format the full traceback so you see WHY it crashed in the console
            full_error = traceback.format_exc()
            self.log_signal.emit(f"CRITICAL ERROR:\n{full_error}")
            self.error_signal.emit(str(e))
        finally:
            self.finished_signal.emit()

    def topological_sort(self, nodes):
        # (Same logic as before)
        visited = set()
        stack = []
        def visit(n):
            if n in visited: return
            visited.add(n)
            for input_socket in n.inputs:
                if input_socket.connected_edges:
                    edge = input_socket.connected_edges[0]
                    if edge.start_socket:
                        visit(edge.start_socket.node)
            stack.append(n)
        for node in nodes:
            visit(node)
        return stack

    def execute_node(self, node, library_def):
        self.log_signal.emit(f"Running: {node.title}...")
        
        # --- A. Inputs ---
        func_inputs = {}
        for socket in node.inputs:
            if socket.connected_edges:
                edge = socket.connected_edges[0]
                source_node = edge.start_socket.node
                source_socket_name = edge.start_socket.name
                if source_node.uid in self.results:
                    data = self.results[source_node.uid].get(source_socket_name)
                    func_inputs[socket.name] = data
                else:
                    raise ValueError(f"Missing upstream data from {source_node.title}")

        # --- B. Parameters & Get Layer ---
        func_params = node.parameters.copy()
        
        if node.node_type == "get_layer":
            target_name = func_params.get("layer_name")
            # READ-ONLY access to viewer layers is usually fine.
            # WRITING (add_image) causes the crash.
            if target_name in self.viewer.layers:
                data = self.viewer.layers[target_name].data
                self.results[node.uid] = {"data_out": data}
                return
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

        args = {**func_inputs, **func_params}
        result = func(**args)
        
        # --- D. Store Results ---
        output_names = def_data.get("outputs", ["out"])
        node_outputs = {}
        if isinstance(result, tuple):
             for i, name in enumerate(output_names):
                if i < len(result): node_outputs[name] = result[i]
        else:
             if output_names: node_outputs[output_names[0]] = result
             
        self.results[node.uid] = node_outputs
        
        # --- E. SEND TO GUI (Do NOT add_image here!) ---
        for out_name, out_data in node_outputs.items():
            # Emit signal so Main Thread handles the Viewer
            self.result_signal.emit(node.title, out_name, out_data)