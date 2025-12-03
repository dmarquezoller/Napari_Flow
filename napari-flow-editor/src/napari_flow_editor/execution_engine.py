import importlib
import numpy as np
import napari

class ExecutionEngine:
    def __init__(self, scene, viewer: napari.Viewer):
        self.scene = scene
        self.viewer = viewer
        self.results = {} # Storage for node outputs: { "node_uid": { "output_name": data } }

    def run(self):
        print("--- Starting Execution ---")
        
        # 1. Get all nodes
        # We import Node locally to avoid circular imports
        from .napari_plugin_v2 import Node
        nodes = [item for item in self.scene.items() if isinstance(item, Node)]
        
        if not nodes:
            print("Pipeline is empty.")
            return

        # 2. Topological Sort (Order of execution)
        sorted_nodes = self.topological_sort(nodes)
        
        # 3. Execution Loop
        self.results.clear()
        
        for node in sorted_nodes:
            try:
                self.execute_node(node)
            except Exception as e:
                print(f"CRITICAL ERROR executing {node.title}: {e}")
                # Stop execution if a node fails
                return 

        print("--- Execution Finished ---")

    def topological_sort(self, nodes):
        """
        Orders nodes so that dependencies are calculated first.
        Using Depth First Search (DFS).
        """
        visited = set()
        stack = []
        
        def visit(n):
            if n in visited: return
            visited.add(n)
            
            # Visit all dependencies (nodes connected to inputs)
            for input_socket in n.inputs:
                if input_socket.connected_edges:
                    edge = input_socket.connected_edges[0]
                    if edge.start_socket:
                        source_node = edge.start_socket.node
                        visit(source_node)
            
            stack.append(n)

        for node in nodes:
            visit(node)
            
        return stack # Return executed order

    def execute_node(self, node):
        print(f"Running: {node.title}...")
        
        # A. Resolve Inputs
        # We need to find the data from the previous nodes' outputs
        func_inputs = {}
        
        for socket in node.inputs:
            if socket.connected_edges:
                # 1. Find who connects to us
                edge = socket.connected_edges[0]
                source_node = edge.start_socket.node
                source_socket_name = edge.start_socket.name
                
                # 2. Grab the data from our results cache
                if source_node.uid in self.results:
                    data = self.results[source_node.uid].get(source_socket_name)
                    func_inputs[socket.name] = data
                else:
                    raise ValueError(f"Missing data from upstream node: {source_node.title}")
            else:
                # SPECIAL CASE: "Get Active Layer"
                # If this is the input node, we grab data from Napari
                if node.title == "Get Active Layer":
                    active_layer = self.viewer.layers.selection.active
                    if active_layer:
                        # We inject it into the first argument
                        func_inputs[socket.name] = active_layer.data
                    else:
                        raise ValueError("No active layer selected in Napari!")

        # B. Prepare Parameters
        func_params = node.parameters.copy()
        
        # C. Import and Run
        # 1. Get execution path from the Library (stored in node_type usually, or we look it up)
        # We need to access the global library. 
        from .napari_plugin_v2 import NODE_LIBRARY
        
        if node.node_type not in NODE_LIBRARY:
            raise ValueError(f"Unknown node type: {node.node_type}")
            
        def_data = NODE_LIBRARY[node.node_type]

        if "executable" in def_data:
            func = def_data["executable"]
        else:
            # Standard path-based import (Built-in Nodes)
            exec_path = def_data["execution_path"] 
            module_name, func_name = exec_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
        
        # Execute Function
        # We combine inputs and params. 
        # Note: If function expects 'image', and socket is 'image_in', 
        # the argument names MUST match in the wrapper!
        args = {**func_inputs, **func_params}
        
        # Special handling for the Input Node wrapper which expects 'image_from_viewer'
        if node.title == "Get Active Layer" and "image_from_viewer" not in args:
             # Map the socket name (e.g. 'image') to function arg ('image_from_viewer') if needed
             # Or just pass positional if only one
             pass

        result = func(**args)
        
        # D. Store Results
        # If result is a tuple (multiple outputs), map to socket names
        output_names = def_data.get("outputs", ["out"])
        
        node_outputs = {}
        if isinstance(result, tuple):
            for i, name in enumerate(output_names):
                if i < len(result):
                    node_outputs[name] = result[i]
        else:
            # Single output
            if output_names:
                node_outputs[output_names[0]] = result
        
        self.results[node.uid] = node_outputs
        
        # E. Display in Napari
        # If this is a terminal node (no outputs connected) OR we just want to see everything
        # Let's add it to viewer.
        for out_name, out_data in node_outputs.items():
            if isinstance(out_data, np.ndarray):
                layer_name = f"{node.title} ({out_name})"
                
                # Check if layer exists to update it (avoid spamming layers)
                try:
                    self.viewer.layers[layer_name].data = out_data
                except KeyError:
                    self.viewer.add_image(out_data, name=layer_name)