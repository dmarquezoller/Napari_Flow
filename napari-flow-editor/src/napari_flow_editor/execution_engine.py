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
        
        # --- A. Resolve Inputs (Data from upstream nodes) ---
        func_inputs = {}
        
        for socket in node.inputs:
            if socket.connected_edges:
                # 1. Find the connection
                edge = socket.connected_edges[0]
                source_node = edge.start_socket.node
                source_socket_name = edge.start_socket.name
                
                # 2. Retrieve data from results cache
                if source_node.uid in self.results:
                    # We look up the specific output name (e.g., 'image_out')
                    data = self.results[source_node.uid].get(source_socket_name)
                    func_inputs[socket.name] = data
                else:
                    raise ValueError(f"Missing data from upstream node: {source_node.title}")

        # --- B. Prepare Parameters ---
        func_params = node.parameters.copy()
        
        # --- C. SPECIAL CASE: "Get Layer" Node ---
        # This node doesn't execute a Python function in the traditional sense.
        # It grabs data from the Napari Viewer.
        if node.title == "Get Layer":
            target_name = func_params.get("layer_name")
            
            if not target_name:
                raise ValueError("No layer selected in 'Get Layer' node.")

            if target_name in self.viewer.layers:
                layer_obj = self.viewer.layers[target_name]
                data = layer_obj.data
                
                # Store result immediately and exit this node
                # We use the key 'data_out' because that matches the decorator output name in inputs.py
                self.results[node.uid] = {"data_out": data}
                return 
            else:
                raise ValueError(f"Layer '{target_name}' not found in Napari. Did you delete it?")

        # --- D. Import and Run (Standard Nodes) ---
        
        # 1. Get definition from the Global Library
        from .napari_plugin_v2 import NODE_LIBRARY
        
        if node.node_type not in NODE_LIBRARY:
            raise ValueError(f"Unknown node type: {node.node_type}")
            
        def_data = NODE_LIBRARY[node.node_type]
        
        # 2. Determine how to run it
        if "executable" in def_data:
            # Case 1: Custom Node (User loaded a .py file) -> Use direct function object
            func = def_data["executable"]
        else:
            # Case 2: Standard Node -> Import dynamically via string path
            exec_path = def_data["execution_path"] 
            module_name, func_name = exec_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
        
        # 3. Combine Inputs and Parameters
        # Note: func_inputs keys must match function argument names!
        args = {**func_inputs, **func_params}
        
        # 4. EXECUTE
        result = func(**args)
        
        # --- E. Store Results ---
        output_names = def_data.get("outputs", ["out"])
        node_outputs = {}
        
        if isinstance(result, tuple):
            # Map tuple outputs to socket names in order
            for i, name in enumerate(output_names):
                if i < len(result):
                    node_outputs[name] = result[i]
        else:
            # Single output
            if output_names:
                node_outputs[output_names[0]] = result
        
        self.results[node.uid] = node_outputs
        
        # --- F. Display in Napari ---
        # We auto-display results to visualize the pipeline
        for out_name, out_data in node_outputs.items():
            # Only display if it looks like image data (NumPy array)
            if isinstance(out_data, np.ndarray):
                # Naming convention: "Node Name (socket name)"
                layer_name = f"{node.title} ({out_name})"
                
                # Check if layer exists to update it (prevents spamming new layers)
                try:
                    self.viewer.layers[layer_name].data = out_data
                    # Force refresh
                    self.viewer.layers[layer_name].refresh()
                except KeyError:
                    # Create new layer if it doesn't exist
                    # We assume it's an image. If it's a mask (int/bool), add_labels might be better,
                    # but add_image handles most things safely.
                    if out_data.dtype == bool or np.issubdtype(out_data.dtype, np.integer):
                         # Optional: Try to guess if it's labels or just int image
                         self.viewer.add_image(out_data, name=layer_name, interpolation2d='nearest')
                    else:
                         self.viewer.add_image(out_data, name=layer_name)