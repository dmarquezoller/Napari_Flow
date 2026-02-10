import json
import inspect
import os
import importlib
import pkgutil
import sys

# 1. Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "node_library.json")
PACKAGE_NAME = "napari_flow_editor.flow_nodes" # change here

# Ensure we can import the local package
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

def get_type_name(py_type):
    """Maps Python types to JSON schema types."""
    if py_type == float: return "float"
    if py_type == int: return "int"
    if py_type == bool: return "bool"
    if py_type == str: return "enum"
    return "string"

def generate():
    library = {}
    
    # 1. Import the package dynamically
    try:
        package = importlib.import_module(PACKAGE_NAME)
        importlib.reload(package)
    except ImportError as e:
        print(f"Error: Could not import '{PACKAGE_NAME}'. Make sure the folder exists.")
        return

    # 2. Iterate over all modules in the package
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        full_module_name = f"{PACKAGE_NAME}.{module_name}"
        
        try:
            module = importlib.import_module(full_module_name)
            importlib.reload(module) # Ensure we get the latest code
        except Exception as e:
            print(f"Skipping module '{module_name}' due to error: {e}")
            continue
            
        # 3. Inspect functions within the module
        for name, func in inspect.getmembers(module, inspect.isfunction):
            # Check for the @register_node marker
            if getattr(func, "_is_flow_node", False):
                try:
                    meta = func._node_meta
                    sig = inspect.signature(func)
                    
                    inputs = []
                    parameters = {}
                    
                    # Extract Parameters and Inputs from signature
                    for param_name, param in sig.parameters.items():
                        if param.default == inspect.Parameter.empty:
                            inputs.append(param_name)
                        else:
                            p_type = get_type_name(param.annotation)
                            extra_config = meta["params_config"].get(param_name, {})
                            
                            param_def = {
                                "type": p_type, 
                                "default": param.default
                            }
                            param_def.update(extra_config)
                            
                            # Fallback for enums without options
                            if param_def["type"] == "enum" and "options" not in param_def:
                                param_def["type"] = "string"

                            parameters[param_name] = param_def

                    # Register the node
                    node_key = name
                    node_def = {
                        "label": meta["label"],
                        "category": meta["category"],
                        "inputs": inputs,
                        "outputs": meta["outputs"],
                        "parameters": parameters,
                        "execution_path": f"{full_module_name}.{name}"
                    }
                    
                    # Add optional metadata fields only if present
                    if "interactive" in meta:
                        node_def["interactive"] = meta["interactive"]
                    if "output_meta" in meta:
                        node_def["output_meta"] = meta["output_meta"]
                    if "validate_inputs" in meta:
                        node_def["validate_inputs"] = meta["validate_inputs"]
                    if "icon" in meta:
                        node_def["icon"] = meta["icon"]
                    
                    # Handle doc field with fallback to function docstring
                    if "doc" in meta:
                        node_def["doc"] = meta["doc"]
                    elif func.__doc__:
                        node_def["doc"] = func.__doc__.strip()
                    
                    library[node_key] = node_def
                except Exception as e:
                    print(f"Error processing node function '{name}': {e}")

    # 4. Write to JSON
    try:
        with open(JSON_PATH, "w") as f:
            json.dump(library, f, indent=2)
        print(f"Successfully updated {JSON_PATH}")
    except Exception as e:
        print(f"Error writing JSON: {e}")

if __name__ == "__main__":
    generate()