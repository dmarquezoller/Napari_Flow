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
                        # Skip the ``interaction`` kwarg – it is injected by the
                        # engine at runtime for interactive nodes and must NOT
                        # appear as a user-facing parameter in the library.
                        if param_name == "interaction":
                            continue

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
                    entry = {
                        "label": meta["label"],
                        "category": meta["category"],
                        "inputs": inputs,
                        "outputs": meta["outputs"],
                        "parameters": parameters,
                        "execution_path": f"{full_module_name}.{name}"
                    }

                    # Persist interactive config so the engine knows at runtime
                    interactive_cfg = meta.get("interactive")
                    if interactive_cfg:
                        entry["interactive"] = interactive_cfg
                    logic_cfg = meta.get("logic")
                    if logic_cfg:
                        entry["logic"] = logic_cfg
                    input_types_cfg = meta.get("input_types")
                    if input_types_cfg:
                        entry["input_types"] = input_types_cfg
                    output_types_cfg = meta.get("output_types")
                    if output_types_cfg:
                        entry["output_types"] = output_types_cfg
                    dynamic_output_types_cfg = meta.get("dynamic_output_types")
                    if dynamic_output_types_cfg:
                        entry["dynamic_output_types"] = dynamic_output_types_cfg

                    library[node_key] = entry
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
