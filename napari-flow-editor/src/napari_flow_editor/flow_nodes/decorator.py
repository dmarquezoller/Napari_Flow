import functools


def register_node(
    label,
    category,
    outputs=None,
    params_config=None,
    interactive=None,
    output_meta=None,
    validate_inputs=None,
    doc=None,
    icon=None,
):
    """
    Decorator to mark a function as a Flow Node.
    
    Args:
        label (str): The human-readable name (e.g., "Gaussian Blur").
        category (str): The submenu category (e.g., "Filters").
        outputs (list): List of output names (e.g., ["image_out"]).
        params_config (dict): Extra info for inputs that type hints can't cover 
                              (min, max, step). e.g., {'sigma': {'min': 0.1, 'max': 10.0}}
        interactive (dict): Configuration for napari interactivity. Keys:
                           - layer_type: "shapes" | "points" | "labels"
                           - tool: Which napari tool to activate (e.g., "rectangle")
                           - prompt: User-facing instruction message
                           - arg_name: Parameter name to receive geometry data
                           - confirm: Whether to show OK/Cancel dialog (default True)
        output_meta (dict): Declarative output metadata. Keys:
                           - layer_type: "image" | "labels" | "plot"
                           - colormap: Optional napari colormap
                           - opacity: Optional napari opacity
                           - name_suffix: Optional name override
        validate_inputs (dict): Input validation rules. Format:
                               {input_name: {"required": bool, "dtype": list, "ndim": list}}
        doc (str): Tooltip/help text for the node. Falls back to function docstring if None.
        icon (str): Visual icon (emoji or string) to display on the node.
    """
    if outputs is None:
        outputs = ["out"]
    if params_config is None:
        params_config = {}

    def decorator(func):
        # Mark the function so the generator knows to pick it up
        func._is_flow_node = True
        
        # Store metadata attached to the function object
        func._node_meta = {
            "label": label,
            "category": category,
            "outputs": outputs,
            "params_config": params_config,
            "interactive": interactive,
            "output_meta": output_meta,
            "validate_inputs": validate_inputs,
            "doc": doc,
            "icon": icon,
            # We will store the import path later during generation
        }
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator