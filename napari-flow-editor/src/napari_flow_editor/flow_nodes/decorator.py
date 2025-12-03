import functools


def register_node(label, category, outputs=None, params_config=None):
    """
    Decorator to mark a function as a Flow Node.
    
    Args:
        label (str): The human-readable name (e.g., "Gaussian Blur").
        category (str): The submenu category (e.g., "Filters").
        outputs (list): List of output names (e.g., ["image_out"]).
        params_config (dict): Extra info for inputs that type hints can't cover 
                              (min, max, step). e.g., {'sigma': {'min': 0.1, 'max': 10.0}}
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
            # We will store the import path later during generation
        }
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator