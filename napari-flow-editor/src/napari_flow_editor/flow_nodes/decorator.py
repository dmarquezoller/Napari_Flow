import functools
import dask.array as da
import numpy as np
import inspect  # <--- CRITICAL IMPORT

def register_node(label, category, outputs=None, params_config=None):
    """
    Decorator to mark a function as a Flow Node.
    """
    if outputs is None:
        outputs = ["out"]
    if params_config is None:
        params_config = {}

    def decorator(func):
        func._is_flow_node = True
        func._node_meta = {
            "label": label,
            "category": category,
            "outputs": outputs,
            "params_config": params_config,
        }
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def smart_compute(dask_func=None):
    """
    Traffic Controller:
    1. Finds the main input data (whether passed via args or kwargs).
    2. If List (Pyramid) -> Recurses for every level.
    3. If Dask -> Uses dask_func.
    4. If NumPy -> Uses standard func.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            
            # --- 1. FIND THE DATA ARGUMENT ---
            # We need to know which argument is the image/data. 
            # Usually it's the first parameter defined in the function.
            data = None
            param_name = None
            
            if args:
                # Easy case: It's the first positional argument
                data = args[0]
            else:
                # Hard case: It's buried in kwargs.
                # We use inspect to find the name of the first parameter (e.g., 'image')
                try:
                    sig = inspect.signature(func)
                    param_name = list(sig.parameters.keys())[0]
                    if param_name in kwargs:
                        data = kwargs[param_name]
                except Exception:
                    # If we can't find it, just pass through (e.g. function with no inputs)
                    pass

            # If we still have no data, execute normally
            if data is None:
                return func(*args, **kwargs)

            # --- 2. HANDLE PYRAMID (List of Arrays) ---
            if isinstance(data, list):
                # print(f"🔄 [Smart Compute] Splitting Pyramid for {func.__name__}...")
                output_pyramid = []
                for level_data in data:
                    # RECURSION: We call 'wrapper' again for this specific level.
                    # We must reconstruct the call exactly as it came in.
                    
                    if args:
                        # If called with args: (list, 1.0) -> (level_data, 1.0)
                        new_args = (level_data,) + args[1:]
                        output_pyramid.append(wrapper(*new_args, **kwargs))
                    else:
                        # If called with kwargs: {'image': list} -> {'image': level_data}
                        new_kwargs = kwargs.copy()
                        new_kwargs[param_name] = level_data
                        output_pyramid.append(wrapper(**new_kwargs))
                
                return output_pyramid

            # --- 3. HANDLE DASK (Lazy) ---
            if isinstance(data, da.Array):
                if dask_func is not None:
                    # print(f"⚡ [Smart Compute] Dask Lazy: {dask_func.__name__}")
                    return dask_func(*args, **kwargs)
                else:
                    print(f"⚠️ [Smart Compute] Input is Dask but no dask_func defined. Running eager.")
                    return func(*args, **kwargs)

            # --- 4. HANDLE NUMPY (RAM) ---
            # print(f"💾 [Smart Compute] NumPy Eager: {func.__name__}")
            return func(*args, **kwargs)
            
        return wrapper
    return decorator