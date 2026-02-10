import functools
import dask.array as da
import numpy as np
from typing import Callable, Optional

def register_node(label, category, outputs=None, params_config=None, interactive=False):
    """
    Decorator to mark a function as a Flow Node.
    
    Args:
        label: Display name for the node
        category: Category for grouping nodes
        outputs: List of output socket names
        params_config: Configuration for node parameters
        interactive: If True, marks this node as requiring user interaction
                    (e.g., drawing ROIs, clicking points). The execution engine
                    can use this flag to emit appropriate log messages.
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
            "interactive": interactive,
        }
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# --- THE DISPATCHER ---
def smart_compute(dask_func: Optional[Callable] = None, cuda_func: Optional[Callable] = None):
    def decorator(node_func):
        @functools.wraps(node_func)
        def wrapper(*args, **kwargs):
            
            # --- CONTEXT ---
            context = {
                "meta": {},
                "type": "image",
                "has_wrapper": False,
                "is_pyramid": False,
                "pyramid_loc": None 
            }

            # --- 1. UNWRAP LOGIC ---
            unwrapped_args = []
            unwrapped_kwargs = {}

            def unwrap(item):
                raw = item
                # A. OME-Zarr Layer List
                if isinstance(item, list) and len(item) > 0 and isinstance(item[0], tuple):
                    target = item[0]
                    # Find 'image' if possible, else take first
                    for layer in item:
                        if len(layer) >= 3 and layer[2] == 'image': target = layer; break
                    raw = target[0]
                    if not context["meta"]:
                        context["meta"] = target[1].copy() if len(target) > 1 else {}
                        context["type"] = target[2] if len(target) > 2 else "image"
                        context["has_wrapper"] = True
                
                # B. Napari Layer Tuple
                elif isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], dict):
                    raw = item[0]
                    if not context["meta"]:
                        context["meta"] = item[1].copy()
                        context["has_wrapper"] = True

                # C. Detect Pyramid
                if isinstance(raw, list) and len(raw) > 0:
                    first = raw[0]
                    if isinstance(first, (da.Array, np.ndarray)) or hasattr(first, 'shape'):
                        context["is_pyramid"] = True
                        return raw 

                return raw

            # Process Args & Kwargs
            for i, arg in enumerate(args):
                val = unwrap(arg)
                unwrapped_args.append(val)
                if isinstance(val, list) and context["is_pyramid"] and context["pyramid_loc"] is None:
                    context["pyramid_loc"] = ("arg", i)

            for k, v in kwargs.items():
                val = unwrap(v)
                unwrapped_kwargs[k] = val
                if isinstance(val, list) and context["is_pyramid"] and context["pyramid_loc"] is None:
                    context["pyramid_loc"] = ("kwarg", k)

            # --- 2. EXECUTION CORE ---
            def execute_core(args_in, kwargs_in):
                check_obj = None
                if args_in: check_obj = args_in[0]
                elif kwargs_in:
                    for v in kwargs_in.values():
                        if isinstance(v, (da.Array, np.ndarray)): check_obj = v; break
                
                is_dask = isinstance(check_obj, da.Array)
                if is_dask and dask_func: return dask_func(*args_in, **kwargs_in)
                else: return node_func(*args_in, **kwargs_in)

            # --- 3. RUN ---
            result = None
            if context["is_pyramid"] and context["pyramid_loc"] is not None:
                loc_type, loc_key = context["pyramid_loc"]
                pyramid_levels = unwrapped_args[loc_key] if loc_type == "arg" else unwrapped_kwargs[loc_key]
                pyramid_output = []
                for level_data in pyramid_levels:
                    current_args = list(unwrapped_args)
                    current_kwargs = unwrapped_kwargs.copy()
                    if loc_type == "arg": current_args[loc_key] = level_data
                    else: current_kwargs[loc_key] = level_data
                    pyramid_output.append(execute_core(current_args, current_kwargs))
                result = pyramid_output
            else:
                if context["is_pyramid"] and context["pyramid_loc"] is None:
                     for k, v in unwrapped_kwargs.items():
                         if isinstance(v, list): unwrapped_kwargs[k] = v[0]
                result = execute_core(unwrapped_args, unwrapped_kwargs)

            # --- 4. RE-WRAP & SANITIZE ---
            if context["has_wrapper"] and result is not None:
                
                # Check data type (Handle Pyramids too)
                check_res = result[0] if isinstance(result, list) else result
                
                # --- AUTO-CORRECT TYPE ---
                if hasattr(check_res, 'dtype') and check_res.dtype.kind == 'f':
                    # It's a Float -> Must be an Image
                    context["type"] = 'image'
                    
                    # --- SANITIZE METADATA ---
                    # Only keep Geometry. Discard "Labels" metadata (color_dict, etc)
                    safe_keys = {'scale', 'translate', 'rotate', 'shear', 'affine', 'opacity', 'blending', 'visible', 'metadata', 'name'}
                    context["meta"] = {k: v for k, v in context["meta"].items() if k in safe_keys}

                # --- AUTO-CONTRAST ---
                try:
                    # 1. Slice small corner
                    sample = check_res
                    if isinstance(sample, da.Array):
                        slices = tuple(slice(0, min(s, 512)) for s in sample.shape)
                        computed_chunk = sample[slices].compute()
                        c_min, c_max = float(computed_chunk.min()), float(computed_chunk.max())
                    else:
                        c_min, c_max = float(sample.min()), float(sample.max())
                    
                    # 2. Avoid flat contrast (0,0)
                    if c_max == c_min: c_max += 0.0001
                        
                    context["meta"]["contrast_limits"] = [c_min, c_max]
                
                except Exception:
                    # Fallback: Let Napari guess
                    context["meta"].pop("contrast_limits", None)

                # Rename
                old_name = context["meta"].get("name", "Layer")
                if "(Processed)" not in old_name:
                    context["meta"]["name"] = f"{old_name} (Processed)"
                

                return (result, context["meta"], context["type"])

            return result

        return wrapper
    return decorator