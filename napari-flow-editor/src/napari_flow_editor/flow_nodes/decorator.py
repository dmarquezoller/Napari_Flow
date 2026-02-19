import functools
import dask.array as da
import numpy as np
from typing import Callable, Optional

def register_node(label, category, outputs=None, params_config=None, interactive=None):
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
            "interactive": bool(interactive),
        }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if interactive:
                layer_name = getattr(interactive, "layer_name", "---- DRAW CROP (Waiting...) ----")
                if hasattr(interactive, "setup_interaction"):
                    print(f">> Please draw a rectangle in the '{layer_name}' layer.")
                    drawing_event = interactive.setup_interaction(layer_name)
                    drawing_event.wait()
                    shapes_data = interactive.finish_interaction(layer_name)
                else:
                    shapes_data = interactive.setup(*args, **kwargs)
                    if hasattr(interactive, "finish"):
                        shapes_data = interactive.finish(shapes_data)

                kwargs["interaction"] = shapes_data

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
                    target = item[0];
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


def dispatch(
    default: Callable,
    dask_func: Optional[Callable] = None,
    cuda_func: Optional[Callable] = None,
    args=(),
    kwargs=None,
    pyramid_strategy: str = "per_level",  # "per_level" | "from_level0"
):
    """
    Choose backend (cuda > dask > default) and handle pyramids.

    - If any input is a pyramid (list of arrays), we support:
        * per_level: run the operation independently on each level
        * from_level0: compute only on level0, downsample to other levels

    This expects image-like arrays (numpy/dask). If you pass napari LayerDataTuples
    or (data, meta) wrappers, it will unwrap them.
    """
    import numpy as np
    import dask.array as da

    if kwargs is None:
        kwargs = {}

    # ---------- Unwrap helpers ----------
    def unwrap(x):
        # list of LayerDataTuples -> pick first image layer
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], tuple):
            target = x[0]
            for layer in x:
                if len(layer) >= 3 and str(layer[2]) == "image":
                    target = layer
                    break
            return target[0]
        # (data, meta) envelope
        if isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], dict):
            return x[0]
        return x

    uargs = [unwrap(a) for a in args]
    ukwargs = {k: unwrap(v) for k, v in kwargs.items()}

    # ---------- Find pyramid location ----------
    pyramid_loc = None  # ("arg", idx) or ("kwarg", key)
    for i, a in enumerate(uargs):
        if isinstance(a, list) and len(a) > 0:
            pyramid_loc = ("arg", i)
            break
    if pyramid_loc is None:
        for k, v in ukwargs.items():
            if isinstance(v, list) and len(v) > 0:
                pyramid_loc = ("kwarg", k)
                break

    # ---------- Backend selection ----------
    def pick_backend(sample):
        # CUDA placeholder (optional later)
        if cuda_func is not None:
            try:
                import cupy as cp  # type: ignore
                if isinstance(sample, cp.ndarray):
                    return cuda_func
            except Exception:
                pass

        if dask_func is not None and isinstance(sample, da.Array):
            return dask_func

        return default

    def execute_once(args_in, kwargs_in):
        sample = None
        # pick the first array-like argument we see
        for v in list(args_in) + list(kwargs_in.values()):
            if isinstance(v, (da.Array, np.ndarray)) or hasattr(v, "shape"):
                sample = v
                break
        func = pick_backend(sample)
        return func(*args_in, **kwargs_in)

    # ---------- Downsample helper for from_level0 ----------
    def downsample_to_shape(arr0, target_shape):
        """
        Downsample arr0 to match target_shape on the last two dims (Y,X).
        Uses dask.coarsen(mean) when possible.
        """
        import numpy as np
        import dask.array as da

        if not (hasattr(arr0, "shape") and len(arr0.shape) >= 2):
            raise ValueError("downsample_to_shape expects an array with >=2 dims")

        y0, x0 = arr0.shape[-2], arr0.shape[-1]
        yt, xt = target_shape[-2], target_shape[-1]

        # if shapes already match
        if y0 == yt and x0 == xt:
            return arr0

        # compute integer factors if possible
        fy = int(y0 // yt) if yt else 1
        fx = int(x0 // xt) if xt else 1
        if fy < 1: fy = 1
        if fx < 1: fx = 1

        # Dask path (lazy)
        if isinstance(arr0, da.Array):
            factors = {arr0.ndim - 2: fy, arr0.ndim - 1: fx}
            out = da.coarsen(np.mean, arr0, factors, trim_excess=True)
            # If trim_excess gave slightly different shape, crop to exact target
            slicer = [slice(None)] * out.ndim
            slicer[-2] = slice(0, yt)
            slicer[-1] = slice(0, xt)
            return out[tuple(slicer)]

        # Numpy fallback (eager) — keep simple
        # (You’re mostly on dask for OME-Zarr, so this rarely runs.)
        from skimage.transform import resize
        out = resize(arr0, (*arr0.shape[:-2], yt, xt), preserve_range=True, anti_aliasing=True)
        return out.astype(arr0.dtype, copy=False)

    # ---------- Pyramid execution ----------
    if pyramid_loc is None:
        return execute_once(uargs, ukwargs)

    loc_type, loc_key = pyramid_loc
    pyramid = uargs[loc_key] if loc_type == "arg" else ukwargs[loc_key]
    if not isinstance(pyramid, list) or len(pyramid) == 0:
        return execute_once(uargs, ukwargs)

    if pyramid_strategy not in ("per_level", "from_level0"):
        raise ValueError(f"Unknown pyramid_strategy={pyramid_strategy!r}")

    if pyramid_strategy == "per_level":
        out_levels = []
        for level in pyramid:
            a2 = list(uargs)
            k2 = dict(ukwargs)
            if loc_type == "arg":
                a2[loc_key] = level
            else:
                k2[loc_key] = level
            out_levels.append(execute_once(a2, k2))
        return out_levels

    # pyramid_strategy == "from_level0"
    # 1) compute on highest-res level only
    level0 = pyramid[0]
    a2 = list(uargs)
    k2 = dict(ukwargs)
    if loc_type == "arg":
        a2[loc_key] = level0
    else:
        k2[loc_key] = level0

    out0 = execute_once(a2, k2)

    # 2) downsample out0 to each target level shape
    out_levels = [out0]
    for lvl in pyramid[1:]:
        if not hasattr(lvl, "shape"):
            raise ValueError(f"Pyramid level has no shape: {type(lvl)}")
        out_levels.append(downsample_to_shape(out0, lvl.shape))

    return out_levels
