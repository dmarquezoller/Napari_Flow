import functools
import inspect
import dask.array as da
import numpy as np
import threading
from typing import Callable, Optional

def _normalize_logic_config(logic):
    # Backward-compatible defaults: every node has both logic sockets and
    # logic sockets accept multiple links unless explicitly constrained.
    cfg = {
        "in": True,
        "out": True,
        "allow_multi_in": True,
        "allow_multi_out": True,
    }
    if isinstance(logic, dict):
        cfg.update(logic)
    elif logic is False:
        cfg["in"] = False
        cfg["out"] = False
    return cfg

def _normalize_io_type_map(type_map):
    """
    Normalize optional socket-type maps used for typed data sockets.

    Expected shape:
      {"socket_name": "image" | "labels" | "table" | ...}
    """
    if not isinstance(type_map, dict):
        return {}

    normalized = {}
    for socket_name, socket_type in type_map.items():
        key = str(socket_name)
        value = str(socket_type).strip().lower() if socket_type is not None else "any"
        normalized[key] = value if value else "any"
    return normalized


def _normalize_dynamic_output_types(config):
    """
    Normalize optional dynamic output type rules.

    Expected shape:
      {
        "socket_name": {
          "from_param": "layer_name",
          "source": "viewer_layer_type",
          "fallback": "any",
        }
      }
    """
    if not isinstance(config, dict):
        return {}

    normalized = {}
    for socket_name, rule in config.items():
        if not isinstance(rule, dict):
            continue
        from_param = rule.get("from_param")
        if not from_param:
            continue

        source_raw = rule.get("source", "viewer_layer_type")
        source = str(source_raw).strip().lower() if source_raw is not None else "viewer_layer_type"
        if not source:
            source = "viewer_layer_type"

        fallback_map = _normalize_io_type_map({"fallback": rule.get("fallback", "any")})
        normalized[str(socket_name)] = {
            "from_param": str(from_param),
            "source": source,
            "fallback": fallback_map["fallback"],
        }

    return normalized


def _normalize_description(description):
    if description is None:
        return ""
    value = str(description).strip()
    return value


_DISPATCH_CONTEXT = threading.local()


def _get_dispatch_stack():
    stack = getattr(_DISPATCH_CONTEXT, "stack", None)
    if stack is None:
        stack = []
        _DISPATCH_CONTEXT.stack = stack
    return stack


def push_dispatch_context(context: dict):
    stack = _get_dispatch_stack()
    stack.append(context or {})
    return len(stack)


def pop_dispatch_context(_token=None):
    stack = _get_dispatch_stack()
    if stack:
        stack.pop()


def get_dispatch_context():
    stack = _get_dispatch_stack()
    return stack[-1] if stack else {}


def register_node(
    label,
    category,
    outputs=None,
    params_config=None,
    description=None,
    interactive=None,
    logic=None,
    input_types=None,
    output_types=None,
    dynamic_output_types=None,
):
    """
    Decorator to mark a function as a Flow Node.

    ``interactive`` accepts:
      - ``True``  → shorthand for ``{"layer_type": "shapes"}``
      - a dict    → full config, e.g.
            ``{"layer_type": "shapes", "mode": "add_rectangle",
               "edge_color": "#00ff00", "prompt": "Draw a rectangle, then click Run"}``
      - ``None`` / ``False`` → non-interactive node (default)

    When an interactive node is executed the engine will:
      1. Create a temporary napari layer (shapes by default).
      2. Open a dialog with a **Run** button.
      3. Block until the user clicks Run.
      4. Pass the drawn data to the node function via the ``interaction`` kwarg.
    """
    if outputs is None:
        outputs = ["out"]
    if params_config is None:
        params_config = {}
    logic_config = _normalize_logic_config(logic)
    input_type_map = _normalize_io_type_map(input_types)
    output_type_map = _normalize_io_type_map(output_types)
    dynamic_output_type_rules = _normalize_dynamic_output_types(dynamic_output_types)

    # --- Normalise interactive config ---------------------------------
    if interactive is True:
        interactive_config = {
            "interaction_type": "shapes",
            "layer_type": "shapes",
        }
    elif isinstance(interactive, dict):
        interactive_config = interactive.copy()
        interaction_type = str(
            interactive_config.get("interaction_type", "")
        ).strip().lower()
        if not interaction_type:
            interaction_type = "shapes"
            interactive_config["interaction_type"] = interaction_type
        if interaction_type == "shapes":
            interactive_config.setdefault("layer_type", "shapes")
    else:
        interactive_config = None

    def decorator(func):
        resolved_description = _normalize_description(description)
        if not resolved_description:
            resolved_description = inspect.getdoc(func) or ""

        func._is_flow_node = True
        func._node_meta = {
            "label": label,
            "category": category,
            "outputs": outputs,
            "params_config": params_config,
            "description": resolved_description,
            # Store the full config (or None); the engine / generate_library
            # will serialise this into the JSON library.
            "interactive": interactive_config,
            "logic": logic_config,
            "input_types": input_type_map,
            "output_types": output_type_map,
            "dynamic_output_types": dynamic_output_type_rules,
        }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # The actual interaction is now driven by the execution engine,
            # which injects the ``interaction`` kwarg before calling us.
            # Nothing to do here — just forward the call.
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
    layout_policy: str = "spatial_only",  # "spatial_only" | "full_nd"
    time_policy: str = "independent",      # "independent" | "joint" | "reject"
    channel_policy: str = "independent",   # "independent" | "joint" | "reject"
    dask_strategy: Optional[str] = None,   # None | "pointwise" | "neighborhood"
    dask_halo_from_param: Optional[str] = None,
    dask_halo_factor: float = 4.0,
    dask_boundary_from_param: Optional[str] = None,
    dask_output_dtype=None,
    pyramid_param_policy: Optional[dict] = None,
    numpy_to_dask_chunks="auto",
):
    """
    Choose backend (cuda > dask > default), handle pyramids and apply
    layout-aware execution policies.

    Layout defaults:
      - time dims (T): independent (no filtering across time)
      - channel dims (C): independent (no channel mixing)
      - volumetric data (ZYX): processed as true 3D
    """
    import numpy as np
    import dask.array as da

    if kwargs is None:
        kwargs = {}

    runtime_axes = None
    runtime_axis_labels = None
    runtime_layout_kind = None

    ctx = get_dispatch_context()
    ctx_meta = (ctx.get("metadata", {}) if isinstance(ctx, dict) else {}) or {}
    ctx_axes = ctx_meta.get("axes")
    ctx_axis_labels = ctx_meta.get("axis_labels")
    ctx_layout_kind = ctx_meta.get("layout_kind")

    layout_to_axes = {
        "2d_image": "YX",
        "2d_image_channels": "YXC",
        "3d_image": "ZYX",
        "3d_image_channels": "ZYXC",
        "3d_timeline": "TYX",
        "3d_timeline_channels": "TYXC",
        "4d_timeline": "TZYX",
        "4d_timeline_channels": "TZYXC",
    }

    def is_array_like(x):
        return isinstance(x, (da.Array, np.ndarray)) or hasattr(x, "shape")

    def _looks_like_napari_multiscale(x):
        t = type(x)
        name = getattr(t, "__name__", "")
        module = getattr(t, "__module__", "")
        if name == "MultiScaleData" or "multiscale" in module.lower():
            return True

        # Defensive fallback for sequence wrappers exposing per-level shapes.
        shapes = getattr(x, "shapes", None)
        if shapes is not None:
            try:
                return len(shapes) > 1
            except Exception:
                return False
        return False

    def is_pyramid_like(x):
        """
        True for sequence-like multiscale containers (e.g. napari MultiScaleData)
        whose items are array-like levels.
        """
        if isinstance(x, (np.ndarray, da.Array, tuple, str, bytes)):
            return False
        # napari MultiScaleData may expose .shape; keep it eligible.
        if hasattr(x, "shape") and not _looks_like_napari_multiscale(x):
            return False
        try:
            n = len(x)
            if n <= 0:
                return False
            first = x[0]
        except Exception:
            return False
        return is_array_like(first)

    def axis_labels_to_axes(labels, ndim):
        if labels is None:
            return None
        try:
            seq = list(labels)
        except Exception:
            return None
        if len(seq) != ndim:
            return None

        mapped = []
        for raw in seq:
            token = str(raw).strip().lower()
            if token in ("t", "time"):
                mapped.append("T")
            elif token in ("z", "depth"):
                mapped.append("Z")
            elif token in ("c", "ch", "channel", "channels"):
                mapped.append("C")
            elif token in ("y",):
                mapped.append("Y")
            elif token in ("x",):
                mapped.append("X")
            else:
                return None
        return "".join(mapped)

    def infer_axes_for_sample(sample, axes_hint=None, axis_labels_hint=None, layout_hint=None):
        if sample is None or not hasattr(sample, "shape"):
            return None
        ndim = len(sample.shape)
        if isinstance(axes_hint, str):
            ax = axes_hint.upper()
            if len(ax) == ndim:
                return ax
        labels_ax = axis_labels_to_axes(axis_labels_hint, ndim)
        if labels_ax:
            return labels_ax
        if isinstance(layout_hint, str):
            mapped = layout_to_axes.get(layout_hint.lower())
            if mapped and len(mapped) == ndim:
                return mapped

        if ndim == 2:
            return "YX"
        if ndim == 3:
            # Default to volumetric when unknown (safer for microscopy stacks).
            return "ZYX"
        if ndim == 4:
            # Heuristic: channels usually in the last axis if small.
            if sample.shape[-1] <= 4:
                return "ZYXC"
            return "TZYX"
        if ndim == 5:
            return "TZYXC"
        return None

    # ---------- Unwrap helpers ----------
    def unwrap(x):
        nonlocal runtime_axes, runtime_axis_labels, runtime_layout_kind
        # list of LayerDataTuples -> pick first image layer
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], tuple):
            target = x[0]
            for layer in x:
                if len(layer) >= 3 and str(layer[2]) == "image":
                    target = layer
                    break
            if (
                len(target) >= 2
                and isinstance(target[1], dict)
                and runtime_axes is None
            ):
                runtime_axes = target[1].get("axes")
                runtime_axis_labels = target[1].get("axis_labels")
                runtime_layout_kind = target[1].get("layout_kind")
            data0 = target[0]
            if is_pyramid_like(data0):
                return list(data0)
            return data0
        # generic pyramid container (napari MultiScaleData, list, etc)
        if is_pyramid_like(x):
            return list(x)
        # (data, meta) envelope
        if isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], dict):
            if runtime_axes is None:
                runtime_axes = x[1].get("axes")
                runtime_axis_labels = x[1].get("axis_labels")
                runtime_layout_kind = x[1].get("layout_kind")
            data0 = x[0]
            if is_pyramid_like(data0):
                return list(data0)
            return data0
        return x

    uargs = [unwrap(a) for a in args]
    ukwargs = {k: unwrap(v) for k, v in kwargs.items()}
    if runtime_axes is None:
        runtime_axes = ctx_axes
    if runtime_axis_labels is None:
        runtime_axis_labels = ctx_axis_labels
    if runtime_layout_kind is None:
        runtime_layout_kind = ctx_layout_kind

    # ---------- Find pyramid location ----------
    pyramid_loc = None  # ("arg", idx) or ("kwarg", key)
    for i, a in enumerate(uargs):
        if is_pyramid_like(a):
            pyramid_loc = ("arg", i)
            break
    if pyramid_loc is None:
        for k, v in ukwargs.items():
            if is_pyramid_like(v):
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

    def has_dask_backend():
        return dask_func is not None or dask_strategy in ("pointwise", "neighborhood")

    def maybe_promote_numpy_to_dask(args_in, kwargs_in, sample):
        if (
            not has_dask_backend()
            or not isinstance(sample, np.ndarray)
        ):
            return args_in, kwargs_in, sample

        def convert(v):
            if isinstance(v, np.ndarray):
                return da.from_array(v, chunks=numpy_to_dask_chunks)
            return v

        promoted_args = [convert(v) for v in args_in]
        promoted_kwargs = {k: convert(v) for k, v in kwargs_in.items()}

        promoted_sample = sample
        for v in promoted_args:
            if isinstance(v, da.Array):
                promoted_sample = v
                break
        if not isinstance(promoted_sample, da.Array):
            for v in promoted_kwargs.values():
                if isinstance(v, da.Array):
                    promoted_sample = v
                    break

        return promoted_args, promoted_kwargs, promoted_sample

    def _replace_sample(args_in, kwargs_in, sample_loc, sample_value):
        a2 = list(args_in)
        k2 = dict(kwargs_in)
        if sample_loc is not None:
            loc_type, loc_key = sample_loc
            if loc_type == "arg":
                a2[loc_key] = sample_value
            else:
                k2[loc_key] = sample_value
        return a2, k2

    def _boundary_for_overlap(kwargs_in):
        if dask_boundary_from_param:
            mode = kwargs_in.get(dask_boundary_from_param, "reflect")
        else:
            mode = "reflect"

        mode_value = str(mode).strip().lower()
        if mode_value == "wrap":
            return "periodic"
        if mode_value == "constant":
            return kwargs_in.get("cval", 0)
        if mode_value == "mirror":
            return "reflect"
        if mode_value in ("nearest", "reflect", "periodic", "none"):
            return mode_value
        return "reflect"

    def _depth_for_overlap(sample_arr, axes, kwargs_in):
        if not dask_halo_from_param:
            return tuple(0 for _ in range(sample_arr.ndim))

        halo_source = kwargs_in.get(dask_halo_from_param, 0)
        if np.isscalar(halo_source):
            seq = [float(halo_source)] * sample_arr.ndim
        else:
            seq = [float(x) for x in list(halo_source)]
            if len(seq) == 1:
                seq = seq * sample_arr.ndim
            elif len(seq) < sample_arr.ndim:
                seq = seq + [seq[-1]] * (sample_arr.ndim - len(seq))
            elif len(seq) > sample_arr.ndim:
                seq = seq[: sample_arr.ndim]

        radii = [max(0, int(np.ceil(abs(v) * float(dask_halo_factor)))) for v in seq]

        if isinstance(axes, str) and len(axes) == sample_arr.ndim:
            spatial = {i for i, a in enumerate(axes) if a in ("Z", "Y", "X")}
            radii = [r if i in spatial else 0 for i, r in enumerate(radii)]

        # Avoid overlap depth > axis size (common failure on small Z/C dims)
        return tuple(min(r, max(0, int(sz) - 1)) for r, sz in zip(radii, sample_arr.shape))

    def _execute_auto_dask(args_in, kwargs_in, sample_loc, axes):
        if dask_strategy not in ("pointwise", "neighborhood"):
            raise ValueError(f"Unknown dask_strategy={dask_strategy!r}")
        sample_exec = None
        if sample_loc is not None:
            loc_type, loc_key = sample_loc
            sample_exec = args_in[loc_key] if loc_type == "arg" else kwargs_in.get(loc_key)
        if not isinstance(sample_exec, da.Array):
            raise TypeError("Auto dask strategy requires a dask array sample.")

        if dask_strategy == "pointwise":
            def block_apply(block):
                a2, k2 = _replace_sample(args_in, kwargs_in, sample_loc, block)
                return default(*a2, **k2)

            out_dtype = dask_output_dtype if dask_output_dtype is not None else sample_exec.dtype
            return da.map_blocks(block_apply, sample_exec, dtype=out_dtype)

        # dask_strategy == "neighborhood"
        def overlap_apply(block):
            a2, k2 = _replace_sample(args_in, kwargs_in, sample_loc, block)
            return default(*a2, **k2)

        depth = _depth_for_overlap(sample_exec, axes, kwargs_in)
        boundary = _boundary_for_overlap(kwargs_in)
        out_dtype = dask_output_dtype if dask_output_dtype is not None else sample_exec.dtype
        return sample_exec.map_overlap(
            overlap_apply,
            depth=depth,
            boundary=boundary,
            dtype=out_dtype,
        )

    def execute_once(args_in, kwargs_in):
        sample = None
        sample_loc = None
        # pick the first array-like argument we see
        for i, v in enumerate(args_in):
            if is_array_like(v):
                sample = v
                sample_loc = ("arg", i)
                break
        if sample is None:
            for k, v in kwargs_in.items():
                if is_array_like(v):
                    sample = v
                    sample_loc = ("kwarg", k)
                    break

        args_exec, kwargs_exec, sample_exec = maybe_promote_numpy_to_dask(
            args_in, kwargs_in, sample
        )

        axes = infer_axes_for_sample(
            sample_exec,
            axes_hint=runtime_axes,
            axis_labels_hint=runtime_axis_labels,
            layout_hint=runtime_layout_kind,
        )

        auto_dask = (
            isinstance(sample_exec, da.Array)
            and dask_func is None
            and dask_strategy in ("pointwise", "neighborhood")
        )
        if auto_dask:
            def run_now(a_now, k_now):
                return _execute_auto_dask(a_now, k_now, sample_loc, axes)
        else:
            func = pick_backend(sample_exec)

            def run_now(a_now, k_now):
                return func(*a_now, **k_now)

        # Fast path: no layout policy or no inferable axes.
        if (
            sample_exec is None
            or layout_policy == "full_nd"
            or axes is None
            or len(axes) != len(sample_exec.shape)
        ):
            return run_now(args_exec, kwargs_exec)

        spatial = [i for i, a in enumerate(axes) if a in ("Z", "Y", "X")]
        if not spatial:
            return run_now(args_exec, kwargs_exec)

        if time_policy == "reject" and "T" in axes:
            raise ValueError("Dispatch rejected time dimension (T) for this node.")
        if channel_policy == "reject" and "C" in axes:
            raise ValueError("Dispatch rejected channel dimension (C) for this node.")

        core = list(spatial)
        if time_policy == "joint":
            core.extend(i for i, a in enumerate(axes) if a == "T")
        if channel_policy == "joint":
            core.extend(i for i, a in enumerate(axes) if a == "C")
        core = sorted(set(core))
        independent = [i for i in range(len(axes)) if i not in core]

        # Nothing to split => run once on full N-D sample.
        if not independent:
            return run_now(args_exec, kwargs_exec)

        independent_shape = tuple(sample_exec.shape[i] for i in independent)

        def slice_if_compatible(value, slicer):
            if not is_array_like(value):
                return value
            if not hasattr(value, "shape") or len(value.shape) != len(sample_exec.shape):
                return value
            for d in independent:
                if value.shape[d] != sample_exec.shape[d]:
                    return value
            return value[tuple(slicer)]

        outputs = []
        for idx in np.ndindex(*independent_shape):
            slicer = [slice(None)] * len(sample.shape)
            for local_i, axis_i in enumerate(independent):
                slicer[axis_i] = idx[local_i]

            a2 = [slice_if_compatible(v, slicer) for v in args_exec]
            k2 = {k: slice_if_compatible(v, slicer) for k, v in kwargs_exec.items()}
            outputs.append(run_now(a2, k2))

        if not outputs:
            return run_now(args_exec, kwargs_exec)

        first = outputs[0]
        if not is_array_like(first):
            # Non-array outputs are not composable here; return per-slice list.
            return outputs

        if isinstance(first, da.Array):
            stacked = da.stack(outputs, axis=0)
            core_shape = outputs[0].shape
        else:
            stacked = np.stack(outputs, axis=0)
            core_shape = outputs[0].shape

        stacked = stacked.reshape(*independent_shape, *core_shape)

        # Reorder from [independent..., core...] back to original axis order.
        perm = []
        for orig_dim in range(len(axes)):
            if orig_dim in independent:
                perm.append(independent.index(orig_dim))
            else:
                perm.append(len(independent) + core.index(orig_dim))

        if isinstance(stacked, da.Array):
            return stacked.transpose(tuple(perm))
        return np.transpose(stacked, axes=tuple(perm))

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
    pyramid_raw = uargs[loc_key] if loc_type == "arg" else ukwargs[loc_key]
    if not is_pyramid_like(pyramid_raw):
        return execute_once(uargs, ukwargs)
    pyramid = list(pyramid_raw)

    if pyramid_strategy not in ("per_level", "from_level0"):
        raise ValueError(f"Unknown pyramid_strategy={pyramid_strategy!r}")

    def _spatial_indices_for_shape(shape, axes_hint):
        ndim = len(shape)
        if isinstance(axes_hint, str) and len(axes_hint) == ndim:
            idx = [i for i, a in enumerate(axes_hint) if a in ("Z", "Y", "X")]
            if idx:
                return idx
        # Fallback: last two dims are spatial
        if ndim >= 2:
            return [ndim - 2, ndim - 1]
        return list(range(ndim))

    def _compute_level_scale_context(base_level, level):
        if not (hasattr(base_level, "shape") and hasattr(level, "shape")):
            return None
        base_shape = tuple(int(s) for s in base_level.shape)
        lvl_shape = tuple(int(s) for s in level.shape)
        if len(base_shape) != len(lvl_shape):
            return None

        base_axes = infer_axes_for_sample(
            base_level,
            axes_hint=runtime_axes,
            axis_labels_hint=runtime_axis_labels,
            layout_hint=runtime_layout_kind,
        )
        spatial_idx = _spatial_indices_for_shape(base_shape, base_axes)
        if not spatial_idx:
            return None

        axis_factors = {}
        for i in spatial_idx:
            b = max(1.0, float(base_shape[i]))
            l = max(1.0, float(lvl_shape[i]))
            axis_factors[i] = max(1.0, b / l)
        return {"axis_factors": axis_factors, "spatial_idx": spatial_idx, "ndim": len(base_shape)}

    def _rebuild_like(original, values):
        if isinstance(original, tuple):
            return tuple(values)
        if isinstance(original, list):
            return list(values)
        if isinstance(original, np.ndarray):
            return np.asarray(values)
        return values

    def _scale_value_fixed_world(value, ctx):
        if ctx is None:
            return value
        axis_factors = ctx["axis_factors"]
        spatial_idx = ctx["spatial_idx"]
        ndim = ctx["ndim"]

        # Scalar sigma -> divide by average spatial scale factor.
        if np.isscalar(value):
            # Use only axes that actually downsampled when present.
            # Example: (T,Y,X) or (Z,Y,X) pyramids often keep the first axis
            # unchanged while downsampling Y/X by 2, so factors are (1,2,2).
            # In that case scalar sigma should scale by 2, not by mean(1,2,2).
            scaled_axes = [axis_factors[i] for i in spatial_idx if axis_factors[i] > 1.0 + 1e-9]
            if not scaled_axes:
                scaled_axes = [axis_factors[i] for i in spatial_idx]
            avg_factor = float(np.mean(scaled_axes))
            return float(value) / avg_factor

        # Sequence sigma:
        # - len == ndim: scale spatial positions by their corresponding axis factor.
        # - len == n_spatial: scale each item by spatial factor order.
        if isinstance(value, (list, tuple, np.ndarray)):
            vals = list(value)
            if len(vals) == ndim:
                scaled = []
                for i, v in enumerate(vals):
                    if np.isscalar(v) and i in axis_factors:
                        scaled.append(float(v) / axis_factors[i])
                    else:
                        scaled.append(v)
                return _rebuild_like(value, scaled)
            if len(vals) == len(spatial_idx):
                scaled = []
                for local_i, v in enumerate(vals):
                    if np.isscalar(v):
                        axis_i = spatial_idx[local_i]
                        scaled.append(float(v) / axis_factors[axis_i])
                    else:
                        scaled.append(v)
                return _rebuild_like(value, scaled)
        return value

    def _apply_pyramid_param_policy(kwargs_in, base_level, level):
        if not isinstance(pyramid_param_policy, dict) or not pyramid_param_policy:
            return kwargs_in
        ctx = _compute_level_scale_context(base_level, level)
        if ctx is None:
            return kwargs_in

        out = dict(kwargs_in)
        for param_name, policy in pyramid_param_policy.items():
            if policy != "fixed_world":
                continue
            if param_name not in out:
                continue
            out[param_name] = _scale_value_fixed_world(out[param_name], ctx)
        return out

    if pyramid_strategy == "per_level":
        out_levels = []
        base_level = pyramid[0]
        for level in pyramid:
            a2 = list(uargs)
            k2 = dict(ukwargs)
            if loc_type == "arg":
                a2[loc_key] = level
            else:
                k2[loc_key] = level
            k2 = _apply_pyramid_param_policy(k2, base_level, level)
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
