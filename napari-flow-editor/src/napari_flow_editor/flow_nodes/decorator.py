import functools
import inspect
import dask.array as da
import numpy as np
import threading
from typing import Callable, Optional


def _try_import_cupy():
    try:
        import cupy as cp  # type: ignore
        return cp
    except Exception:
        return None


def _is_cupy_array(value, cp=None):
    cp_mod = cp or _try_import_cupy()
    if cp_mod is None:
        return False
    try:
        return isinstance(value, cp_mod.ndarray)
    except Exception:
        return False


def _convert_cuda_output_to_numpy(value, cp):
    if cp is None:
        return value
    if _is_cupy_array(value, cp):
        return cp.asnumpy(value)
    if isinstance(value, list):
        return [_convert_cuda_output_to_numpy(v, cp) for v in value]
    if isinstance(value, tuple):
        return tuple(_convert_cuda_output_to_numpy(v, cp) for v in value)
    if isinstance(value, dict):
        return {k: _convert_cuda_output_to_numpy(v, cp) for k, v in value.items()}
    return value


def _dask_array_has_cupy_chunks(value, cp=None):
    if not isinstance(value, da.Array):
        return False
    return _is_cupy_array(getattr(value, "_meta", None), cp)


def _resolve_output_dtype_policy(policy, sample):
    """
    Resolve a semantic output dtype policy against a sample array.

    ``None`` keeps legacy behavior. Policies are intentionally small and
    intent-based so nodes can request "image_float" without hardcoding one
    dtype separately for every backend.
    """
    if policy is None:
        return None

    if isinstance(policy, str):
        key = policy.strip().lower()
        if key in ("", "none", "default"):
            return None
        if key in ("preserve", "same", "input"):
            dtype = getattr(sample, "dtype", None)
            return np.dtype(dtype) if dtype is not None else None
        if key in ("image_float", "float_image", "float", "auto"):
            return np.dtype(np.float32)
        if key in ("bool", "boolean", "mask"):
            return np.dtype(np.bool_)
        if key in ("label", "labels"):
            return np.dtype(np.int32)
        try:
            return np.dtype(key)
        except TypeError as exc:
            raise ValueError(f"Unknown output_dtype_policy={policy!r}") from exc

    try:
        return np.dtype(policy)
    except TypeError as exc:
        raise ValueError(f"Invalid output_dtype_policy={policy!r}") from exc


def _cast_array_output(value, dtype, cp=None):
    if dtype is None:
        return value
    dtype = np.dtype(dtype)
    if isinstance(value, da.Array):
        return value.astype(dtype)
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)
    if _is_cupy_array(value, cp):
        return value.astype(dtype, copy=False)
    if isinstance(value, list):
        return [_cast_array_output(v, dtype, cp=cp) for v in value]
    if isinstance(value, tuple):
        return tuple(_cast_array_output(v, dtype, cp=cp) for v in value)
    return value

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
    cuda_function=None,
    cuda_arg_names=None,
    cuda_kwarg_names=None,
    backend: str = "auto",           # "auto" | "cpu" | "dask" | "cuda" | "dask_cuda"
    gpu_min_nbytes: int = 0,          # only applies to forced cuda selection on NumPy inputs
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
    independent_axes_param: Optional[str] = None,  # Optional param to set zero on independent axes (e.g. sigma)
    dask_output_dtype=None,
    cuda_output_dtype=None,
    output_dtype_policy=None,
    pyramid_param_policy: Optional[dict] = None,
    numpy_to_dask_chunks="auto",
    dask_target_chunk_mb: float = 64.0,
    dask_options: Optional[dict] = None,
    output_type: str = "image",   # "image" | "points"
):
    """
    Choose backend, handle pyramids, and apply layout-aware execution policies.

    By default, ``backend="auto"`` uses one fixed priority order:
    Dask+CUDA when available, then Dask, then CPU. In-memory NumPy arrays are
    promoted to Dask whenever a Dask strategy/function is configured.

    ``dask_options`` can be used to keep all Dask-specific controls in one place.
    It is backward-compatible with the legacy ``dask_*`` parameters and can
    override them selectively, e.g.:

      {
        "strategy": "neighborhood",
        "numpy_chunks": "spatial_auto",
        "map_overlap": {
          "depth": {"t": 0, "y": 2, "x": 2},
          "boundary": "none",
          "trim": True,
          "allow_rechunk": True,
        },
      }

    ``output_dtype_policy`` lets nodes declare dtype intent once for all
    backends. For example, ``"image_float"`` returns float32 image outputs on
    CPU, Dask, CUDA, and Dask+CUDA unless an explicit legacy dtype override is
    provided.

    Layout defaults:
      - time dims (T): independent (no filtering across time)
      - channel dims (C): independent (no channel mixing)
      - volumetric data (ZYX): processed as true 3D
    """
    import numpy as np
    import dask.array as da

    if kwargs is None:
        kwargs = {}

    backend_requested = str(backend or "auto").strip().lower()
    valid_backends = {"auto", "cpu", "dask", "cuda", "dask_cuda"}
    if backend_requested not in valid_backends:
        raise ValueError(
            f"Unknown backend={backend!r}. Expected one of: {sorted(valid_backends)}"
        )

    if dask_options is None:
        dask_options = {}
    elif not isinstance(dask_options, dict):
        raise TypeError(
            f"dask_options must be a dict when provided, got {type(dask_options)!r}"
        )
    else:
        dask_options = dict(dask_options)

    map_overlap_options = dask_options.get("map_overlap", {})
    if map_overlap_options is None:
        map_overlap_options = {}
    elif not isinstance(map_overlap_options, dict):
        raise TypeError(
            "dask_options['map_overlap'] must be a dict when provided, "
            f"got {type(map_overlap_options)!r}"
        )
    else:
        map_overlap_options = dict(map_overlap_options)

    def _dask_opt(name, legacy_value):
        return dask_options.get(name, legacy_value)

    dask_strategy = _dask_opt("strategy", dask_strategy)
    dask_halo_from_param = _dask_opt("halo_from_param", dask_halo_from_param)
    dask_halo_factor = _dask_opt("halo_factor", dask_halo_factor)
    dask_boundary_from_param = _dask_opt("boundary_from_param", dask_boundary_from_param)
    independent_axes_param = _dask_opt("independent_axes_param", independent_axes_param)
    dask_output_dtype = _dask_opt("output_dtype", dask_output_dtype)
    output_dtype_policy = _dask_opt("output_dtype_policy", output_dtype_policy)
    numpy_to_dask_chunks = _dask_opt("numpy_chunks", numpy_to_dask_chunks)
    dask_target_chunk_mb = _dask_opt("target_chunk_mb", dask_target_chunk_mb)
    output_type = str(_dask_opt("output_type", output_type)).strip().lower()
    output_coord_cols = _dask_opt("output_coord_cols", None)  # e.g. slice(None,-1) for blob_log
    point_size_col = _dask_opt("point_size_col", None)
    point_size_mode = str(_dask_opt("point_size_mode", "raw")).strip().lower()
    point_size_scale = _dask_opt("point_size_scale", 1.0)
    point_kwargs = dict(_dask_opt("point_kwargs", {}) or {})

    try:
        dask_halo_factor = float(dask_halo_factor or 0.0)
    except Exception:
        dask_halo_factor = 4.0
    if dask_halo_factor < 0:
        dask_halo_factor = abs(dask_halo_factor)
    try:
        dask_target_chunk_mb = float(dask_target_chunk_mb or 64.0)
    except Exception:
        dask_target_chunk_mb = 64.0
    if dask_target_chunk_mb <= 0:
        dask_target_chunk_mb = 64.0

    if cuda_func is not None and cuda_function is not None:
        raise ValueError("Use either cuda_func or cuda_function, not both.")

    resolved_cuda_target = None
    cuda_function_path = None
    cuda_resolve_error = None

    if callable(cuda_function):
        resolved_cuda_target = cuda_function
    elif isinstance(cuda_function, str):
        path = cuda_function.strip()
        if not path or "." not in path:
            raise ValueError(
                f"cuda_function={cuda_function!r} must be a callable or dotted import path."
            )
        cuda_function_path = path
    elif cuda_function is not None:
        raise TypeError(
            f"Unsupported cuda_function type: {type(cuda_function)!r}. Expected callable or str."
        )

    use_cuda_mapping = cuda_function is not None
    gpu_func = cuda_func or resolved_cuda_target
    cuda_arg_names_list = [str(n) for n in (cuda_arg_names or [])]
    cuda_kwarg_names_list = [str(n) for n in (cuda_kwarg_names or [])]
    try:
        default_sig = inspect.signature(default)
    except Exception:
        default_sig = None

    cp = _try_import_cupy() if (cuda_func is not None or cuda_function is not None) else None

    def _get_cuda_callable():
        nonlocal gpu_func, resolved_cuda_target, cuda_resolve_error
        if gpu_func is not None:
            return gpu_func
        if resolved_cuda_target is not None:
            gpu_func = resolved_cuda_target
            return gpu_func
        if not cuda_function_path:
            return None
        if cp is None:
            cuda_resolve_error = "CuPy not installed"
            return None
        try:
            import importlib
            module_name, attr_name = cuda_function_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            target = getattr(module, attr_name)
            if not callable(target):
                cuda_resolve_error = f"resolved object {cuda_function_path!r} is not callable"
                return None
            resolved_cuda_target = target
            gpu_func = target
            return target
        except Exception as exc:
            cuda_resolve_error = f"failed to resolve {cuda_function_path!r} ({exc})"
            return None

    def _cuda_runtime_ready():
        if (cuda_func is None) and (cuda_function is None):
            return False, "no CUDA function configured for this node"
        if cp is None:
            return False, "CuPy not installed"
        if _get_cuda_callable() is None:
            return False, cuda_resolve_error or "unable to resolve CUDA function"
        try:
            ndev = int(cp.cuda.runtime.getDeviceCount())
            if ndev < 1:
                return False, "no CUDA device detected"
            return True, f"{ndev} CUDA device(s) available"
        except Exception as exc:
            return False, f"CUDA runtime unavailable ({exc})"

    def _estimate_nbytes(sample):
        if sample is None:
            return 0
        try:
            return int(sample.nbytes)
        except Exception:
            pass
        if hasattr(sample, "shape") and hasattr(sample, "dtype"):
            try:
                return int(np.prod(sample.shape)) * int(sample.dtype.itemsize)
            except Exception:
                return 0
        return 0

    def _fit_chunk_tuple_to_sample(raw_chunks, sample, axes=None):
        shape = tuple(int(s) for s in getattr(sample, "shape", ()) or ())
        ndim = len(shape)
        if ndim == 0:
            return raw_chunks

        try:
            chunks = [int(c) for c in tuple(raw_chunks)]
        except Exception:
            return raw_chunks
        if not chunks:
            return "auto"

        if len(chunks) > ndim:
            chunks = chunks[-ndim:]
        elif len(chunks) < ndim:
            prefix = []
            axes_norm = str(axes or "").upper()
            for axis_i in range(ndim - len(chunks)):
                axis_name = axes_norm[axis_i] if len(axes_norm) == ndim else ""
                if axis_name in ("T", "C"):
                    prefix.append(1)
                elif axis_name == "Z":
                    prefix.append(min(shape[axis_i], 16))
                else:
                    prefix.append(shape[axis_i])
            chunks = prefix + chunks

        return tuple(max(1, min(int(size), int(chunk))) for size, chunk in zip(shape, chunks))

    def _spatial_auto_chunks(sample, axes=None):
        shape = tuple(int(s) for s in getattr(sample, "shape", ()) or ())
        ndim = len(shape)
        if ndim == 0:
            return "auto"

        dtype = getattr(sample, "dtype", np.dtype(np.float32))
        try:
            itemsize = max(1, int(np.dtype(dtype).itemsize))
        except Exception:
            itemsize = 4

        axes_norm = str(axes or "").upper()
        if len(axes_norm) != ndim:
            if ndim == 2:
                axes_norm = "YX"
            elif ndim == 3:
                axes_norm = "ZYX"
            elif ndim == 4:
                axes_norm = "TZYX"
            else:
                axes_norm = ("N" * max(0, ndim - 2)) + "YX"

        chunks = [None] * ndim
        fixed_element_factor = 1
        spatial_axes = []
        for axis_i, (axis_name, axis_size) in enumerate(zip(axes_norm, shape)):
            if axis_name in ("T", "C"):
                chunks[axis_i] = 1
                fixed_element_factor *= 1
            elif axis_name == "Z":
                chunk = min(axis_size, 16)
                chunks[axis_i] = chunk
                fixed_element_factor *= max(1, int(chunk))
            elif axis_name in ("Y", "X"):
                spatial_axes.append(axis_i)
            else:
                chunk = min(axis_size, 16) if ndim > 2 and axis_i < ndim - 2 else axis_size
                chunks[axis_i] = chunk
                fixed_element_factor *= max(1, int(chunk))

        target_bytes = max(1e-6, dask_target_chunk_mb) * (1024.0 ** 2)
        spatial_ndim = max(1, len(spatial_axes))
        target_spatial_elements = max(
            1.0, target_bytes / float(itemsize * max(1, fixed_element_factor))
        )
        spatial_edge = int(round(target_spatial_elements ** (1.0 / spatial_ndim)))
        spatial_edge = max(128, min(1024, spatial_edge))

        for axis_i in spatial_axes:
            chunks[axis_i] = max(1, min(shape[axis_i], spatial_edge))

        return tuple(int(c if c is not None else s) for c, s in zip(chunks, shape))

    def _resolve_numpy_to_dask_chunks(sample, axes=None):
        raw = numpy_to_dask_chunks
        if isinstance(raw, str):
            value = raw.strip().lower()
            if value in ("spatial_auto", "spatial-auto", "spatial"):
                return _spatial_auto_chunks(sample, axes=axes)
            if value == "auto":
                return "auto"
        if isinstance(raw, (tuple, list)):
            return _fit_chunk_tuple_to_sample(raw, sample, axes=axes)
        return raw

    def _numpy_to_dask_reason(sample):
        if not isinstance(sample, np.ndarray):
            return "input is not a NumPy array"
        axes = infer_axes_for_sample(
            sample,
            axes_hint=runtime_axes,
            axis_labels_hint=runtime_axis_labels,
            layout_hint=runtime_layout_kind,
        )
        chunks = _resolve_numpy_to_dask_chunks(sample, axes=axes)
        return f"auto-promoted NumPy input to Dask with chunks={chunks!r}"

    backend_message_printed = False

    def _log_backend(message):
        nonlocal backend_message_printed
        if backend_message_printed:
            return
        print(f"[dispatch:{getattr(default, '__name__', 'func')}] {message}")
        backend_message_printed = True

    def _emit_backend_trace(payload):
        recorder = None
        if isinstance(ctx, dict):
            recorder = ctx.get("backend_recorder")
        if not callable(recorder):
            return
        try:
            recorder(dict(payload or {}))
        except Exception:
            # Backend tracing must never affect node execution.
            pass

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

    def _last_axis_is_channel_like(shape):
        if not shape:
            return False
        try:
            return 1 <= int(shape[-1]) <= 4
        except Exception:
            return False

    def _coerce_axes_to_shape(axes_value, shape):
        if not isinstance(axes_value, str):
            return None
        axes_text = axes_value.strip().upper()
        ndim = len(shape)
        if len(axes_text) == ndim:
            return axes_text
        if (
            len(axes_text) == ndim - 1
            and "C" not in axes_text
            and _last_axis_is_channel_like(shape)
        ):
            return f"{axes_text}C"
        return None

    def axis_labels_to_axes(labels, ndim=None):
        if labels is None:
            return None
        try:
            seq = list(labels)
        except Exception:
            return None
        if ndim is not None and len(seq) != ndim:
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
        shape = tuple(sample.shape)
        ndim = len(shape)
        ax = _coerce_axes_to_shape(axes_hint, shape)
        if ax:
            return ax

        labels_ax = _coerce_axes_to_shape(axis_labels_to_axes(axis_labels_hint), shape)
        if labels_ax:
            return labels_ax
        if isinstance(layout_hint, str):
            mapped = layout_to_axes.get(layout_hint.lower())
            ax = _coerce_axes_to_shape(mapped, shape)
            if ax:
                return ax

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
    def has_dask_backend():
        return dask_func is not None or dask_strategy in ("pointwise", "neighborhood", "chunked_delayed")

    def has_dask_cuda_capability():
        cuda_ok, _ = _cuda_runtime_ready()
        return (
            cuda_ok
            and has_dask_backend()
            and dask_strategy in ("pointwise", "neighborhood")
            and gpu_func is not None
        )

    def _choose_backend_mode(sample):
        cuda_ok, cuda_reason = _cuda_runtime_ready()
        dask_ok = has_dask_backend()
        dask_cuda_ok = has_dask_cuda_capability()
        sample_is_dask = isinstance(sample, da.Array)
        sample_is_numpy = isinstance(sample, np.ndarray)
        sample_is_cupy = _is_cupy_array(sample, cp)
        sample_nbytes = _estimate_nbytes(sample)
        gpu_big_enough = int(gpu_min_nbytes or 0) <= 0 or sample_nbytes >= int(gpu_min_nbytes)

        def fallback_for(mode):
            if mode == "cuda":
                if dask_ok:
                    return "dask", f"requested cuda, fallback dask ({cuda_reason})"
                return "cpu", f"requested cuda, fallback cpu ({cuda_reason})"
            if mode == "dask":
                return "cpu", "requested dask, fallback cpu (no dask strategy/function)"
            if mode == "dask_cuda":
                if dask_cuda_ok:
                    return "dask_cuda", "requested dask_cuda"
                if dask_ok:
                    return "dask", f"requested dask_cuda, fallback dask ({cuda_reason})"
                return "cpu", f"requested dask_cuda, fallback cpu ({cuda_reason})"
            return "cpu", "fallback cpu"

        if backend_requested == "cpu":
            return "cpu", "forced cpu backend"
        if backend_requested == "dask":
            if dask_ok:
                return "dask", "forced dask backend"
            return fallback_for("dask")
        if backend_requested == "cuda":
            if cuda_ok and (sample_is_numpy or sample_is_cupy):
                if not gpu_big_enough:
                    if dask_ok:
                        return "dask", (
                            f"requested cuda but input below gpu_min_nbytes={gpu_min_nbytes}, "
                            "fallback dask"
                        )
                    return "cpu", (
                        f"requested cuda but input below gpu_min_nbytes={gpu_min_nbytes}, "
                        "fallback cpu"
                    )
                return "cuda", "forced cuda backend"
            return fallback_for("cuda")
        if backend_requested == "dask_cuda":
            return fallback_for("dask_cuda")

        # auto
        if dask_cuda_ok and (sample_is_dask or sample_is_numpy):
            return "dask_cuda", "auto backend selected dask_cuda"

        if dask_ok:
            if sample_is_numpy:
                return "dask", f"auto backend selected dask; {_numpy_to_dask_reason(sample)}"
            if sample_is_dask:
                if gpu_func is not None and not cuda_ok:
                    return "dask", f"auto fallback dask ({cuda_reason})"
                return "dask", "auto backend selected dask"
            return "dask", "auto backend selected dask"

        if gpu_func is not None and (sample_is_numpy or sample_is_dask or sample_is_cupy):
            return "cpu", f"auto fallback cpu ({cuda_reason})"
        return "cpu", "auto backend selected cpu"

    def maybe_promote_numpy_to_cuda(args_in, kwargs_in, sample):
        if not isinstance(sample, np.ndarray):
            return args_in, kwargs_in, sample
        if cp is None:
            return args_in, kwargs_in, sample

        def convert(v):
            if isinstance(v, np.ndarray):
                return cp.asarray(v)
            return v

        promoted_args = [convert(v) for v in args_in]
        promoted_kwargs = {k: convert(v) for k, v in kwargs_in.items()}

        promoted_sample = sample
        for v in promoted_args:
            if _is_cupy_array(v, cp):
                promoted_sample = v
                break
        if not _is_cupy_array(promoted_sample, cp):
            for v in promoted_kwargs.values():
                if _is_cupy_array(v, cp):
                    promoted_sample = v
                    break

        return promoted_args, promoted_kwargs, promoted_sample

    def maybe_promote_numpy_to_dask(args_in, kwargs_in, sample):
        if not isinstance(sample, np.ndarray):
            return args_in, kwargs_in, sample
        axes = infer_axes_for_sample(
            sample,
            axes_hint=runtime_axes,
            axis_labels_hint=runtime_axis_labels,
            layout_hint=runtime_layout_kind,
        )
        resolved_chunks = _resolve_numpy_to_dask_chunks(sample, axes=axes)

        def convert(v):
            if isinstance(v, np.ndarray):
                # Avoid hashing large in-memory arrays just to create a Dask name.
                # Dask's default tokenization can dominate dispatch time on GB-scale
                # NumPy inputs; a generated name is enough for transient node graphs.
                return da.from_array(v, chunks=resolved_chunks, name=False)
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

    def _bind_default_call(args_in, kwargs_in):
        if default_sig is None:
            return None
        try:
            bound = default_sig.bind_partial(*args_in, **kwargs_in)
            bound.apply_defaults()
            return dict(bound.arguments)
        except TypeError:
            return None

    def _execute_cuda(args_in, kwargs_in):
        target = _get_cuda_callable()
        if target is None:
            raise RuntimeError("CUDA path selected but no CUDA function is configured.")

        # Legacy mode: explicit cuda_func receives the same args/kwargs.
        if not use_cuda_mapping:
            return target(*args_in, **kwargs_in)

        # If no explicit mapping is provided, pass through.
        if not cuda_arg_names_list and not cuda_kwarg_names_list:
            return target(*args_in, **kwargs_in)

        # Mapping mode: build CUDA call from default-function argument names.
        bound_map = _bind_default_call(args_in, kwargs_in)
        if bound_map is None:
            raise TypeError(
                "Unable to bind runtime call to default function signature "
                "for CUDA argument mapping."
            )

        cuda_args_resolved = []
        for name in cuda_arg_names_list:
            if name not in bound_map:
                raise TypeError(
                    f"cuda_arg_names contains unknown parameter {name!r} for this call."
                )
            cuda_args_resolved.append(bound_map[name])

        cuda_kwargs_resolved = {}
        for name in cuda_kwarg_names_list:
            if name not in bound_map:
                raise TypeError(
                    f"cuda_kwarg_names contains unknown parameter {name!r} for this call."
                )
            cuda_kwargs_resolved[name] = bound_map[name]

        return target(*cuda_args_resolved, **cuda_kwargs_resolved)

    def _small_probe_shape(shape):
        dims = []
        for dim in tuple(shape):
            try:
                n = int(dim)
            except Exception:
                n = 1
            dims.append(max(1, min(n, 2)))
        return tuple(dims) if dims else (1,)

    def _probe_value_for_cuda(v):
        if isinstance(v, da.Array):
            probe = np.zeros(_small_probe_shape(v.shape), dtype=v.dtype)
            return cp.asarray(probe) if cp is not None else probe
        if isinstance(v, np.ndarray):
            probe = np.zeros(_small_probe_shape(v.shape), dtype=v.dtype)
            return cp.asarray(probe) if cp is not None else probe
        if _is_cupy_array(v, cp):
            dtype = getattr(v, "dtype", np.float32)
            shape = _small_probe_shape(getattr(v, "shape", (1,)))
            return cp.asarray(np.zeros(shape, dtype=dtype))
        return v

    def _cuda_preflight(args_in, kwargs_in):
        """
        Automatic CUDA compatibility check for the *current* call:
        Tiny runtime probe (catches unsupported signatures and values).
        """
        if gpu_func is None:
            return False, "no CUDA function configured"

        try:
            probe_args = [_probe_value_for_cuda(v) for v in args_in]
            probe_kwargs = {k: _probe_value_for_cuda(v) for k, v in kwargs_in.items()}
            _execute_cuda(probe_args, probe_kwargs)
            return True, "CUDA preflight passed"
        except (TypeError, ValueError, NotImplementedError) as exc:
            return False, f"CUDA cannot preserve current user parameters ({exc})"
        except Exception as exc:
            return False, f"CUDA preflight runtime issue ({exc})"

    def _fallback_after_cuda_preflight_failure(mode, sample=None):
        dask_ok = has_dask_backend()
        if mode in ("cuda", "dask_cuda"):
            if dask_ok:
                return "dask", "fallback to dask to preserve user-selected parameters"
            return "cpu", "fallback to cpu to preserve user-selected parameters"
        return "cpu", "fallback to cpu"

    def _axis_key_to_index(axis_key, ndim, axes):
        if isinstance(axis_key, int):
            if axis_key < 0:
                axis_key += int(ndim)
            return axis_key if 0 <= axis_key < int(ndim) else None
        token = str(axis_key).strip().lower()
        if token == "":
            return None
        if token.isdigit():
            idx = int(token)
            return idx if 0 <= idx < int(ndim) else None

        axis_alias = {
            "t": "T",
            "time": "T",
            "z": "Z",
            "depth": "Z",
            "y": "Y",
            "x": "X",
            "c": "C",
            "ch": "C",
            "channel": "C",
            "channels": "C",
        }
        axis_name = axis_alias.get(token)
        if axis_name and isinstance(axes, str) and len(axes) == int(ndim):
            try:
                return int(axes.index(axis_name))
            except ValueError:
                return None
        return None

    def _normalize_overlap_depth(depth_cfg, sample_arr, axes):
        ndim = int(getattr(sample_arr, "ndim", 0) or 0)
        if ndim <= 0:
            return depth_cfg

        def _as_int_depth(value):
            try:
                return max(0, int(np.ceil(abs(float(value)))))
            except Exception:
                return 0

        if np.isscalar(depth_cfg):
            depth = [_as_int_depth(depth_cfg)] * ndim
        elif isinstance(depth_cfg, dict):
            depth = [0] * ndim
            for key, raw_value in depth_cfg.items():
                axis_i = _axis_key_to_index(key, ndim, axes)
                if axis_i is None:
                    continue
                depth[axis_i] = _as_int_depth(raw_value)
        else:
            try:
                seq = [_as_int_depth(v) for v in list(depth_cfg)]
            except Exception:
                return tuple(0 for _ in range(ndim))
            if not seq:
                return tuple(0 for _ in range(ndim))
            if len(seq) == 1:
                depth = seq * ndim
            elif len(seq) < ndim:
                depth = seq + [seq[-1]] * (ndim - len(seq))
            else:
                depth = seq[:ndim]

        return tuple(min(int(d), max(0, int(sz) - 1)) for d, sz in zip(depth, sample_arr.shape))

    def _resolve_map_overlap_boundary(sample_arr, axes, kwargs_in):
        boundary_cfg = map_overlap_options.get("boundary")
        if boundary_cfg is None:
            return None

        if callable(boundary_cfg):
            return boundary_cfg(sample_arr, axes, kwargs_in)

        ndim = int(getattr(sample_arr, "ndim", 0) or 0)
        if isinstance(boundary_cfg, dict):
            out = {}
            for key, raw_value in boundary_cfg.items():
                axis_i = _axis_key_to_index(key, ndim, axes)
                if axis_i is None:
                    continue
                out[int(axis_i)] = raw_value
            return out

        return boundary_cfg

    def _resolve_map_overlap_depth(sample_arr, axes, kwargs_in):
        depth_cfg = map_overlap_options.get("depth")
        if depth_cfg is None:
            return None
        if callable(depth_cfg):
            depth_cfg = depth_cfg(sample_arr, axes, kwargs_in)
        return _normalize_overlap_depth(depth_cfg, sample_arr, axes)

    def _boundary_for_overlap(sample_arr, axes, kwargs_in):
        explicit_boundary = _resolve_map_overlap_boundary(sample_arr, axes, kwargs_in)
        if explicit_boundary is not None:
            return explicit_boundary

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
        explicit_depth = _resolve_map_overlap_depth(sample_arr, axes, kwargs_in)
        if explicit_depth is not None:
            return explicit_depth

        if not dask_halo_from_param:
            return tuple(0 for _ in range(sample_arr.ndim))

        halo_source = kwargs_in.get(dask_halo_from_param, 0)
        ndim = int(getattr(sample_arr, "ndim", 0) or 0)
        if np.isscalar(halo_source):
            seq = [float(halo_source)] * ndim
        else:
            seq = [float(x) for x in list(halo_source)]
            # Only expand/adjust if sequence length doesn't already match ndim
            if len(seq) != ndim:
                if len(seq) == 1:
                    seq = seq * ndim
                elif len(seq) < ndim:
                    seq = seq + [seq[-1]] * (ndim - len(seq))
                elif len(seq) > ndim:
                    seq = seq[:ndim]

        radii = [max(0, int(np.ceil(abs(v) * float(dask_halo_factor)))) for v in seq]

        if isinstance(axes, str) and len(axes) == ndim:
            spatial = {i for i, a in enumerate(axes) if a in ("Z", "Y", "X")}
            radii = [r if i in spatial else 0 for i, r in enumerate(radii)]

        # Avoid overlap depth > axis size (common failure on small Z/C dims)
        return tuple(min(r, max(0, int(sz) - 1)) for r, sz in zip(radii, sample_arr.shape))

    def _as_axis_sequence(value, ndim):
        if np.isscalar(value):
            return [float(value)] * int(ndim)
        try:
            seq = [float(x) for x in list(value)]
        except Exception:
            return None
        if not seq:
            return None
        if len(seq) == int(ndim):
            return seq
        if len(seq) == 1:
            return seq * int(ndim)
        if len(seq) < int(ndim):
            return seq + [seq[-1]] * (int(ndim) - len(seq))
        return seq[: int(ndim)]

    def _apply_independent_axes_param(kwargs_in, independent_axes, ndim):
        """
        Optional optimization: instead of explicit per-index splitting on independent
        axes (T/C), encode independence in a per-axis parameter vector by setting
        those axes to zero (e.g. sigma=(0, s, s) for TYX).

        Returns:
            (new_kwargs, changed_flag, detail_message)
        """
        pname = str(independent_axes_param or "").strip()
        if not pname:
            return kwargs_in, False, ""
        if pname not in kwargs_in:
            return kwargs_in, False, f"independent_axes_param={pname!r} not found in kwargs"

        seq = _as_axis_sequence(kwargs_in[pname], ndim)
        if seq is None:
            return kwargs_in, False, f"independent_axes_param={pname!r} could not be broadcast to ndim={ndim}"

        changed = False
        for axis_i in independent_axes:
            if axis_i < 0 or axis_i >= ndim:
                continue
            if seq[axis_i] != 0.0:
                seq[axis_i] = 0.0
                changed = True

        if not changed:
            return kwargs_in, False, ""

        updated = dict(kwargs_in)
        updated[pname] = tuple(seq)
        return updated, True, f"applied {pname} zeroing on independent axes"

    def _policy_output_dtype(sample):
        return _resolve_output_dtype_policy(output_dtype_policy, sample)

    def _dask_block_output_dtype(sample, *, use_gpu=False):
        if use_gpu and cuda_output_dtype is not None:
            return np.dtype(cuda_output_dtype)
        if dask_output_dtype is not None:
            return np.dtype(dask_output_dtype)
        return _policy_output_dtype(sample)

    def _eager_output_dtype(sample, *, use_gpu=False):
        if use_gpu and cuda_output_dtype is not None:
            return np.dtype(cuda_output_dtype)
        return _policy_output_dtype(sample)

    def _execute_auto_dask(
        args_in,
        kwargs_in,
        sample_loc,
        axes,
        compute_func,
        use_gpu=False,
        depth_kwargs_in=None,
    ):
        if dask_strategy not in ("pointwise", "neighborhood"):
            raise ValueError(f"Unknown dask_strategy={dask_strategy!r}")
        sample_exec = None
        if sample_loc is not None:
            loc_type, loc_key = sample_loc
            sample_exec = args_in[loc_key] if loc_type == "arg" else kwargs_in.get(loc_key)
        if not isinstance(sample_exec, da.Array):
            raise TypeError("Auto dask strategy requires a dask array sample.")
        if use_gpu and cp is None:
            raise RuntimeError("dask_cuda path selected but CuPy is unavailable.")

        def make_meta(dtype, *, gpu=False, ndim=0):
            shape = (0,) * max(0, int(ndim or 0))
            meta = np.empty(shape, dtype=np.dtype(dtype))
            return cp.asarray(meta) if gpu else meta

        if use_gpu and not _dask_array_has_cupy_chunks(sample_exec, cp):
            sample_exec = sample_exec.map_blocks(
                cp.asarray,
                dtype=sample_exec.dtype,
                meta=make_meta(sample_exec.dtype, gpu=True, ndim=sample_exec.ndim),
            )

        block_output_dtype = _dask_block_output_dtype(sample_exec, use_gpu=use_gpu)

        def finalize_block_output(out):
            return _cast_array_output(
                out,
                block_output_dtype,
                cp=cp if use_gpu else None,
            )

        if dask_strategy == "pointwise":
            def block_apply(block):
                a2, k2 = _replace_sample(args_in, kwargs_in, sample_loc, block)
                out = compute_func(*a2, **k2)
                return finalize_block_output(out)

            out_dtype = block_output_dtype if block_output_dtype is not None else sample_exec.dtype
            return da.map_blocks(
                block_apply,
                sample_exec,
                dtype=out_dtype,
                meta=make_meta(out_dtype, gpu=use_gpu, ndim=sample_exec.ndim),
            )

        # dask_strategy == "neighborhood"
        def overlap_apply(block):
            a2, k2 = _replace_sample(args_in, kwargs_in, sample_loc, block)
            out = compute_func(*a2, **k2)
            return finalize_block_output(out)

        depth_kwargs = kwargs_in if depth_kwargs_in is None else depth_kwargs_in
        depth = _depth_for_overlap(sample_exec, axes, depth_kwargs)
        boundary = _boundary_for_overlap(sample_exec, axes, kwargs_in)
        trim = bool(map_overlap_options.get("trim", True))
        allow_rechunk = bool(map_overlap_options.get("allow_rechunk", True))
        align_arrays = bool(map_overlap_options.get("align_arrays", True))
        out_dtype = block_output_dtype if block_output_dtype is not None else sample_exec.dtype
        result = sample_exec.map_overlap(
            overlap_apply,
            depth=depth,
            boundary=boundary,
            trim=trim,
            allow_rechunk=allow_rechunk,
            align_arrays=align_arrays,
            dtype=out_dtype,
            meta=make_meta(out_dtype, gpu=use_gpu, ndim=sample_exec.ndim),
        )
        if use_gpu:
            result._meta = make_meta(out_dtype, gpu=True, ndim=sample_exec.ndim)
        return result

    def _execute_chunked_delayed(args_in, kwargs_in, sample_loc, axes):
        """
        Coordinate-output Dask strategy using dask.delayed.

        Each spatial chunk (with halo) is processed independently by `default`.
        Returned coordinates are trimmed to the inner (non-halo) region, shifted
        to global image coordinates, and expanded back to the original axis
        order. Non-spatial axes such as T/C are handled independently, matching
        the image-output dispatcher semantics.

        `output_coord_cols` (from dask_options) lets nodes strip extra columns
        from the coordinate array (e.g. slice(None,-1) drops the sigma column
        returned by skimage.feature.blob_log).
        """
        import dask
        import itertools

        if sample_loc is None:
            raise ValueError("chunked_delayed strategy requires an array input.")
        loc_type, loc_key = sample_loc
        sample = args_in[loc_key] if loc_type == "arg" else kwargs_in[loc_key]
        sample_ndim = int(getattr(sample, "ndim", 2))
        axes_norm = str(axes or "").upper()
        if len(axes_norm) != sample_ndim:
            axes_norm = ""
        output_ndim = sample_ndim
        extra_cols = 1 if point_size_col is not None else 0

        def _coerce(raw, coord_ndim, size_ndim):
            """Normalise raw function output to coords plus optional attrs."""
            if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
                return (
                    np.empty((0, coord_ndim), dtype=np.float32),
                    np.empty((0, extra_cols), dtype=np.float32),
                )
            out = np.asarray(raw, dtype=np.float32)
            if out.ndim == 1:
                out = out.reshape(1, -1)
            attrs = np.empty((len(out), 0), dtype=np.float32)
            if point_size_col is not None:
                size = out[:, point_size_col].astype(np.float32, copy=False)
                if point_size_mode in {"blob_diameter", "sigma_diameter"}:
                    size = size * (2.0 * float(np.sqrt(max(1, int(size_ndim)))))
                size = size * float(point_size_scale)
                attrs = size.reshape(-1, 1).astype(np.float32, copy=False)
            if output_coord_cols is not None:
                out = out[:, output_coord_cols]
            return out, attrs

        def _expand_coords(coords, core_axis_indices, fixed_axis_values, attrs=None):
            if coords is None or len(coords) == 0:
                return np.empty((0, output_ndim + extra_cols), dtype=np.float32)
            full = np.zeros((len(coords), output_ndim), dtype=np.float32)
            for axis_i, value in fixed_axis_values:
                full[:, int(axis_i)] = float(value)
            coord_cols = min(coords.shape[1], len(core_axis_indices))
            for coord_i in range(coord_cols):
                full[:, int(core_axis_indices[coord_i])] = coords[:, coord_i]
            if extra_cols:
                if attrs is None or len(attrs) != len(coords):
                    attrs = np.empty((len(coords), extra_cols), dtype=np.float32)
                return np.concatenate([full, attrs], axis=1)
            return full

        def _run_chunk(
            chunk_np,
            args_no_img,
            kwargs_no_img,
            padded_origin,
            inner_lo,
            inner_hi,
            core_axis_indices,
            fixed_axis_values,
            size_ndim,
        ):
            """Run default on one padded chunk, return inner global coords."""
            a2 = list(args_no_img)
            k2 = dict(kwargs_no_img)
            if loc_type == "arg":
                a2[loc_key] = chunk_np
            else:
                k2[loc_key] = chunk_np

            coords, attrs = _coerce(default(*a2, **k2), len(core_axis_indices), size_ndim)
            if len(coords) == 0:
                return _expand_coords(coords, core_axis_indices, fixed_axis_values, attrs)

            # Keep only coords whose center falls inside the inner region.
            ncols = coords.shape[1]
            mask = np.ones(len(coords), dtype=bool)
            for dim_i in range(min(ncols, len(inner_lo))):
                mask &= (coords[:, dim_i] >= inner_lo[dim_i])
                mask &= (coords[:, dim_i] < inner_hi[dim_i])
            coords = coords[mask]
            attrs = attrs[mask] if len(attrs) else attrs
            if len(coords) == 0:
                return _expand_coords(coords, core_axis_indices, fixed_axis_values, attrs)

            # Shift chunk-local → global coordinates.
            origin = np.array(padded_origin[:ncols], dtype=np.float32)
            coords += origin
            return _expand_coords(coords, core_axis_indices, fixed_axis_values, attrs)

        def _vstack(arrays):
            valid = [a for a in arrays if a is not None and len(a) > 0]
            if not valid:
                return np.empty((0, output_ndim + extra_cols), dtype=np.float32)
            return np.concatenate(valid, axis=0)

        if axes_norm:
            core_axis_indices = [
                i for i, axis_name in enumerate(axes_norm) if axis_name in ("Z", "Y", "X")
            ]
            if time_policy == "joint":
                core_axis_indices.extend(
                    i for i, axis_name in enumerate(axes_norm) if axis_name == "T"
                )
            if channel_policy == "joint":
                core_axis_indices.extend(
                    i for i, axis_name in enumerate(axes_norm) if axis_name == "C"
                )
            core_axis_indices = sorted(set(core_axis_indices))
        else:
            core_axis_indices = list(range(sample_ndim))

        if not core_axis_indices:
            core_axis_indices = list(range(sample_ndim))

        independent_axes = [i for i in range(sample_ndim) if i not in core_axis_indices]

        def _slice_for_independent(index_tuple):
            slicer = [slice(None)] * sample_ndim
            fixed_values = []
            for local_i, axis_i in enumerate(independent_axes):
                value = int(index_tuple[local_i])
                slicer[axis_i] = value
                fixed_values.append((axis_i, value))
            return tuple(slicer), fixed_values

        def _run_core(core_sample, core_axis_indices_now, fixed_axis_values):
            core_axes = (
                "".join(axes_norm[i] for i in core_axis_indices_now)
                if axes_norm
                else None
            )
            core_ndim = int(getattr(core_sample, "ndim", len(core_axis_indices_now)))
            if axes_norm:
                size_ndim = sum(
                    1 for i in core_axis_indices_now if axes_norm[i] in ("Z", "Y", "X")
                )
            else:
                size_ndim = core_ndim

            a2, k2 = list(args_in), dict(kwargs_in)
            if loc_type == "arg":
                a2[loc_key] = core_sample
            else:
                k2[loc_key] = core_sample

            # NumPy path: run once on this independent slice.
            if not isinstance(core_sample, da.Array):
                coords, attrs = _coerce(default(*a2, **k2), core_ndim, size_ndim)
                return _expand_coords(coords, core_axis_indices_now, fixed_axis_values, attrs)

            shape = core_sample.shape
            chunks = core_sample.chunks
            depth = _depth_for_overlap(core_sample, core_axes, kwargs_in)

            dim_starts = []
            for dim_i in range(core_ndim):
                acc, starts = 0, []
                for c in chunks[dim_i]:
                    starts.append(acc)
                    acc += c
                dim_starts.append(starts)

            args_no_img = list(a2)
            kwargs_no_img = dict(k2)
            if loc_type == "arg":
                args_no_img[loc_key] = None
            else:
                kwargs_no_img[loc_key] = None

            delayed_chunks = []
            for chunk_idx in itertools.product(*[range(len(c)) for c in chunks]):
                padded_slices = []
                padded_origin = []
                inner_lo = []
                inner_hi = []

                for dim_i in range(core_ndim):
                    start = dim_starts[dim_i][chunk_idx[dim_i]]
                    size = chunks[dim_i][chunk_idx[dim_i]]
                    end = start + size
                    d = depth[dim_i]

                    p_start = max(0, start - d)
                    p_end = min(shape[dim_i], end + d)

                    inner_local_start = start - p_start
                    inner_local_end = inner_local_start + size

                    padded_slices.append(slice(p_start, p_end))
                    padded_origin.append(p_start)
                    inner_lo.append(inner_local_start)
                    inner_hi.append(inner_local_end)

                chunk_dask = core_sample[tuple(padded_slices)]
                delayed_chunks.append(
                    dask.delayed(_run_chunk)(
                        chunk_dask,
                        args_no_img,
                        kwargs_no_img,
                        tuple(padded_origin),
                        tuple(inner_lo),
                        tuple(inner_hi),
                        tuple(core_axis_indices_now),
                        tuple(fixed_axis_values),
                        int(size_ndim),
                    )
                )

            return dask.delayed(_vstack)(delayed_chunks)

        if independent_axes:
            pieces = []
            independent_shape = tuple(int(sample.shape[i]) for i in independent_axes)
            for index_tuple in np.ndindex(*independent_shape):
                slicer, fixed_values = _slice_for_independent(index_tuple)
                core_sample = sample[slicer]
                pieces.append(_run_core(core_sample, core_axis_indices, fixed_values))
            return dask.delayed(_vstack)(pieces)

        return _run_core(sample, core_axis_indices, ())

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

        backend_mode, initial_reason = _choose_backend_mode(sample)
        reasons = [initial_reason]
        base_args, base_kwargs, base_sample = list(args_in), dict(kwargs_in), sample
        args_exec, kwargs_exec, sample_exec = base_args, base_kwargs, base_sample

        # Resolve/validate backend with automatic CUDA compatibility probing.
        for _ in range(3):
            args_exec, kwargs_exec, sample_exec = list(base_args), dict(base_kwargs), base_sample

            if backend_mode == "cuda":
                args_exec, kwargs_exec, sample_exec = maybe_promote_numpy_to_cuda(
                    args_exec, kwargs_exec, sample_exec
                )
            elif backend_mode in ("dask", "dask_cuda"):
                args_exec, kwargs_exec, sample_exec = maybe_promote_numpy_to_dask(
                    args_exec, kwargs_exec, sample_exec
                )

            if backend_mode in ("cuda", "dask_cuda"):
                ok_cuda, cuda_reason = _cuda_preflight(args_exec, kwargs_exec)
                reasons.append(cuda_reason)
                if not ok_cuda:
                    next_mode, fallback_reason = _fallback_after_cuda_preflight_failure(
                        backend_mode, sample=base_sample
                    )
                    reasons.append(fallback_reason)
                    if next_mode == backend_mode:
                        backend_mode = "cpu"
                        break
                    backend_mode = next_mode
                    continue
            break

        backend_reason = "; ".join(reasons)
        _log_backend(
            f"requested={backend_requested}, selected={backend_mode}. {backend_reason}"
        )
        _emit_backend_trace(
            {
                "dispatch_function": getattr(default, "__name__", "func"),
                "requested_backend": backend_requested,
                "selected_backend": backend_mode,
                "reason": backend_reason,
            }
        )

        axes = infer_axes_for_sample(
            sample_exec,
            axes_hint=runtime_axes,
            axis_labels_hint=runtime_axis_labels,
            layout_hint=runtime_layout_kind,
        )

        # chunked_delayed: coordinate-output strategy. It returns coordinates
        # instead of an image, but still uses the inferred axis layout above.
        if dask_strategy == "chunked_delayed" and backend_mode in ("dask", "cpu"):
            raw = _execute_chunked_delayed(args_exec, kwargs_exec, sample_loc, axes)
            if output_type == "points":
                coords = raw.compute() if hasattr(raw, "compute") else raw
                meta = dict(point_kwargs)
                if point_size_col is not None:
                    coord_ndim = int(getattr(sample_exec, "ndim", 0) or 0)
                    if coord_ndim <= 0 or coord_ndim > coords.shape[1]:
                        coord_ndim = max(0, coords.shape[1] - 1)
                    if coords.shape[1] > coord_ndim:
                        meta["size"] = coords[:, coord_ndim].astype(np.float32, copy=False)
                        coords = coords[:, :coord_ndim]
                return (coords, meta, "points")
            return raw

        auto_dask = (
            isinstance(sample_exec, da.Array)
            and (backend_mode in ("dask", "dask_cuda"))
            and dask_func is None
            and dask_strategy in ("pointwise", "neighborhood")
        )
        if auto_dask:
            if backend_mode == "dask_cuda":
                def _compute_cuda(*a_now, **k_now):
                    return _execute_cuda(a_now, k_now)
                compute_func = _compute_cuda
            else:
                compute_func = default
            use_gpu = backend_mode == "dask_cuda"

            def run_now(a_now, k_now, *, depth_kwargs=None):
                return _execute_auto_dask(
                    a_now,
                    k_now,
                    sample_loc,
                    axes,
                    compute_func,
                    use_gpu=use_gpu,
                    depth_kwargs_in=depth_kwargs,
                )
        else:
            if (
                backend_mode == "dask"
                and isinstance(sample_exec, da.Array)
                and dask_func is not None
            ):
                func = dask_func
            else:
                func = default

            def run_now(a_now, k_now, *, depth_kwargs=None):
                if backend_mode == "cuda":
                    out = _execute_cuda(a_now, k_now)
                    out = _cast_array_output(
                        out,
                        _eager_output_dtype(sample_exec, use_gpu=True),
                        cp=cp,
                    )
                    return _convert_cuda_output_to_numpy(out, cp)
                out = func(*a_now, **k_now)
                if backend_mode == "dask":
                    out = _cast_array_output(
                        out,
                        _dask_block_output_dtype(sample_exec, use_gpu=False),
                        cp=None,
                    )
                else:
                    out = _cast_array_output(
                        out,
                        _eager_output_dtype(sample_exec, use_gpu=False),
                        cp=None,
                    )
                return out

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

        # Prefer vectorized independent-axis handling when available (e.g. sigma
        # for Gaussian) to avoid building one graph branch per T/C index.
        if independent and sample_exec is not None and hasattr(sample_exec, "shape"):
            kwargs_adjusted, changed, detail = _apply_independent_axes_param(
                kwargs_exec, independent, len(sample_exec.shape)
            )
            if changed:
                if detail:
                    _log_backend(
                        f"requested={backend_requested}, selected={backend_mode}. {backend_reason}; {detail}"
                    )
                return run_now(
                    args_exec,
                    kwargs_adjusted,
                    depth_kwargs=kwargs_exec,
                )

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
            slicer = [slice(None)] * len(sample_exec.shape)
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
