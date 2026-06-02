from .decorator import register_node, dispatch, smart_compute
import skimage.filters
import skimage.morphology
import numpy as np
import functools


# ---------------------------------------------------------------------------
# CUDA helpers (cuda_func path — callable, not string)
# These are called directly by the dispatcher with the CuPy array already
# transferred; they must not import cupy at module level.
# ---------------------------------------------------------------------------

def _cuda_sobel(image, mode='reflect'):
    import cupyx.scipy.ndimage as cnd
    import cupy as cp
    sx = cnd.sobel(image, axis=-1, mode=mode)
    sy = cnd.sobel(image, axis=-2, mode=mode)
    return cp.hypot(sx, sy)


def _cuda_prewitt(image, mode='reflect'):
    import cupyx.scipy.ndimage as cnd
    import cupy as cp
    px = cnd.prewitt(image, axis=-1, mode=mode)
    py = cnd.prewitt(image, axis=-2, mode=mode)
    return cp.hypot(px, py)


def _cuda_unsharp_mask(image, radius=1.0, amount=1.0):
    import cupyx.scipy.ndimage as cnd
    blurred = cnd.gaussian_filter(image, sigma=radius)
    return image + amount * (image - blurred)


def _cuda_difference_of_gaussians(image, low_sigma=1.0, high_sigma=2.0, mode='nearest'):
    import cupyx.scipy.ndimage as cnd
    low = cnd.gaussian_filter(image, sigma=low_sigma, mode=mode)
    high = cnd.gaussian_filter(image, sigma=high_sigma, mode=mode)
    return low - high


def _cuda_threshold_mean(image):
    import cupy as cp
    return float(cp.mean(image))


def _threshold_to_mask(threshold, image, **_):
    """Convert a threshold image to this app's boolean mask output."""
    return image > threshold


def _validate_odd_window(value, name):
    value = int(value)
    if value < 3 or value % 2 == 0:
        raise ValueError(f"{name} must be an odd integer >= 3, got {value!r}.")
    return value


def _sigma_sequence(start, stop, step, *, name="sigmas"):
    try:
        start = float(start)
        stop = float(stop)
        step = float(step)
    except Exception as exc:
        raise ValueError(f"{name} bounds must be numeric.") from exc
    if step <= 0:
        raise ValueError(f"{name} step must be > 0, got {step!r}.")
    if stop <= start:
        raise ValueError(
            f"{name} stop must be greater than start, got start={start!r}, stop={stop!r}."
        )
    sigmas = tuple(float(v) for v in np.arange(start, stop, step))
    if not sigmas:
        raise ValueError(f"{name} range produced no sigma values.")
    return sigmas


def _spatial_depth_from_sigmas(sample_arr, axes, kwargs_in):
    sigmas = kwargs_in.get("sigmas", ())
    max_sigma = max(float(s) for s in sigmas) if sigmas else 0.0
    depth = int(np.ceil(max_sigma * 6.0))
    ndim = int(getattr(sample_arr, "ndim", 0) or 0)
    if depth <= 0 or ndim <= 0:
        return tuple(0 for _ in range(ndim))
    if isinstance(axes, str) and len(axes) == ndim:
        return {i: depth for i, axis_name in enumerate(axes) if axis_name in ("Z", "Y", "X")}
    if ndim <= 2:
        return tuple(depth for _ in range(ndim))
    return tuple(0 if i < ndim - 2 else depth for i in range(ndim))


def _ensure_numpy_full_image(image):
    try:
        import dask.array as da

        if isinstance(image, da.Array):
            image = image.compute()
    except Exception:
        pass
    try:
        import cupy as cp

        if isinstance(image, cp.ndarray):
            return cp.asnumpy(image)
    except Exception:
        pass
    return image

# --- ALREADY IMPLEMENTED --- #
# - hysteresis threshold      #
# - butterworth               #
# - correlate sparse          #
# - difference of gaussians   #
# - farid                     #
# - farid horizontal          #
# - farid vertical            #
# - filter forward            #
# - filter inverse            #
# - frangi                    #
# - gabor                     #
# - gaussian blur             #
# - median filter             #
# - sobel filter              #
# - sobel split               #
# - prewitt filter            #
# - unsharp mask              #
# - hessian                   #
# - laplace                   #
# - meijering                 #
# - rank order                #
# - roberts                   #
# - sato                      #
# - scharr                    #
# - threshold otsu            #
# - threshold local           #
# - threshold niblack         #
# - threshold sauvola         #
# - threshold li              #
# - threshold mean            #
# - threshold minimum         #
# - threshold triangle        #
# - threshold yen             #
# - threshold isodata         #
# - wiener                    #
# --- --- --- --- --- --- --- #

# --- HYSTERESIS THRESHOLD ---
@register_node(
    label="Hysteresis Threshold",
    category="Filters",
    outputs=["image_out"],
    params_config={
        "low": {"min": 0.0, "max": 255.0, "step": 1},
        "high": {"min": 0.0, "max": 255.0, "step": 1}
    }
)
def hysteresis_threshold(image, low: float = 128, high: float = 255):
    """Wraps skimage.filters.apply_hysteresis_threshold"""
    return skimage.filters.apply_hysteresis_threshold(image, low=low, high=high)

# --- BUTTERWORTH ---
# FFT-based: Dask chunked execution gives per-chunk spectra, not the full-image
# spectrum, so we force CPU and apply output_dtype_policy only.
@register_node(
    label="Butterworth",
    category="Filters",
    description="Frequency-domain Butterworth filter. Runs on CPU only (FFT requires the full image).",
    outputs=["image_out"],
    input_types={"image": "image"},
    output_types={"image_out": "image"},
    params_config={
        "cutoff_frequency_ratio": {"min": 0.0, "max": 0.5, "step": 0.001},
        "order": {"min": 1, "max": 10}
    }
)
def butterworth(image, cutoff_frequency_ratio: float = 0.005, order: int = 2):
    return dispatch(
        default=skimage.filters.butterworth,
        args=(image,),
        kwargs={"cutoff_frequency_ratio": cutoff_frequency_ratio, "order": order},
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        # No dask_options: FFT on independent chunks gives wrong spectral results.
    )

# --- CORRELATE SPARSE ---
@register_node(
    label="Correlate Sparse",
    category="Filters",
    outputs=["image_out"],
    params_config={
        "mode": {"options": ['reflect', 'constant', 'nearest', 'mirror', 'wrap']}
    }
)
def correlate_sparse(image, kernel ,mode: str = 'reflect'):
    """Wraps skimage.filters.correlate_sparse"""
    return skimage.filters.correlate_sparse(image, kernel, mode=mode)

# --- DIFFERENCE OF GAUSSIANS ---
@register_node(
    label="Difference of Gaussians",
    category="Filters",
    description="Band-pass filter via subtraction of two Gaussian smoothings. Supports Dask.",
    outputs=["image_out"],
    input_types={"image": "image"},
    output_types={"image_out": "image"},
    params_config={
        "low_sigma": {"min": 0.0, "max": 20.0, "step": 0.1},
        "high_sigma": {"min": 0.0, "max": 20.0, "step": 0.1},
        "mode": {"options": ["nearest", "reflect", "wrap", "constant", "mirror"]}
    }
)
def difference_of_gaussians(image, low_sigma: float = 1.0, high_sigma: float = 2.0, mode: str = 'nearest'):
    return dispatch(
        default=skimage.filters.difference_of_gaussians,
        args=(image,),
        kwargs={"low_sigma": low_sigma, "high_sigma": high_sigma, "mode": mode},
        cuda_func=_cuda_difference_of_gaussians,
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "halo_from_param": "high_sigma",  # halo scaled from the larger sigma
            "boundary_from_param": "mode",
            "numpy_chunks": "spatial_auto",
        }
    )

# --- FARID ---
@register_node(
    label="Farid",
    category="Filters",
    description="Farid & Simoncelli gradient magnitude. Supports Dask.",
    outputs=["image_out"],
    input_types={"image": "image"},
    output_types={"image_out": "image"},
    params_config={
        "mode": {"options": ["nearest", "reflect", "wrap", "constant", "mirror"]}
    }
)
def farid(image, mode: str = 'reflect'):
    return dispatch(
        default=skimage.filters.farid,
        args=(image,),
        kwargs={"mode": mode},
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "boundary_from_param": "mode",
            "numpy_chunks": "spatial_auto",
            "map_overlap": {"depth": 4},  # 7-tap derivative filter
        }
    )

# --- FARID HORIZONTAL ---
@register_node(
    label="Farid Horizontal",
    category="Filters",
    description="Farid & Simoncelli horizontal derivative. Supports Dask.",
    outputs=["image_out"],
    input_types={"image": "image"},
    output_types={"image_out": "image"},
    params_config={}
)
def farid_horizontal(image):
    return dispatch(
        default=skimage.filters.farid_h,
        args=(image,),
        kwargs={},
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "numpy_chunks": "spatial_auto",
            "map_overlap": {"depth": 4},  # 7-tap derivative filter
        }
    )

# --- FARID VERTICAL ---
@register_node(
    label="Farid Vertical",
    category="Filters",
    description="Farid & Simoncelli vertical derivative. Supports Dask.",
    outputs=["image_out"],
    input_types={"image": "image"},
    output_types={"image_out": "image"},
    params_config={}
)
def farid_vertical(image):
    return dispatch(
        default=skimage.filters.farid_v,
        args=(image,),
        kwargs={},
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "numpy_chunks": "spatial_auto",
            "map_overlap": {"depth": 4},  # 7-tap derivative filter
        }
    )


# --- FILTER FORWARD ---
@register_node(
    label="Filter Forward",
    category="Filters",
    outputs=["image_out"],
    params_config={
    }
)
def filter_forward(image):
    """Wraps skimage.filters.filter_forward"""
    return skimage.filters.filter_forward(image)

# --- FILTER INVERSE ---
@register_node(
    label="Filter Inverse",
    category="Filters",
    outputs=["image_out"],
    params_config={
        "max_gain":{"min": 0.0, "max": 10, "step": 0.1}
    }
)
def filter_inverse(image, max_gain: float = 2.0):
    """Wraps skimage.filters.filter_inverse"""
    return skimage.filters.filter_inverse(image, max_gain=max_gain)

# --- FRANGI ---
@register_node(
    label="Frangi",
    category="Filters",
    description="Frangi vesselness filter (multi-scale). Full-image CPU because default gamma depends on a global Hessian norm.",
    outputs=["image_out"],
    input_types={"image": "image"},
    output_types={"image_out": "image"},
    params_config={
        "sigma_range_low": {"min": 0.0, "max": 20.0, "step": 0.1},
        "sigma_range_high": {"min": 0.0, "max": 20.0, "step": 0.1},
        "sigma_step": {"min": 0.0, "max": 5.0, "step": 1},
        "alpha": {"min": 0.0, "max": 20.0, "step": 0.1},
        "beta": {"min": 0.0, "max": 20.0, "step": 0.1},
        "mode": {"options": ["nearest", "reflect", "wrap", "constant", "mirror"]}
    }
)
def frangi(image, sigma_range_low: float = 1.0, sigma_range_high: float = 10.0, sigma_step: int = 2, alpha: float = 0.5, beta: float = 0.5, mode: str = 'reflect'):
    sigmas = _sigma_sequence(sigma_range_low, sigma_range_high, sigma_step, name="Frangi sigmas")

    def _cpu_frangi_full_image(image, sigmas, alpha, beta, mode):
        image = _ensure_numpy_full_image(image)
        return skimage.filters.frangi(
            image,
            sigmas=sigmas,
            alpha=alpha,
            beta=beta,
            mode=mode,
        )

    _cpu_frangi_full_image.__name__ = "frangi"

    return dispatch(
        default=_cpu_frangi_full_image,
        args=(image,),
        kwargs={"sigmas": sigmas, "alpha": alpha, "beta": beta, "mode": mode},
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
    )

# --- GABOR ---
@register_node(
    label="Gabor",
    category="Filters",
    outputs=["image_out"],
    params_config={
        "frequency": {"min": 0.0, "max": 20.0, "step": 0.1},
        "theta": {"min": 0.0, "max": 20.0, "step": 0.1},
        "mode": {"options": ["nearest", "reflect", "wrap", "constant", "mirror"]}
    }
)
def gabor(image, frequency: float = 1.0, theta: float = 0.0, mode: str = 'reflect'):
    """Wraps skimage.filters.gabor"""
    return skimage.filters.gabor(image, frequency=frequency, theta=theta, mode=mode)



# --- BACKEND 2: NUMPY (The Main Node) ---
@register_node(
    label="Gaussian Blur",
    category="Filters",
    description="Applies Gaussian smoothing. Auto-selects Dask+CUDA, Dask, then CPU.",
    outputs=["image_out"],
    input_types={"image": "image"},
    output_types={"image_out": "image"},
    params_config={
        "sigma": {"min": 0.0, "max": 20.0, "step": 0.1},
        "mode": {"options": ["nearest", "reflect", "wrap", "constant"]}
    }
)
def gaussian_blur(image, sigma: float = 1.0, mode: str = "nearest"):
    out = dispatch(
        default=skimage.filters.gaussian,
        args=(image,),
        kwargs={"sigma": sigma, "mode": mode, "preserve_range": True},
        # Inline CUDA mapping: dispatch resolves this callable lazily and
        # maps only the selected parameters from this node call.
        cuda_function="cupyx.scipy.ndimage.gaussian_filter",
        cuda_arg_names=["image"],
        cuda_kwarg_names=["sigma", "mode"],
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        # per_level: apply the blur independently to each pyramid level.
        # This keeps every level as a lazy dask array backed by its own zarr
        # resolution group, so napari can stream the right level on zoom.
        # "from_level0" would chain all levels off the full-res computation,
        # making the coarse levels impossibly expensive to render.
        pyramid_strategy="per_level",
        # Keep blur radius consistent in world units across pyramid levels.
        # For a 2x downsample pyramid this becomes sigma, sigma/2, sigma/4, ...
        pyramid_param_policy={"sigma": "fixed_world"},
        dask_options={
            # Keep Dask controls together for easier tuning/debugging.
            "strategy": "neighborhood",
            "halo_from_param": "sigma",
            "boundary_from_param": "mode",
            "independent_axes_param": "sigma",
            "numpy_chunks": "spatial_auto",
            # Optional explicit overlap override example:
            # "map_overlap": {"depth": {"t": 0, "y": 2, "x": 2}, "boundary": "none"},
        }
    )

    # IMPORTANT: avoid overwriting the source layer
    return (out)


# --- MEDIAN FILTER ---
@register_node(
    label="Median Filter",
    category="Filters",
    description="Median filter with disk footprint. Supports Dask.",
    outputs=["image_out"],
    input_types={"image": "image"},
    output_types={"image_out": "image"},
    params_config={
        "radius": {"min": 1, "max": 50}
    }
)
def median_filter(image, radius: int = 2):
    footprint = skimage.morphology.disk(radius)
    # Bake footprint into the callable so it never appears in kwargs.
    # dispatch's maybe_promote_numpy_to_dask converts every numpy array in
    # kwargs to a Dask array with image-shaped chunks, which breaks non-image
    # array arguments like footprint.
    fn = functools.partial(skimage.filters.median, footprint=footprint)

    def _cuda_median(image):
        import cupyx.scipy.ndimage as cnd
        return cnd.median_filter(image, footprint=footprint)

    return dispatch(
        default=fn,
        args=(image,),
        kwargs={},
        cuda_func=_cuda_median,
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "numpy_chunks": "spatial_auto",
            "map_overlap": {"depth": radius},
        }
    )


# --- SOBEL FILTER ---
@register_node(
    label="Sobel Edge Det.",
    category="Filters",
    description="Sobel gradient magnitude edge detector. Supports Dask.",
    outputs=["edges"],
    input_types={"image": "image"},
    output_types={"edges": "image"},
    params_config={
        "mode": {"options": ["reflect", "constant", "nearest", "mirror", "wrap"]}
    }
)
def sobel_filter(image, mode: str = 'reflect'):
    return dispatch(
        default=skimage.filters.sobel,
        args=(image,),
        kwargs={"mode": mode},
        cuda_func=_cuda_sobel,
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "boundary_from_param": "mode",
            "numpy_chunks": "spatial_auto",
            "map_overlap": {"depth": 1},  # 3x3 Sobel kernel
        }
    )


# --- SOBEL SPLIT ---
@register_node(
    label="Sobel Split",
    category="Filters",
    # LIST MULTIPLE OUTPUTS HERE:
    outputs=["horizontal_edges", "vertical_edges"]
)
def sobel_split(image):
    # Calculate both
    h_edges = skimage.filters.sobel_h(image)
    v_edges = skimage.filters.sobel_v(image)

    return h_edges, v_edges

# --- PREWITT FILTER ---
@register_node(
    label="Prewitt Edge Det.",
    category="Filters",
    description="Prewitt gradient magnitude edge detector. Supports Dask.",
    outputs=["edges"],
    input_types={"image": "image"},
    output_types={"edges": "image"},
    params_config={
        "mode": {"options": ["nearest", "reflect", "wrap", "constant"]}
    }
)
def prewitt_filter(image, mode: str = 'reflect'):
    return dispatch(
        default=skimage.filters.prewitt,
        args=(image,),
        kwargs={"mode": mode},
        cuda_func=_cuda_prewitt,
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "boundary_from_param": "mode",
            "numpy_chunks": "spatial_auto",
            "map_overlap": {"depth": 1},  # 3x3 Prewitt kernel
        }
    )

# --- UNSHARP MASK (SHARPEN) ---
@register_node(
    label="Unsharp Mask (Sharpen)",
    category="Filters",
    description="Sharpens image via unsharp masking (image - blurred). Supports Dask.",
    outputs=["sharpened"],
    input_types={"image": "image"},
    output_types={"sharpened": "image"},
    params_config={
        "radius": {"min": 0.0, "max": 20.0, "step": 0.5},
        "amount": {"min": 0.0, "max": 5.0, "step": 0.1}
    }
)
def unsharp_mask(image, radius: float = 1.0, amount: float = 1.0):
    return dispatch(
        default=skimage.filters.unsharp_mask,
        args=(image,),
        kwargs={"radius": radius, "amount": amount},
        cuda_func=_cuda_unsharp_mask,
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "halo_from_param": "radius",  # radius is the internal Gaussian sigma
            "numpy_chunks": "spatial_auto",
        }
    )

# --- HESSIAN ---
@register_node(
    label="Hessian (Ridge)",
    category="Filters",
    description="Hessian-based ridge filter (multi-scale). Auto-selects Dask+CUDA, Dask, then CPU.",
    outputs=["image_out"],
    input_types={"image": "image"},
    output_types={"image_out": "image"},
    params_config={
        "sigmas_min": {"min": 0.1, "max": 20.0},
        "sigmas_max": {"min": 1.0, "max": 50.0},
        "mode": {"options": ["reflect", "constant", "nearest", "mirror", "wrap"]},
        "black_ridges": {"type": "bool"}
    }
)
def hessian(image, sigmas_min: float = 1.0, sigmas_max: float = 10.0, mode: str = 'reflect', black_ridges: bool = True):
    sigmas = _sigma_sequence(sigmas_min, sigmas_max, 2.0, name="Hessian sigmas")
    return dispatch(
        default=skimage.filters.hessian,
        args=(image,),
        kwargs={"sigmas": sigmas, "mode": mode, "black_ridges": black_ridges},
        cuda_function="cucim.skimage.filters.hessian",
        cuda_arg_names=["image"],
        cuda_kwarg_names=["sigmas", "mode", "black_ridges"],
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "numpy_chunks": "spatial_auto",
            "map_overlap": {
                "depth": _spatial_depth_from_sigmas,
                "boundary": "none",
            },
        }
    )

# --- LAPLACE ---
@register_node(
    label="Laplace",
    category="Filters",
    description="Laplacian edge detection via discrete convolution. Supports Dask.",
    outputs=["image_out"],
    input_types={"image": "image"},
    output_types={"image_out": "image"},
    params_config={
        "ksize": {"min": 1, "max": 10, "step": 1}
    }
)
def laplace(image, ksize: int = 3):
    return dispatch(
        default=skimage.filters.laplace,
        args=(image,),
        kwargs={"ksize": ksize},
        # cupyx.scipy.ndimage.laplace has no ksize — always uses a 3-point
        # stencil. On the CUDA path ksize is intentionally not forwarded;
        # the dispatcher preflight will fall back to Dask if the call fails.
        cuda_function="cupyx.scipy.ndimage.laplace",
        cuda_arg_names=["image"],
        cuda_kwarg_names=[],
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "halo_from_param": "ksize",
            "halo_factor": 0.5,  # halo = ceil(ksize * 0.5) ≈ kernel radius
            "numpy_chunks": "spatial_auto",
        }
    )

# --- MEIJERING ---
@register_node(
    label="Meijering (Ridge)",
    category="Filters",
    description="Meijering neuriteness filter (multi-scale). Full-image CPU because the algorithm normalizes by global response maxima.",
    outputs=["image_out"],
    input_types={"image": "image"},
    output_types={"image_out": "image"},
    params_config={
        "sigmas_min": {"min": 0.1, "max": 20.0, "step": 0.1},
        "sigmas_max": {"min": 1.0, "max": 50.0, "step": 1},
        "sigmas_step": {"min": 1, "max": 10, "step": 1},
        "mode": {"options": ["reflect", "constant", "nearest", "mirror", "wrap"]},
        "black_ridges": {"type": "bool"}
    }
)
def meijering(image, sigmas_min: float = 1.0, sigmas_max: float = 10.0, sigmas_step: int = 1, mode: str = 'reflect', black_ridges: bool = True):
    sigmas = _sigma_sequence(sigmas_min, sigmas_max, sigmas_step, name="Meijering sigmas")

    def _cpu_meijering_full_image(image, sigmas, mode, black_ridges):
        image = _ensure_numpy_full_image(image)
        return skimage.filters.meijering(
            image,
            sigmas=sigmas,
            mode=mode,
            black_ridges=black_ridges,
        )

    _cpu_meijering_full_image.__name__ = "meijering"

    return dispatch(
        default=_cpu_meijering_full_image,
        args=(image,),
        kwargs={"sigmas": sigmas, "mode": mode, "black_ridges": black_ridges},
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
    )

# --- RANK ORDER ---
@register_node(
    label="Rank Order",
    category="Filters",
    outputs=["ordered_image", "original_values"]
)
def rank_order(image):
    return skimage.filters.rank_order(image)

# --- ROBERTS ---
@register_node(
    label="Roberts",
    category="Filters",
    description="Roberts cross gradient edge detector. Supports Dask.",
    outputs=["image_out"],
    input_types={"image": "image"},
    output_types={"image_out": "image"},
)
def roberts(image):
    return dispatch(
        default=skimage.filters.roberts,
        args=(image,),
        kwargs={},
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "numpy_chunks": "spatial_auto",
            "map_overlap": {"depth": 1},  # 2x2 Roberts cross kernel
        }
    )

# --- SATO ---
@register_node(
    label="Sato (Ridge)",
    category="Filters",
    description="Sato tubeness filter (multi-scale). Auto-selects Dask+CUDA, Dask, then CPU.",
    outputs=["image_out"],
    input_types={"image": "image"},
    output_types={"image_out": "image"},
    params_config={
        "sigmas_min": {"min": 0.1, "max": 20.0},
        "sigmas_max": {"min": 1.0, "max": 50.0},
        "mode": {"options": ["reflect", "constant", "nearest", "mirror", "wrap"]},
        "black_ridges": {"type": "bool"}
    }
)
def sato(image, sigmas_min: float = 1.0, sigmas_max: float = 10.0, mode: str = 'reflect', black_ridges: bool = True):
    sigmas = _sigma_sequence(sigmas_min, sigmas_max, 2.0, name="Sato sigmas")
    return dispatch(
        default=skimage.filters.sato,
        args=(image,),
        kwargs={"sigmas": sigmas, "mode": mode, "black_ridges": black_ridges},
        cuda_function="cucim.skimage.filters.sato",
        cuda_arg_names=["image"],
        cuda_kwarg_names=["sigmas", "mode", "black_ridges"],
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "numpy_chunks": "spatial_auto",
            "map_overlap": {
                "depth": _spatial_depth_from_sigmas,
                "boundary": "none",
            },
        }
    )

# --- SCHARR ---
@register_node(
    label="Scharr",
    category="Filters",
    description="Scharr gradient magnitude edge detector. Supports Dask.",
    outputs=["image_out"],
    input_types={"image": "image"},
    output_types={"image_out": "image"},
    params_config={
        "mode": {"options": ["reflect", "constant", "nearest", "mirror", "wrap"]}
    }
)
def scharr(image, mode: str = 'reflect'):
    return dispatch(
        default=skimage.filters.scharr,
        args=(image,),
        kwargs={"mode": mode},
        output_dtype_policy="image_float",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "boundary_from_param": "mode",
            "numpy_chunks": "spatial_auto",
            "map_overlap": {"depth": 1},  # 3x3 Scharr kernel
        }
    )

# Global threshold nodes intentionally do not declare a Dask block strategy:
# the scalar threshold must be computed per full spatial image/frame, never per
# chunk. The postprocess step can still keep the final mask lazy for Dask input.

# --- THRESHOLD OTSU ---
@register_node(
    label="Threshold Otsu",
    category="Filters",
    description="Global Otsu threshold mask. Computes the scalar threshold on the full spatial image, then applies a lazy mask when possible.",
    outputs=["mask_out"],
    input_types={"image": "image"},
    output_types={"mask_out": "image"},
    params_config={
        "nbins": {"min": 2, "max": 1024}
    }
)
def threshold_otsu(image, nbins: int = 256):
    def _cpu_threshold_otsu_global(image, nbins):
        image = _ensure_numpy_full_image(image)
        return skimage.filters.threshold_otsu(image, nbins=nbins)

    _cpu_threshold_otsu_global.__name__ = "threshold_otsu"
    return dispatch(
        default=_cpu_threshold_otsu_global,
        args=(image,),
        kwargs={"nbins": nbins},
        cuda_function="cucim.skimage.filters.threshold_otsu",
        cuda_arg_names=["image"],
        cuda_kwarg_names=["nbins"],
        postprocess=_threshold_to_mask,
        output_dtype_policy="bool",
        backend="auto",
        gpu_min_nbytes=0,
        pyramid_strategy="per_level",
    )

# --- THRESHOLD LOCAL ---
@register_node(
    label="Threshold Local",
    category="Filters",
    description="Adaptive local threshold mask. Auto-selects Dask+CUDA, Dask, then CPU.",
    outputs=["mask_out"],
    input_types={"image": "image"},
    output_types={"mask_out": "image"},
    params_config={
        "block_size": {"min": 3, "max": 101, "step": 2},
        "method": {"options": ["gaussian", "mean", "median"]},
        "offset": {"min": -1.0, "max": 1.0, "step": 0.01},
        "mode": {"options": ["reflect", "constant", "nearest", "mirror", "wrap"]}
    }
)
def threshold_local(image, block_size: int = 15, method: str = 'gaussian', offset: float = 0.0, mode: str = 'reflect'):
    block_size = _validate_odd_window(block_size, "block_size")
    return dispatch(
        default=skimage.filters.threshold_local,
        args=(image,),
        kwargs={
            "block_size": block_size,
            "method": method,
            "offset": offset,
            "mode": mode,
        },
        cuda_function="cucim.skimage.filters.threshold_local",
        cuda_arg_names=["image"],
        cuda_kwarg_names=["block_size", "method", "offset", "mode"],
        postprocess=_threshold_to_mask,
        output_dtype_policy="bool",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "halo_from_param": "block_size",
            "halo_factor": 0.5,
            "numpy_chunks": "spatial_auto",
            # The skimage/cuCIM threshold function handles image borders.
            # Dask only supplies real neighboring pixels across chunk borders.
            "map_overlap": {"boundary": "none"},
        },
    )

# --- THRESHOLD NIBLACK ---
@register_node(
    label="Threshold Niblack",
    category="Filters",
    description="Niblack local threshold mask. Auto-selects Dask+CUDA, Dask, then CPU.",
    outputs=["mask_out"],
    input_types={"image": "image"},
    output_types={"mask_out": "image"},
    params_config={
        "window_size": {"min": 3, "max": 101, "step": 2},
        "k": {"min": 0.0, "max": 1.0, "step": 0.01}
    }
)
def threshold_niblack(image, window_size: int = 15, k: float = 0.2):
    window_size = _validate_odd_window(window_size, "window_size")
    return dispatch(
        default=skimage.filters.threshold_niblack,
        args=(image,),
        kwargs={"window_size": window_size, "k": k},
        cuda_function="cucim.skimage.filters.threshold_niblack",
        cuda_arg_names=["image"],
        cuda_kwarg_names=["window_size", "k"],
        postprocess=_threshold_to_mask,
        output_dtype_policy="bool",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "halo_from_param": "window_size",
            "halo_factor": 0.5,
            "numpy_chunks": "spatial_auto",
            "map_overlap": {"boundary": "none"},
        },
    )

# --- THRESHOLD SAUVOLA ---
@register_node(
    label="Threshold Sauvola",
    category="Filters",
    description="Sauvola local threshold mask. Auto-selects Dask+CUDA, Dask, then CPU.",
    outputs=["mask_out"],
    input_types={"image": "image"},
    output_types={"mask_out": "image"},
    params_config={
        "window_size": {"min": 3, "max": 101, "step": 2},
        "k": {"min": 0.0, "max": 1.0, "step": 0.01}
    }
)
def threshold_sauvola(image, window_size: int = 15, k: float = 0.2):
    window_size = _validate_odd_window(window_size, "window_size")
    return dispatch(
        default=skimage.filters.threshold_sauvola,
        args=(image,),
        kwargs={"window_size": window_size, "k": k},
        cuda_function="cucim.skimage.filters.threshold_sauvola",
        cuda_arg_names=["image"],
        cuda_kwarg_names=["window_size", "k"],
        postprocess=_threshold_to_mask,
        output_dtype_policy="bool",
        backend="auto",
        gpu_min_nbytes=0,
        dask_options={
            "strategy": "neighborhood",
            "halo_from_param": "window_size",
            "halo_factor": 0.5,
            "numpy_chunks": "spatial_auto",
            "map_overlap": {"boundary": "none"},
        },
    )

# --- THRESHOLD LI ---
@register_node(
    label="Threshold Li",
    category="Filters",
    description="Global Li threshold mask. Computes the scalar threshold on the full spatial image, then applies a lazy mask when possible.",
    outputs=["mask_out"],
    input_types={"image": "image"},
    output_types={"mask_out": "image"},
    params_config={}
)
def threshold_li(image):
    def _cpu_threshold_li_global(image):
        image = _ensure_numpy_full_image(image)
        return skimage.filters.threshold_li(image)

    _cpu_threshold_li_global.__name__ = "threshold_li"
    return dispatch(
        default=_cpu_threshold_li_global,
        args=(image,),
        kwargs={},
        cuda_function="cucim.skimage.filters.threshold_li",
        cuda_arg_names=["image"],
        cuda_kwarg_names=[],
        postprocess=_threshold_to_mask,
        output_dtype_policy="bool",
        backend="auto",
        gpu_min_nbytes=0,
        pyramid_strategy="per_level",
    )

# --- THRESHOLD MEAN ---
@register_node(
    label="Threshold Mean",
    category="Filters",
    description="Global mean threshold mask. Computes one scalar threshold per spatial image, then applies a lazy mask when possible.",
    outputs=["mask_out"],
    input_types={"image": "image"},
    output_types={"mask_out": "image"},
    params_config={}
)
def threshold_mean(image):
    return dispatch(
        default=skimage.filters.threshold_mean,
        args=(image,),
        kwargs={},
        cuda_func=_cuda_threshold_mean,
        postprocess=_threshold_to_mask,
        output_dtype_policy="bool",
        backend="auto",
        gpu_min_nbytes=0,
        pyramid_strategy="per_level",
    )

# --- THRESHOLD MINIMUM ---
@register_node(
    label="Threshold Minimum",
    category="Filters",
    description="Global minimum histogram threshold mask. Computes the scalar threshold on the full spatial image, then applies a lazy mask when possible.",
    outputs=["mask_out"],
    input_types={"image": "image"},
    output_types={"mask_out": "image"},
    params_config={
        "nbins": {"min": 2, "max": 1024}
    }
)
def threshold_minimum(image, nbins: int = 256):
    def _cpu_threshold_minimum_global(image, nbins):
        image = _ensure_numpy_full_image(image)
        return skimage.filters.threshold_minimum(image, nbins=nbins)

    _cpu_threshold_minimum_global.__name__ = "threshold_minimum"
    return dispatch(
        default=_cpu_threshold_minimum_global,
        args=(image,),
        kwargs={"nbins": nbins},
        cuda_function="cucim.skimage.filters.threshold_minimum",
        cuda_arg_names=["image"],
        cuda_kwarg_names=["nbins"],
        postprocess=_threshold_to_mask,
        output_dtype_policy="bool",
        backend="auto",
        gpu_min_nbytes=0,
        pyramid_strategy="per_level",
    )

# --- THRESHOLD TRIANGLE ---
@register_node(
    label="Threshold Triangle",
    category="Filters",
    description="Global triangle histogram threshold mask. Computes the scalar threshold on the full spatial image, then applies a lazy mask when possible.",
    outputs=["mask_out"],
    input_types={"image": "image"},
    output_types={"mask_out": "image"},
    params_config={
        "nbins": {"min": 2, "max": 1024}
    }
)
def threshold_triangle(image, nbins: int = 256):
    def _cpu_threshold_triangle_global(image, nbins):
        image = _ensure_numpy_full_image(image)
        return skimage.filters.threshold_triangle(image, nbins=nbins)

    _cpu_threshold_triangle_global.__name__ = "threshold_triangle"
    return dispatch(
        default=_cpu_threshold_triangle_global,
        args=(image,),
        kwargs={"nbins": nbins},
        cuda_function="cucim.skimage.filters.threshold_triangle",
        cuda_arg_names=["image"],
        cuda_kwarg_names=["nbins"],
        postprocess=_threshold_to_mask,
        output_dtype_policy="bool",
        backend="auto",
        gpu_min_nbytes=0,
        pyramid_strategy="per_level",
    )

# --- THRESHOLD YEN ---
@register_node(
    label="Threshold Yen",
    category="Filters",
    description="Global Yen threshold mask. Computes the scalar threshold on the full spatial image, then applies a lazy mask when possible.",
    outputs=["mask_out"],
    input_types={"image": "image"},
    output_types={"mask_out": "image"},
    params_config={
        "nbins": {"min": 2, "max": 1024}
    }
)
def threshold_yen(image, nbins: int = 256):
    def _cpu_threshold_yen_global(image, nbins):
        image = _ensure_numpy_full_image(image)
        return skimage.filters.threshold_yen(image, nbins=nbins)

    _cpu_threshold_yen_global.__name__ = "threshold_yen"
    return dispatch(
        default=_cpu_threshold_yen_global,
        args=(image,),
        kwargs={"nbins": nbins},
        cuda_function="cucim.skimage.filters.threshold_yen",
        cuda_arg_names=["image"],
        cuda_kwarg_names=["nbins"],
        postprocess=_threshold_to_mask,
        output_dtype_policy="bool",
        backend="auto",
        gpu_min_nbytes=0,
        pyramid_strategy="per_level",
    )

# --- THRESHOLD ISODATA ---
@register_node(
    label="Threshold Isodata",
    category="Filters",
    description="Global Isodata threshold mask. Computes the scalar threshold on the full spatial image, then applies a lazy mask when possible.",
    outputs=["mask_out"],
    input_types={"image": "image"},
    output_types={"mask_out": "image"},
    params_config={
        "nbins": {"min": 2, "max": 1024}
    }
)
def threshold_isodata(image, nbins: int = 256):
    def _cpu_threshold_isodata_global(image, nbins):
        image = _ensure_numpy_full_image(image)
        return skimage.filters.threshold_isodata(image, nbins=nbins)

    _cpu_threshold_isodata_global.__name__ = "threshold_isodata"
    return dispatch(
        default=_cpu_threshold_isodata_global,
        args=(image,),
        kwargs={"nbins": nbins},
        cuda_function="cucim.skimage.filters.threshold_isodata",
        cuda_arg_names=["image"],
        cuda_kwarg_names=["nbins"],
        postprocess=_threshold_to_mask,
        output_dtype_policy="bool",
        backend="auto",
        gpu_min_nbytes=0,
        pyramid_strategy="per_level",
    )

# --- WIENER ---
@register_node(
    label="Wiener Deconvolution",
    category="Filters",
    outputs=["image_out"],
    params_config={
        "balance": {"min": 0.0, "max": 100.0, "step": 0.1},
        "clip": {"type": "bool"}
    }
)
def wiener(image, psf, balance: float = 0.25, clip: bool = True):
    # psf is impulse_response. It must be an image input.
    return skimage.filters.wiener(image, impulse_response=psf, K=balance, clip=clip)
