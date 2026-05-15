from .decorator import register_node, dispatch, smart_compute
import skimage.filters
import numpy as np

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
@register_node(
    label="Butterworth",
    category="Filters",
    outputs=["image_out"],
    params_config={
        "cutoff_frequency_ratio": {"min": 0.0, "max": 0.5, "step": 0.001},
        "order": {"min": 1, "max": 10}
    }
)
def butterworth(image, cutoff_frequency_ratio: float = 0.005, order: int = 2):
    """Wraps skimage.filters.butterworth"""
    return skimage.filters.butterworth(image, cutoff_frequency_ratio=cutoff_frequency_ratio, order=order)

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
    outputs=["image_out"],
    params_config={
        "low_sigma": {"min": 0.0, "max": 20.0, "step": 0.1},
        "high_sigma": {"min": 0.0, "max": 20.0, "step": 0.1},
        "mode": {"options": ["nearest", "reflect", "wrap", "constant", "mirror"]}
    }
)
def difference_of_gaussians(image, low_sigma: float = 1.0, high_sigma: float = 2.0, mode: str = 'nearest'):
    """Wraps skimage.filters.difference_of_gaussians"""
    return skimage.filters.difference_of_gaussians(image, low_sigma=low_sigma, high_sigma=high_sigma, mode=mode)

# --- FARID ---
@register_node(
    label="Farid",
    category="Filters",
    outputs=["image_out"],
    params_config={
        "mode": {"options": ["nearest", "reflect", "wrap", "constant", "mirror"]}
    }
)
def farid(image, mode: str = 'reflect'):
    """Wraps skimage.filters.farid"""
    return skimage.filters.farid(image, mode=mode)

# --- FARID HORIZONTAL ---
@register_node(
    label="Farid Horizontal",
    category="Filters",
    outputs=["image_out"],
    params_config={
    }
)
def farid_horizontal(image):
    """Wraps skimage.filters.farid_h"""
    return skimage.filters.farid_h(image)

# --- FARID VERTICAL ---
@register_node(
    label="Farid Vertical",
    category="Filters",
    outputs=["image_out"],
    params_config={
    }
)
def farid_vertical(image):
    """Wraps skimage.filters.farid_v"""
    return skimage.filters.farid_v(image)


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
    outputs=["image_out"],
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
    """Wraps skimage.filters.frangi"""
    sigmas = np.arange(sigma_range_low, sigma_range_high, sigma_step)
    return skimage.filters.frangi(image, sigmas=sigmas, alpha=alpha, beta=beta, mode=mode)

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
    outputs=["image_out"],
    params_config={
        "radius": {"min":   1, "max": 50}
    }
)
def median_filter(image, radius: int = 2):
    """Wraps skimage.filters.median with a disk footprint"""
    footprint = skimage.morphology.disk(radius)
    return skimage.filters.median(image, footprint=footprint)


# --- SOBEL FILTER---
@register_node(
    label="Sobel Edge Det.",
    category="Filters",
    outputs=["edges"]
)
def sobel_filter(image):
    """Wraps skimage.filters.sobel"""
    return skimage.filters.sobel(image)


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

# --- PREWITT FILTER---
@register_node(
    label="Prewitt Edge Det.",
    category="Filters",
    outputs=["edges"],
    params_config={
        "mode": {"options": ["nearest", "reflect", "wrap", "constant"]}
    }
)
def prewitt_filter(image):
    """Wraps skimage.filters.prewitt"""
    return skimage.filters.prewitt(image, mode='reflect' )

# --- UNSHARP MASK (SHARPEN) ---
@register_node(
    label="Unsharp Mask (Sharpen)",
    category="Filters",
    outputs=["sharpened"],
    params_config={
        "radius": {"min": 0.0, "max": 20.0, "step": 0.5},
        "amount": {"min": 0.0, "max": 5.0, "step": 0.1}
    }
)
def unsharp_mask(image, radius: float = 1.0, amount: float = 1.0):
    return skimage.filters.unsharp_mask(image, radius=radius, amount=amount)

# --- HESSIAN ---
@register_node(
    label="Hessian (Ridge)",
    category="Filters",
    outputs=["image_out"],
    params_config={
        "sigmas_min": {"min": 0.1, "max": 20.0},
        "sigmas_max": {"min": 1.0, "max": 50.0},
        "mode": {"options": ["reflect", "constant", "nearest", "mirror", "wrap"]},
        "black_ridges": {"type": "bool"}
    }
)
def hessian(image, sigmas_min: float = 1.0, sigmas_max: float = 10.0, mode: str = 'reflect', black_ridges: bool = True):
    sigmas = range(int(sigmas_min), int(sigmas_max), 2)
    return skimage.filters.hessian(image, sigmas=sigmas, mode=mode, black_ridges=black_ridges)

# --- LAPLACE ---
@register_node(
    label="Laplace",
    category="Filters",
    outputs=["image_out"],
    params_config={
        "ksize": {"min": 1, "max": 10, "step": 1}
    }
)
def laplace(image, ksize: int = 3):
    return skimage.filters.laplace(image, ksize=ksize)

# --- MEIJERING ---
@register_node(
    label="Meijering (Ridge)",
    category="Filters",
    outputs=["image_out"],
    params_config={
        "sigmas_min": {"min": 0.1, "max": 20.0, "step": 0.1},
        "sigmas_max": {"min": 1.0, "max": 50.0, "step": 1},
        "sigmas_step": {"min": 1, "max": 10, "step": 1},
        "mode": {"options": ["reflect", "constant", "nearest", "mirror", "wrap"]},
        "black_ridges": {"type": "bool"}
    }
)
def meijering(image, sigmas_min: float = 1.0, sigmas_max: float = 10.0, sigmas_step: int = 1,mode: str = 'reflect', black_ridges: bool = True):
    sigmas = range(int(sigmas_min), int(sigmas_max), int(sigmas_step))
    return skimage.filters.meijering(image, sigmas=sigmas, mode=mode, black_ridges=black_ridges)

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
    outputs=["image_out"]
)
def roberts(image):
    return skimage.filters.roberts(image)

# --- SATO ---
@register_node(
    label="Sato (Ridge)",
    category="Filters",
    outputs=["image_out"],
    params_config={
        "sigmas_min": {"min": 0.1, "max": 20.0},
        "sigmas_max": {"min": 1.0, "max": 50.0},
        "mode": {"options": ["reflect", "constant", "nearest", "mirror", "wrap"]},
        "black_ridges": {"type": "bool"}
    }
)
def sato(image, sigmas_min: float = 1.0, sigmas_max: float = 10.0, mode: str = 'reflect', black_ridges: bool = True):
    sigmas = range(int(sigmas_min), int(sigmas_max), 2)
    return skimage.filters.sato(image, sigmas=sigmas, mode=mode, black_ridges=black_ridges)

# --- SCHARR ---
@register_node(
    label="Scharr",
    category="Filters",
    outputs=["image_out"],
    params_config={
        "mode": {"options": ["reflect", "constant", "nearest", "mirror", "wrap"]}
    }
)
def scharr(image, mode: str = 'reflect'):
    return skimage.filters.scharr(image, mode=mode)

# --- THRESHOLD OTSU ---
@register_node(
    label="Threshold Otsu",
    category="Filters",
    outputs=["mask_out"],
    params_config={
        "nbins": {"min": 2, "max": 1024}
    }
)
def threshold_otsu(image, nbins: int = 256):
    thresh = skimage.filters.threshold_otsu(image, nbins=nbins)
    return image > thresh

# --- THRESHOLD LOCAL ---
@register_node(
    label="Threshold Local",
    category="Filters",
    outputs=["mask_out"],
    params_config={
        "block_size": {"min": 3, "max": 101, "step": 2},
        "method": {"options": ["gaussian", "mean", "median"]},
        "offset": {"min": -1.0, "max": 1.0, "step": 0.01},
        "mode": {"options": ["reflect", "constant", "nearest", "mirror", "wrap"]}
    }
)
def threshold_local(image, block_size: int = 15, method: str = 'gaussian', offset: float = 0.0, mode: str = 'reflect'):
    thresh = skimage.filters.threshold_local(image, block_size=block_size, method=method, offset=offset, mode=mode)
    return image > thresh

# --- THRESHOLD NIBLACK ---
@register_node(
    label="Threshold Niblack",
    category="Filters",
    outputs=["mask_out"],
    params_config={
        "window_size": {"min": 3, "max": 101, "step": 2},
        "k": {"min": 0.0, "max": 1.0, "step": 0.01}
    }
)
def threshold_niblack(image, window_size: int = 15, k: float = 0.2):
    thresh = skimage.filters.threshold_niblack(image, window_size=window_size, k=k)
    return image > thresh

# --- THRESHOLD SAUVOLA ---
@register_node(
    label="Threshold Sauvola",
    category="Filters",
    outputs=["mask_out"],
    params_config={
        "window_size": {"min": 3, "max": 101, "step": 2},
        "k": {"min": 0.0, "max": 1.0, "step": 0.01}
    }
)
def threshold_sauvola(image, window_size: int = 15, k: float = 0.2):
    thresh = skimage.filters.threshold_sauvola(image, window_size=window_size, k=k)
    return image > thresh

# --- THRESHOLD LI ---
@register_node(
    label="Threshold Li",
    category="Filters",
    outputs=["mask_out"],
    params_config={}
)
def threshold_li(image):
    thresh = skimage.filters.threshold_li(image)
    return image > thresh

# --- THRESHOLD MEAN ---
@register_node(
    label="Threshold Mean",
    category="Filters",
    outputs=["mask_out"],
    params_config={}
)
def threshold_mean(image):
    thresh = skimage.filters.threshold_mean(image)
    return image > thresh

# --- THRESHOLD MINIMUM ---
@register_node(
    label="Threshold Minimum",
    category="Filters",
    outputs=["mask_out"],
    params_config={
        "nbins": {"min": 2, "max": 1024}
    }
)
def threshold_minimum(image, nbins: int = 256):
    thresh = skimage.filters.threshold_minimum(image, nbins=nbins)
    return image > thresh

# --- THRESHOLD TRIANGLE ---
@register_node(
    label="Threshold Triangle",
    category="Filters",
    outputs=["mask_out"],
    params_config={
        "nbins": {"min": 2, "max": 1024}
    }
)
def threshold_triangle(image, nbins: int = 256):
    thresh = skimage.filters.threshold_triangle(image, nbins=nbins)
    return image > thresh

# --- THRESHOLD YEN ---
@register_node(
    label="Threshold Yen",
    category="Filters",
    outputs=["mask_out"],
    params_config={
        "nbins": {"min": 2, "max": 1024}
    }
)
def threshold_yen(image, nbins: int = 256):
    thresh = skimage.filters.threshold_yen(image, nbins=nbins)
    return image > thresh

# --- THRESHOLD ISODATA ---
@register_node(
    label="Threshold Isodata",
    category="Filters",
    outputs=["mask_out"],
    params_config={
        "nbins": {"min": 2, "max": 1024}
    }
)
def threshold_isodata(image, nbins: int = 256):
    thresh = skimage.filters.threshold_isodata(image, nbins=nbins)
    return image > thresh

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
