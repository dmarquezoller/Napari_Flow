from .decorator import register_node
import skimage.exposure
import numpy as np
import dask.array as da

# --- ALREADY IMPLEMENTED --- #
# - adjust gamma              #
# - adjust log                #
# - adjust sigmoid            #
# - cummulative distribution  #
# - equalize adaptive         #
# - equalize histogram        #
# - rescale intensity         #
# --- --- --- --- --- --- --- #

# --- CONVERT TO GRAYSCALE ---
def _rgb_to_gray(image):
    """Collapse trailing channel dim using ITU-R BT.709 luminance weights.

    Supports NumPy and Dask. Returns the image unchanged if the last dim is
    not 3 or 4 (already single-channel).
    """
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    if isinstance(image, da.Array):
        n = image.shape[-1]
        if n not in (3, 4):
            return image
        w = weights[:n]
        return da.map_blocks(
            lambda c: np.tensordot(c.astype(np.float32), w, axes=[[-1], [0]]),
            image,
            drop_axis=image.ndim - 1,
            dtype=np.float32,
        )
    arr = np.asarray(image, dtype=np.float32)
    n = arr.shape[-1]
    if n not in (3, 4):
        return arr
    return np.tensordot(arr, weights[:n], axes=[[-1], [0]])


@register_node(
    label="Convert to Grayscale",
    category="Exposure",
    description="Collapse RGB/RGBA images to a single luminance channel using ITU-R BT.709 weights. Supports Dask arrays (lazy). Images whose last dim is not 3 or 4 pass through unchanged.",
    outputs=["grayscale"],
    input_types={"image": "image"},
    output_types={"grayscale": "image"},
)
def convert_to_grayscale(image):
    return _rgb_to_gray(image)


# --- ADJUST GAMMA ---
@register_node(
    label="Adjust Gamma",
    category="Exposure",
    outputs=["adjusted"],
    params_config={
        "gamma": {"min": 0.1, "max": 3.0, "step": 0.1},
        "gain": {"min": 0.1, "max": 3.0, "step": 0.1}
    }
)
def adjust_gamma(image, gamma: float = 1.0, gain: float = 1.0):
    return skimage.exposure.adjust_gamma(image, gamma, gain)

# --- ADJUST LOG ---
@register_node(
    label="Adjust Log",
    category="Exposure",
    outputs=["adjusted"],
    params_config={
        "gain": {"min": 0.1, "max": 3.0, "step": 0.1},
        "inv": {"type": "bool", "default": False}
    }
)
def adjust_log(image, gain: float = 1.0, inv: bool = False):
    return skimage.exposure.adjust_log(image, gain, inv)

# --- ADJUST SIGMOID ---
@register_node(
    label="Adjust Sigmoid",
    category="Exposure",
    outputs=["adjusted"],
    params_config={
        "cut_off": {"min": 0.0, "max": 1.0, "step": 0.1},
        "gain": {"min": 1.0, "max": 20.0, "step": 0.5},
        "inv": {"type": "bool", "default": False}
    }
)
def adjust_sigmoid(image, cut_off: float = 0.5, gain: float = 10, inv: bool = False):
    return skimage.exposure.adjust_sigmoid(image, cut_off, gain, inv)

# --- CUMMULATIVE DISTRIBUTION ---
@register_node(
    label="Cummulative Distribution",
    category="Exposure",
    outputs=["adjusted"],
    params_config={
        "nbins": {"min": 1, "max": 256, "step": 1}
    }
)
def cummulative_distribution(image, nbins: int = 256):
    return skimage.exposure.cummulative_distribution(image, nbins)

# --- EQUALIZE ADAPTHIST ---
@register_node(
    label="Equalize Adaptive",
    category="Exposure",
    outputs=["adjusted"],
    params_config={
        "nbins": {"min": 1, "max": 256, "step": 1},
        "clip_limit": {"min": 0.0, "max": 1.0, "step": 0.1}
    }
)
def equalize_adapthist(image, nbins: int = 256, clip_limit: float = 0.01):
    return skimage.exposure.equalize_adapthist(image, nbins, clip_limit)

# --- EQUALIZE HISTOGRAM ---
@register_node(
    label="Equalize Histogram",
    category="Exposure",
    outputs=["adjusted"],
    params_config={
        "nbins": {"min": 1, "max": 256, "step": 1}
    }
)
def equalize_histogram(image, nbins: int = 256):
    return skimage.exposure.equalize_histogram(image, nbins)

# --- RESCALE INTENSITY ---
@register_node(
    label="Rescale Intensity",
    category="Exposure",
    outputs=["adjusted"],
    params_config={
        "in_min": {"min": 0.0, "max": 1.0, "step": 0.1},
        "in_max": {"min": 0.0, "max": 1.0, "step": 0.1},
        "out_min": {"min": 0.0, "max": 1.0, "step": 0.1},
        "out_max": {"min": 0.0, "max": 1.0, "step": 0.1}
    }
)
def rescale_intensity(image, in_min: float = 0.0, in_max: float = 1.0, out_min: float = 0.0, out_max: float = 1.0):
    return skimage.exposure.rescale_intensity(image, tuple(in_min, in_max), tuple(out_min, out_max))







