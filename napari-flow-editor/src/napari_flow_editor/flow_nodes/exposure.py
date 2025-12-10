from .decorator import register_node
import skimage.exposure

# --- ALREADY IMPLEMENTED --- #
# - adjust gamma              #
# - adjust log                #
# - adjust sigmoid            #
# - cummulative distribution  #
# - equalize adaptive         #
# - equalize histogram        #
# - rescale intensity         #
# --- --- --- --- --- --- --- #

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







