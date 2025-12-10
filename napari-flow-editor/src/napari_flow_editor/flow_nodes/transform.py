from .decorator import register_node
import skimage.transform

# --- ALREADY IMPLEMENTED --- #
# - rotate                    #
# - rescale (zoom)            #
# - downscale (binning)       #
# - swirl                     #
# - warp polar (ring to line) #
# --- --- --- --- --- --- --- #

# ==========================================
#           GEOMETRIC TRANSFORMS
# ==========================================

# --- ROTATE IMAGE ---
@register_node(
    label="Rotate",
    category="Transform",
    outputs=["rotated"],
    params_config={
        "angle": {"min": -180, "max": 180, "step": 1},
        "resize": {"type": "bool", "default": True}
    }
)
def rotate_image(image, angle: float = 90.0, resize: bool = True):
    return skimage.transform.rotate(image, angle, resize=resize)

# --- RESCALE IMAGE ---
@register_node(
    label="Rescale (Zoom)",
    category="Transform",
    outputs=["rescaled"],
    params_config={
        "scale": {"min": 0.1, "max": 4.0, "step": 0.1}
    }
)
def rescale_image(image, scale: float = 0.5):
    """
    Resizes the image by a scale factor.
    Note: Safely handles RGB images by checking dimensions.
    """
    # If image is RGB (3D with depth 3), don't scale the color channel
    c_axis = None
    if image.ndim == 3 and image.shape[-1] in [3, 4]:
        c_axis = 2
        
    return skimage.transform.rescale(image, scale, anti_aliasing=True, channel_axis=c_axis)

# --- DOWNSCALE LOCAL MEAN ---
@register_node(
    label="Downscale (Binning)",
    category="Transform",
    outputs=["downscaled"],
    params_config={
        "factor": {"min": 1, "max": 10, "step": 1}
    }
)
def downscale_local_mean(image, factor: int = 2):
    """
    Downsamples the image by local averaging (integer binning).
    Good for reducing noise while shrinking.
    """
    # Create a tuple factors: (2, 2) or (2, 2, 1) for RGB
    factors = (factor, factor)
    if image.ndim == 3:
        factors = (factor, factor, 1) # Don't downscale color channel
        
    return skimage.transform.downscale_local_mean(image, factors)

# ==========================================
#           DISTORTIONS & WARPS
# ==========================================

# --- SWIRL ---
@register_node(
    label="Swirl",
    category="Transform",
    outputs=["swirled"],
    params_config={
        "strength": {"min": 0.0, "max": 10.0, "step": 0.1},
        "radius": {"min": 10, "max": 1000, "step": 10},
        "rotation": {"min": 0.0, "max": 6.28} # 2*pi
    }
)
def swirl(image, strength: float = 1.0, radius: float = 100.0, rotation: float = 0.0):
    """
    Performs a non-linear whirlpool distortion.
    """
    return skimage.transform.swirl(image, strength=strength, radius=radius, rotation=rotation)

# --- WARP POLAR ---
@register_node(
    label="Warp Polar (Ring to Line)",
    category="Transform",
    outputs=["polar_image"],
    params_config={
        "radius": {"min": 10, "max": 2000, "step": 10},
        "scaling": {"options": ["linear", "log"]}
    }
)
def warp_polar(image, radius: float = 100.0, scaling: str = 'linear'):
    """
    Remaps image to polar coordinates (Theta x Radius).
    Useful for analyzing circular objects (like clocks, irises, or radial patterns).
    """
    # Handle RGB safely (warp_polar expects 2D or uses channel axis implicit logic differently in older versions)
    # Basic implementation for 2D images:
    if image.ndim == 3 and image.shape[-1] in [3, 4]:
        c_axis = 2
    else:
        c_axis = None
        
    return skimage.transform.warp_polar(image, radius=radius, scaling=scaling, channel_axis=c_axis)


