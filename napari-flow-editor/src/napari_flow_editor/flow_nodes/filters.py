from .decorator import register_node
import skimage.filters

# --- GAUSSIAN BLUR ---
@register_node(
    label="Gaussian Blur",
    category="Filters",
    outputs=["image_out"],
    params_config={
        "sigma": {"min": 0.0, "max": 20.0, "step": 0.1},
        "mode": {"options": ["nearest", "reflect", "wrap", "constant"]}
    }
)
def gaussian_blur(image, sigma: float = 1.0, mode: str = 'nearest'):
    """Wraps skimage.filters.gaussian"""
    return skimage.filters.gaussian(image, sigma=sigma, mode=mode)


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

