from .decorator import register_node
import skimage.morphology

# --- DILATION ---
@register_node(
    label="Dilation",
    category="Morphology",
    outputs=["image_out"],
    params_config={
        "radius": {"min": 1, "max": 50}
    }
)
def dilation(image, radius: int = 1):
    """Wraps skimage.morphology.dilation with a disk footprint"""
    footprint = skimage.morphology.disk(radius)
    return skimage.morphology.dilation(image, footprint=footprint)