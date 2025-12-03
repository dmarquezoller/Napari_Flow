from .decorator import register_node
import skimage.filters

# --- OTSU THRESHOLD ---
@register_node(
    label="Otsu Threshold",
    category="Segmentation",
    outputs=["mask_out"]
)
def threshold_otsu(image):
    return image > skimage.filters.threshold_otsu(image)