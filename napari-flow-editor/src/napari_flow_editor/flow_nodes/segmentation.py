from .decorator import register_node
import skimage.measure
import skimage.segmentation 
import skimage.filters  

# --- OTSU THRESHOLD ---
@register_node(
    label="Otsu Threshold",
    category="Segmentation",
    outputs=["mask_out"]
)
def threshold_otsu(image):
    return image > skimage.filters.threshold_otsu(image)

# --- LOCAL (ADAPTIVE) THRESHOLD ---
@register_node(
    label="Threshold Local (Adaptive)",
    category="Segmentation",
    outputs=["mask_out"],
    params_config={
        "block_size": {"min": 3, "max": 99, "step": 2}, # Must be odd
        "offset": {"min": -0.5, "max": 0.5, "step": 0.01}
    }
)
def threshold_local(image, block_size: int = 35, offset: float = 0.0):
    # Block size must be odd
    if block_size % 2 == 0: block_size += 1
    
    local_thresh = skimage.filters.threshold_local(image, block_size, offset=offset)
    return image > local_thresh

# --- CLEAR BORDER ---
@register_node(
    label="Clear Border",
    category="Segmentation",
    outputs=["cleaned_mask"]
)
def clear_border(image):
    return skimage.segmentation.clear_border(image)

# --- LABEL OBJECTS ---
@register_node(
    label="Label Objects",
    category="Segmentation",
    outputs=["labels"]
)
def label_objects(image):
    # Turns binary mask into integer labels (1, 2, 3...)
    # This is great for viewing individual cells in Napari with a 'Labels' layer
    return skimage.measure.label(image > 0)