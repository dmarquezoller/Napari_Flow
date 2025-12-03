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

# --- CLOSING ---
@register_node(
    label="Binary Closing (Fill Holes)",
    category="Morphology",
    outputs=["mask_out"],
    params_config={"radius": {"min": 1, "max": 20}}
)
def binary_closing(image, radius: int = 3):
    # Closing = Dilation followed by Erosion
    footprint = skimage.morphology.disk(radius)
    return skimage.morphology.binary_closing(image, footprint=footprint)

# --- OPENING ---
@register_node(
    label="Binary Opening (Remove Noise)",
    category="Morphology",
    outputs=["mask_out"],
    params_config={"radius": {"min": 1, "max": 20}}
)
def binary_opening(image, radius: int = 3):
    # Opening = Erosion followed by Dilation
    footprint = skimage.morphology.disk(radius)
    return skimage.morphology.binary_opening(image, footprint=footprint)

# --- SKELETONIZE ---
@register_node(
    label="Skeletonize",
    category="Morphology",
    outputs=["skeleton"]
)
def skeletonize(image):
    # Expects binary image
    return skimage.morphology.skeletonize(image > 0)

# --- REMOVE SMALL OBJECTS ---
@register_node(
    label="Remove Small Objects",
    category="Morphology",
    outputs=["cleaned_mask"],
    params_config={
        "min_size": {"min": 10, "max": 1000, "step": 10}
    }
)
def remove_small_objects(image, min_size: int = 64):
    # Convert to boolean, remove objects, return
    bool_img = image > 0
    return skimage.morphology.remove_small_objects(bool_img, min_size=min_size)