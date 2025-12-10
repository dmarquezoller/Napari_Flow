from .decorator import register_node
import skimage.morphology

# --- ALREADY IMPLEMENTED --- #
# - dilation                  #
# - closing                   #
# - opening                   #
# - skeletonize               #
# - remove small objects      #
# - remove small holes        #
# - white tophat              #
# - black tophat              #
# - erosion                   #
# - convex hull               #
# --- --- --- --- --- --- --- #

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
    # Check dimensions
    if image.ndim == 3:
        # Use a 3D Sphere for 3D data
        footprint = skimage.morphology.ball(radius)
    else:
        # Use a 2D Disk for 2D data
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

# --- EROSION (Shrink) ---
@register_node(
    label="Erosion (Shrink)",
    category="Morphology",
    outputs=["image_out"],
    params_config={"radius": {"min": 1, "max": 50}}
)
def erosion(image, radius: int = 1):
    """Wraps skimage.morphology.erosion with a disk footprint"""
    footprint = skimage.morphology.disk(radius)
    return skimage.morphology.erosion(image, footprint=footprint)

# --- WHITE TOPHAT ---
@register_node(
    label="White Tophat (Bright Spots)",
    category="Morphology",
    outputs=["tophat"],
    params_config={"radius": {"min": 1, "max": 50}}
)
def white_tophat(image, radius: int = 15):
    """
    Returns bright spots smaller than the structuring element.
    Great for background subtraction (uniform background).
    """
    footprint = skimage.morphology.disk(radius)
    return skimage.morphology.white_tophat(image, footprint=footprint)

# --- BLACK TOPHAT ---
@register_node(
    label="Black Tophat (Dark Spots)",
    category="Morphology",
    outputs=["tophat"],
    params_config={"radius": {"min": 1, "max": 50}}
)
def black_tophat(image, radius: int = 15):
    """
    Returns dark spots smaller than the structuring element.
    """
    footprint = skimage.morphology.disk(radius)
    return skimage.morphology.black_tophat(image, footprint=footprint)

# --- CONVEX HULL ---
@register_node(
    label="Convex Hull",
    category="Morphology",
    outputs=["hull"]
)
def convex_hull(image):
    """Computes the convex hull of the binary image objects."""
    # Note: This computes the hull of the *entire* True area as one object
    return skimage.morphology.convex_hull_image(image > 0)

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
    bool_img = image > 0
    return skimage.morphology.remove_small_objects(bool_img, min_size=min_size)

# --- REMOVE SMALL HOLES ---
@register_node(
    label="Remove Small Holes",
    category="Morphology",
    outputs=["filled_mask"],
    params_config={
        "area_threshold": {"min": 10, "max": 1000, "step": 10}
    }
)
def remove_small_holes(image, area_threshold: int = 64):
    """Fills holes smaller than the threshold inside objects."""
    bool_img = image > 0
    return skimage.morphology.remove_small_holes(bool_img, area_threshold=area_threshold)

# --- LABEL OBJECTS ---
@register_node(
    label="Label Objects",
    category="Morphology",
    outputs=["labels"]
)
def label_objects(image):
    """Labels connected components in a binary image."""
    return skimage.morphology.label(image > 0)


