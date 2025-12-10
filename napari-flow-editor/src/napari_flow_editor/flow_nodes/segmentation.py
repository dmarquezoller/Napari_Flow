from .decorator import register_node
import skimage.segmentation

# --- ALREADY IMPLEMENTED --- #
# - clear border              #     
# - expand labels             #     
# - find boundaries           #     
# - mark boundaries (overlay) #     
# - slic (superpixels)        #     
# - felzenszwalb              #     
# - quickshift                #     
# - watershed                 #     
# - random walker             #     
# - chan vese (active contour)#
# --- --- --- --- --- --- --- #


# ==========================================
#           BOUNDARY & UTILITIES
# ==========================================

# --- CLEAR BORDER ---
@register_node(
    label="Clear Border",
    category="Segmentation",
    outputs=["cleaned_mask"]
)
def clear_border(image):
    """
    Clear objects connected to the border of the image.
    """
    return skimage.segmentation.clear_border(image)

# --- EXPAND LABELS ---
@register_node(
    label="Expand Labels",
    category="Segmentation",
    outputs=["expanded_labels"],
    params_config={"distance": {"min": 1, "max": 100}}
)
def expand_labels(image, distance: int = 10):
    """
    Expand labels in label image by distance pixels without overlapping.
    """
    return skimage.segmentation.expand_labels(image, distance=distance)

# --- FIND BOUNDARIES ---
@register_node(
    label="Find Boundaries",
    category="Segmentation",
    outputs=["boundaries"],
    params_config={
        "mode": {"options": ["thick", "inner", "outer", "subpixel"]}
    }
)
def find_boundaries(image, mode: str = "thick"):
    """
    Return bool mask where boundaries between labeled regions are True.
    """
    return skimage.segmentation.find_boundaries(image, mode=mode)

# --- MARK BOUNDARIES (OVERLAY) ---
@register_node(
    label="Mark Boundaries (Overlay)",
    category="Segmentation",
    outputs=["overlay_image"],
    params_config={
        "color": {"options": ["red", "blue", "yellow"]}, # Simplified options
        "mode": {"options": ["thick", "inner", "outer", "subpixel"]}
    }
)
def mark_boundaries(image, label_img, mode: str = "outer", color: str = "red"):
    """
    Returns the image with boundaries of the label_img superimposed.
    NOTE: Requires 2 Inputs (Original Image + Label Mask).
    """
    # Map string colors to RGB tuples for skimage
    c_map = {"red": (1, 0, 0), "blue": (0, 0, 1), "yellow": (1, 1, 0)}
    return skimage.segmentation.mark_boundaries(image, label_img, color=c_map.get(color), mode=mode)

# ==========================================
#           SUPERPIXELS / CLUSTERING
# ==========================================

# --- SLIC (Superpixels) ---
@register_node(
    label="SLIC (Superpixels)",
    category="Segmentation",
    outputs=["labels"],
    params_config={
        "n_segments": {"min": 10, "max": 5000, "step": 10},
        "compactness": {"min": 0.1, "max": 100.0, "step": 0.1},
        "sigma": {"min": 0.0, "max": 5.0}
    }
)
def slic(image, n_segments: int = 100, compactness: float = 10.0, sigma: float = 1.0):
    """
    Segments image using k-means clustering in Color-(x,y,z) space.
    """
    return skimage.segmentation.slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=1)

# --- FELZENSWALB ---
@register_node(
    label="Felzenszwalb",
    category="Segmentation",
    outputs=["labels"],
    params_config={
        "scale": {"min": 1, "max": 1000},
        "sigma": {"min": 0.0, "max": 10.0},
        "min_size": {"min": 10, "max": 1000}
    }
)
def felzenszwalb(image, scale: int = 100, sigma: float = 0.5, min_size: int = 20):
    """
    Efficient graph-based image segmentation.
    """
    return skimage.segmentation.felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)

# --- QUICKSHIFT ---
@register_node(
    label="Quickshift",
    category="Segmentation",
    outputs=["labels"],
    params_config={
        "ratio": {"min": 0.0, "max": 1.0, "step": 0.05},
        "kernel_size": {"min": 1, "max": 20},
        "max_dist": {"min": 1, "max": 100}
    }
)
def quickshift(image, ratio: float = 0.5, kernel_size: int = 5, max_dist: int = 10):
    """
    Segments image using quickshift mode seeking.
    """
    return skimage.segmentation.quickshift(image, ratio=ratio, kernel_size=kernel_size, max_dist=max_dist)

# ==========================================
#           ADVANCED SEGMENTATION
# ==========================================

# --- WATERSHED ---
@register_node(
    label="Watershed",
    category="Segmentation",
    outputs=["labels"],
    params_config={
        "compactness": {"min": 0.0, "max": 5.0, "step": 0.01}
    }
)
def watershed(image, markers, compactness: float = 0.0):
    """
    Find watershed basins in `image` flooded from `markers`.
    
    Inputs:
    1. image: The "basin" image (usually gradient or inverted distance).
    2. markers: An image with seeds (integers > 0) indicating where to start.
    """
    return skimage.segmentation.watershed(image, markers, compactness=compactness)

# --- RANDOM WALKER ---
@register_node(
    label="Random Walker",
    category="Segmentation",
    outputs=["labels"],
    params_config={
        "beta": {"min": 10, "max": 5000, "step": 10},
        "mode": {"options": ["bf", "cg_mg", "cg"]}
    }
)
def random_walker(image, markers, beta: int = 130, mode: str = "bf"):
    """
    Random walker algorithm.
    Determines segmentation of pixels based on probability of reaching markers.
    
    Inputs:
    1. image: Grayscale image.
    2. markers: Seed map.
    """
    # random_walker expects markers > 0.
    return skimage.segmentation.random_walker(image, markers, beta=beta, mode=mode)

# --- CHAN VESSE (ACTIVE CONTOUR) ---
@register_node(
    label="Chan Vese (Active Contour)",
    category="Segmentation",
    outputs=["mask_out"],
    params_config={
        "mu": {"min": 0.0, "max": 1.0, "step": 0.05},
        "lambda1": {"min": 0.1, "max": 5.0, "step": 0.1},
        "lambda2": {"min": 0.1, "max": 5.0, "step": 0.1},
        "max_num_iter": {"min": 10, "max": 500, "step": 10}
    }
)
def chan_vese(image, mu: float = 0.25, lambda1: float = 1.0, lambda2: float = 1.0, max_num_iter: int = 100):
    """
    Active contours without edges. Can segment objects without clear boundaries.
    Returns a binary mask.
    """
    # Typically returns (mask, evolution_history, energy)
    # We return just the mask [0]
    result = skimage.segmentation.chan_vese(
        image, 
        mu=mu, 
        lambda1=lambda1, 
        lambda2=lambda2, 
        max_num_iter=max_num_iter
    )
    return result[0] # Return the mask only