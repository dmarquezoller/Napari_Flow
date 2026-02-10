"""
Advanced Examples demonstrating the enhanced @register_node decorator API.

This module showcases:
- interactive: Interactive Napari layer creation and geometry capture
- output_meta: Declarative output metadata for layer styling
- validate_inputs: Input validation with dtype and ndim checks
- doc: Tooltip documentation
- icon: Visual node icons
"""

from .decorator import register_node
import numpy as np
import skimage.filters


# --- EXAMPLE 1: Interactive ROI Crop ---
@register_node(
    label="Interactive Crop",
    category="Examples",
    outputs=["cropped"],
    interactive={
        "layer_type": "shapes",
        "tool": "rectangle",
        "prompt": "Draw a rectangle to define the crop region",
        "arg_name": "roi_geometry",
        "confirm": True,
    },
    doc="Interactively crop an image by drawing a rectangle ROI",
    icon="✂️"
)
def interactive_crop(image_in, roi_geometry=None):
    """
    Crops an image based on user-drawn rectangle.
    If no geometry is provided, returns the original image.
    """
    if roi_geometry is None or len(roi_geometry) == 0:
        return image_in
    
    # Extract bounding box from first rectangle
    # Shapes layer returns rectangles as 4 corner points
    rect = roi_geometry[0]
    if len(rect) < 3:
        # Invalid rectangle, return original
        return image_in
    
    y_min, x_min = int(rect[0][0]), int(rect[0][1])
    y_max, x_max = int(rect[2][0]), int(rect[2][1])
    
    # Ensure bounds are valid
    y_min, y_max = min(y_min, y_max), max(y_min, y_max)
    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
    
    # Crop the image
    cropped = image_in[y_min:y_max, x_min:x_max]
    return cropped


# --- EXAMPLE 2: Threshold with Output Metadata ---
@register_node(
    label="Threshold with Style",
    category="Examples",
    outputs=["mask"],
    output_meta={
        "layer_type": "labels",
        "colormap": "cyan",
        "opacity": 0.5,
        "name_suffix": "Binary Mask",
    },
    validate_inputs={
        "image": {
            "required": True,
            "dtype": ["float32", "float64", "uint8", "uint16"],
            "ndim": [2, 3],
        }
    },
    doc="Applies Otsu thresholding with automatic styling for the mask layer",
    icon="🎭"
)
def threshold_with_style(image, invert: bool = False):
    """
    Creates a binary mask using Otsu's method.
    Output is automatically styled as a cyan semi-transparent label layer.
    """
    threshold = skimage.filters.threshold_otsu(image)
    mask = image > threshold
    
    if invert:
        mask = ~mask
    
    return mask.astype(np.uint8)


# --- EXAMPLE 3: Smart Gaussian Blur with Validation ---
@register_node(
    label="Smart Gaussian Blur",
    category="Examples",
    outputs=["blurred"],
    validate_inputs={
        "image": {
            "required": True,
            "dtype": ["float32", "float64", "uint8", "uint16"],
            "ndim": [2, 3],
        }
    },
    params_config={
        "sigma": {"min": 0.0, "max": 10.0, "step": 0.1},
    },
    doc="Gaussian blur with input validation. Ensures image is 2D or 3D with correct dtype.",
    icon="🔬"
)
def smart_gaussian_blur(image, sigma: float = 1.0):
    """
    Applies Gaussian blur with validated inputs.
    Automatically checks that image has correct dimensions and data type.
    """
    # Input validation is handled automatically by the engine
    return skimage.filters.gaussian(image, sigma=sigma, preserve_range=True)


# --- EXAMPLE 4: Multi-output with Different Metadata ---
@register_node(
    label="Edge Detection Suite",
    category="Examples",
    outputs=["sobel", "canny"],
    output_meta={
        "colormap": "viridis",
        "opacity": 0.7,
        "name_suffix": "Edges",
    },
    validate_inputs={
        "image": {
            "required": True,
            "ndim": [2],
        }
    },
    params_config={
        "sigma": {"min": 0.0, "max": 5.0, "step": 0.1},
    },
    doc="Computes multiple edge detection algorithms (Sobel and Canny) with styled outputs",
    icon="🌊"
)
def edge_detection_suite(image, sigma: float = 1.0):
    """
    Returns both Sobel and Canny edge detection results.
    Both outputs are styled with the same colormap.
    """
    sobel_edges = skimage.filters.sobel(image)
    canny_edges = skimage.filters.canny(image, sigma=sigma)
    
    return sobel_edges, canny_edges.astype(np.float32)


# --- EXAMPLE 5: Interactive Point Picker ---
@register_node(
    label="Interactive Point Analysis",
    category="Examples",
    outputs=["analysis"],
    interactive={
        "layer_type": "points",
        "tool": "add",
        "prompt": "Click to add points of interest on the image",
        "arg_name": "points",
        "confirm": True,
    },
    doc="Pick points interactively and analyze intensity values at those locations",
    icon="📍"
)
def interactive_point_analysis(image_in, points=None):
    """
    Analyzes intensity values at user-selected points.
    Returns a dictionary with point coordinates and their intensities.
    """
    if points is None or len(points) == 0:
        return {"message": "No points selected"}
    
    intensities = []
    for point in points:
        y, x = int(point[0]), int(point[1])
        if 0 <= y < image_in.shape[0] and 0 <= x < image_in.shape[1]:
            intensities.append(float(image_in[y, x]))
    
    return {
        "num_points": len(points),
        "mean_intensity": np.mean(intensities) if intensities else 0,
        "std_intensity": np.std(intensities) if intensities else 0,
        "points": points.tolist() if hasattr(points, 'tolist') else points,
        "intensities": intensities
    }
