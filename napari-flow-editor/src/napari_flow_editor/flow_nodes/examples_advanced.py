"""
Advanced examples demonstrating the enhanced @register_node decorator features.

This module showcases:
- interactive: Declarative Napari interactivity
- output_meta: Declarative output metadata
- validate_inputs: Declarative input validation
- doc: Tooltip/help text
- icon: Visual category hint
"""

try:
    from .decorator import register_node
except ImportError:
    # Fallback for direct module loading
    from napari_flow_editor.flow_nodes.decorator import register_node

import numpy as np

# Import scipy at module level to avoid repeated import overhead
try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Example 1: Node with doc and icon
@register_node(
    label="Example Threshold",
    category="Examples",
    outputs=["mask"],
    params_config={
        "threshold": {"min": 0.0, "max": 1.0, "step": 0.01}
    },
    doc="Applies a simple threshold to create a binary mask. Values above the threshold become 1, below become 0.",
    icon="🎯"
)
def example_threshold(image, threshold: float = 0.5):
    """Simple thresholding operation."""
    return (image > threshold).astype(np.uint8)


# Example 2: Node with output_meta and validate_inputs
@register_node(
    label="Example Blur with Validation",
    category="Examples",
    outputs=["blurred"],
    params_config={
        "sigma": {"min": 0.1, "max": 10.0, "step": 0.1}
    },
    output_meta={
        "layer_type": "image",
        "opacity": 0.8,
        "name_suffix": "Blurred"
    },
    validate_inputs={
        "image": {
            "required": True,
            "ndim": [2, 3]
        }
    },
    doc="Applies Gaussian blur with input validation. Requires a 2D or 3D image.",
    icon="🔬"
)
def example_blur_validated(image, sigma: float = 1.0):
    """Applies a simple Gaussian blur using convolution."""
    if not HAS_SCIPY:
        # Fallback: return original image if scipy not available
        return image.astype(float)
    return gaussian_filter(image.astype(float), sigma=sigma)


# Example 3: Node with all validation features
@register_node(
    label="Example Advanced Filter",
    category="Examples",
    outputs=["filtered"],
    params_config={
        "strength": {"min": 0.0, "max": 10.0, "step": 0.1}
    },
    output_meta={
        "layer_type": "labels",
        "colormap": "viridis",
        "opacity": 0.6
    },
    validate_inputs={
        "image": {
            "required": True,
            "dtype": ["float32", "float64", "uint8", "uint16"],
            "ndim": [2, 3]
        }
    },
    doc="Advanced filter with comprehensive validation. Requires specific dtypes and dimensions.",
    icon="⚡"
)
def example_advanced_filter(image, strength: float = 1.0):
    """
    Advanced filtering with validation.
    
    This demonstrates comprehensive input validation including:
    - Required input check
    - Data type validation (float32, float64, uint8, uint16)
    - Dimension validation (2D or 3D only)
    """
    # Simple processing: multiply by strength and convert to integer
    result = (image.astype(float) * strength).astype(np.int32)
    return result


# Example 4: Interactive node placeholder
# Note: Full interactive functionality requires UI thread coordination
# This is a placeholder showing the decorator syntax
@register_node(
    label="Example Interactive ROI",
    category="Examples",
    outputs=["cropped"],
    interactive={
        "layer_type": "shapes",
        "tool": "rectangle",
        "prompt": "Draw a rectangle to define the crop region",
        "arg_name": "roi_geometry",
        "confirm": True
    },
    doc="Interactive crop node (placeholder). Would allow user to draw a rectangle for cropping.",
    icon="✂️"
)
def example_interactive_crop(image, roi_geometry=None):
    """
    Interactive crop node placeholder.
    
    In a full implementation, the execution engine would:
    1. Create a temporary Shapes layer
    2. Activate rectangle drawing tool
    3. Wait for user confirmation
    4. Pass the geometry to this function as roi_geometry
    5. Clean up the temporary layer
    
    For now, this just returns the original image.
    """
    if roi_geometry is None:
        # No ROI drawn - return original
        return image
    
    # In a real implementation, roi_geometry would contain bounding box coordinates
    # and we would crop the image accordingly
    # For now, just return a smaller central region as a placeholder
    if image.ndim == 2:
        h, w = image.shape
        return image[h//4:3*h//4, w//4:3*w//4]
    elif image.ndim == 3:
        d, h, w = image.shape
        return image[d//4:3*d//4, h//4:3*h//4, w//4:3*w//4]
    else:
        return image


# Example 5: Node with multiple outputs and metadata
@register_node(
    label="Example Split Channels",
    category="Examples",
    outputs=["channel_0", "channel_1", "channel_2"],
    validate_inputs={
        "image": {
            "required": True,
            "ndim": [3]
        }
    },
    doc="Splits a 3-channel image into separate channels. Input must be 3D (H, W, C) with 3 channels. Note: declarative validation only checks ndim, not channel count.",
    icon="🎨"
)
def example_split_channels(image):
    """
    Splits a 3-channel image into separate channels.
    
    Note: The decorator's validate_inputs only checks ndim=[3], not the actual
    number of channels. This function includes runtime validation for the
    specific shape requirement.
    
    Returns:
        tuple: (channel_0, channel_1, channel_2)
    """
    # Runtime validation for channel count (beyond what declarative validation provides)
    if image.ndim != 3 or image.shape[-1] != 3:
        # Return empty channels if not a 3-channel image
        dummy = np.zeros_like(image[..., 0] if image.ndim == 3 else image)
        return dummy, dummy, dummy
    
    return image[..., 0], image[..., 1], image[..., 2]
