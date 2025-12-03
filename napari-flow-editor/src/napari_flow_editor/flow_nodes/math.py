from .decorator import register_node
import numpy as np
import skimage.util
import skimage.exposure

# --- BLEND IMAGES ---
@register_node(
    label="Blend Images",
    category="Math",
    # We only have one output, the result
    outputs=["blended_image"],
    params_config={
        "alpha": {"min": 0.0, "max": 1.0, "step": 0.1}
    }
)
def blend_images(image_a, image_b, alpha: float = 0.5):
    """
    Because image_a and image_b have no defaults, 
    the Generator automatically creates 2 Input Sockets for them.
    """
    # Simple blend logic
    return (image_a * alpha) + (image_b * (1 - alpha))

# --- INVERT IMAGE ---
@register_node(
    label="Invert Image",
    category="Math",
    outputs=["inverted"]
)
def invert_image(image):
    return skimage.util.invert(image)

# --- GAMMA CORRECTION ---
@register_node(
    label="Gamma Correction",
    category="Math",
    outputs=["corrected"],
    params_config={
        "gamma": {"min": 0.1, "max": 3.0, "step": 0.1}
    }
)
def gamma_correction(image, gamma: float = 1.0):
    return skimage.exposure.adjust_gamma(image, gamma)