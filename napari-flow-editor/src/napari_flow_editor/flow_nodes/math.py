from .decorator import register_node
import numpy as np

# --- ALREADY IMPLEMENTED --- #
# - blend images              #          
# --- --- --- --- --- --- --- #

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



