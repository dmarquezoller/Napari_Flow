from .decorator import register_node
from .deep_learning import _ensure_numpy
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


# --- CROP/SLICE IMAGE --- #


def _parse_slice(slice_str):
    """Converts string '0:10' or '10:' to a python slice object."""
    if not slice_str or slice_str.strip() == ":" or slice_str.strip() == "":
        return slice(None)
    try:
        # Handles 'start:stop' and 'start:stop:step'
        parts = [int(p) if p.strip() else None for p in slice_str.split(':')]
        return slice(*parts)
    except ValueError:
        print(f"Warning: Invalid slice string '{slice_str}'. Using full range.")
        return slice(None)

@register_node(
    label="Crop / Slice Image",
    category="Math",
    outputs=["cropped_image"],
    params_config={
        "t_crop": {
            "type": "text", 
            "label": "Time/Z (Dim 0) [e.g. 0:10]", 
            "value": "0:10"
        },
        "y_crop": {
            "type": "text", 
            "label": "Y (Dim 1) [e.g. 100:500]", 
            "value": ":"
        },
        "x_crop": {
            "type": "text", 
            "label": "X (Dim 2) [e.g. 100:500]", 
            "value": ":"
        }
    }
)
def crop_image(image, t_crop=":", y_crop=":", x_crop=":"):
    """
    Crops an n-dimensional image using string slices.
    """
    image = _ensure_numpy(image)
    if image is None:
        return None

    # Parse inputs
    sl_t = _parse_slice(t_crop)
    sl_y = _parse_slice(y_crop)
    sl_x = _parse_slice(x_crop)
    
    print(f"--- Cropping Image (Original: {image.shape}) ---")
    
    # Apply logic based on dimensions
    # Assuming shape is (Time, Y, X, C) or (Time, Y, X)
    try:
        # Case 4D: (Time, Y, X, Channel) -> Your case (81, 894, 894, 3)
        if image.ndim == 4:
            out = image[sl_t, sl_y, sl_x, :]
            
        # Case 3D: (Time, Y, X) or (Z, Y, X)
        elif image.ndim == 3:
            out = image[sl_t, sl_y, sl_x]
            
        # Case 2D: (Y, X) - Ignore T input
        elif image.ndim == 2:
            print("  > 2D Image detected, ignoring Time crop.")
            out = image[sl_y, sl_x]
            
        else:
            print(f"  > Warning: Unsupported dimensions {image.ndim}. returning original.")
            out = image

        print(f"  > New Shape: {out.shape}")
        return out

    except Exception as e:
        print(f"Crop Failed: {e}")
        raise e