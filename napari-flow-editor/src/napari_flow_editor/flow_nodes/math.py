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
    

# --- INTERACTIVE CROP/SLICE --- #
@register_node(
    label="Interactive Crop",
    category="Math",
    outputs=["cropped_image"],
    params_config={
        "t_crop": {
            "type": "text", 
            "label": "Time/Z (Dim 0) [e.g. 0:10 or :]", 
            "value": ":"
        }
    }
)
def crop_image_interactive(image, roi_layer, t_crop=":"):
    image = _ensure_numpy(image)
    if image is None: return None

    # 1. Parse Time Slice
    sl_t = _parse_slice(t_crop)

    # 2. Parse ROI (Y/X)
    shapes_data = []
    if hasattr(roi_layer, "data"): shapes_data = roi_layer.data
    elif isinstance(roi_layer, list): shapes_data = roi_layer

    if not shapes_data or len(shapes_data) == 0:
        raise ValueError("Crop failed: No shapes found. Please draw a Rectangle.")

    last_shape = shapes_data[-1]
    min_coords = np.min(last_shape, axis=0)
    max_coords = np.max(last_shape, axis=0)
    
    # Napari shapes are always (..., Y, X)
    y_min, x_min = int(min_coords[-2]), int(min_coords[-1])
    y_max, x_max = int(max_coords[-2]), int(max_coords[-1])

    # Clamp to image spatial limits (Y, X are usually the 2nd and 3rd to last dims)
    # We'll validate strictly inside the logic blocks below
    y_min = max(0, y_min); x_min = max(0, x_min)
    if y_max <= y_min: y_max = y_min + 1
    if x_max <= x_min: x_max = x_min + 1
    
    sl_y = slice(y_min, y_max)
    sl_x = slice(x_min, x_max)

    print(f"--- Cropping Input: {image.shape} ---")
    print(f"  > ROI: Y[{y_min}:{y_max}], X[{x_min}:{x_max}]")

    # 3. Smart Slicing Logic
    try:
        out = None
        
        # Case 4D
        if image.ndim == 4:
            # Check if last dim is small (Channels) vs large (Spatial X)
            if image.shape[-1] < 10: 
                # (Time, Y, X, Channel) -> THIS IS YOUR CASE
                print("  > Detecting (Time, Y, X, C) structure")
                # Clamp X/Y against the correct dimensions
                y_max = min(image.shape[1], y_max)
                x_max = min(image.shape[2], x_max)
                sl_y = slice(y_min, y_max)
                sl_x = slice(x_min, x_max)
                
                out = image[sl_t, sl_y, sl_x, :]
                
            else:
                # (Time, Z, Y, X)
                print("  > Detecting (Time, Z, Y, X) structure")
                y_max = min(image.shape[2], y_max)
                x_max = min(image.shape[3], x_max)
                sl_y = slice(y_min, y_max)
                sl_x = slice(x_min, x_max)
                
                out = image[sl_t, :, sl_y, sl_x]

        # Case 3D: (Time, Y, X) or (Z, Y, X)
        elif image.ndim == 3:
            y_max = min(image.shape[1], y_max)
            x_max = min(image.shape[2], x_max)
            sl_y = slice(y_min, y_max)
            sl_x = slice(x_min, x_max)
            
            out = image[sl_t, sl_y, sl_x]

        # Case 2D: (Y, X)
        elif image.ndim == 2:
            y_max = min(image.shape[0], y_max)
            x_max = min(image.shape[1], x_max)
            sl_y = slice(y_min, y_max)
            sl_x = slice(x_min, x_max)
            
            out = image[sl_y, sl_x]

        else:
            out = image

        # Safety Check
        if out is None or out.size == 0:
            raise ValueError(f"Resulting crop is empty! Shape: {out.shape}. Check your Time/Z slice and ROI.")

        print(f"  > Success. New Shape: {out.shape}")
        return (out, {"name": "Cropped Output"}, "image")

    except Exception as e:
        print(f"Crop Failed: {e}")
        raise e