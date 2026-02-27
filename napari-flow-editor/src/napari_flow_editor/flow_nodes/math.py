from .decorator import register_node
from .deep_learning import _ensure_numpy
import numpy as np
import napari

# --- ALREADY IMPLEMENTED --- #
# - blend images              #          
# --- --- --- --- --- --- --- #

# --- BLEND IMAGES ---
@register_node(
    label="Blend Images",
    category="Math",
    # We only have one output, the result
    outputs=["blended_image"],
    input_types={"image_a": "image", "image_b": "image"},
    output_types={"blended_image": "image"},
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

# =============================================================================
# INTERACTIVE CROP  (new declarative approach)
#
# ``interactive={...}`` tells the engine to:
#   1. Create a temporary Shapes layer in napari.
#   2. Show a dialog with a **Run** button.
#   3. Wait for the user to draw a rectangle and click Run.
#   4. Inject the drawn shape data as ``interaction=<list of arrays>``.
# =============================================================================
@register_node(
    label="Interactive Crop",
    category="Math",
    outputs=["cropped_image"],
    input_types={"image_input": "image"},
    output_types={"cropped_image": "image"},
    params_config={
        "t_crop": {"type": "text", "label": "Time/Z Slice", "value": ":"}
    },
    interactive={
        "layer_type": "shapes",
        "mode": "add_rectangle",
        "edge_color": "#00ff00",
        "face_color": [0, 0, 0, 0],
        "edge_width": 3,
        "prompt": "Draw a rectangle on the image, then click Run.",
    },
)
def interactive_crop(image_input, t_crop=":", interaction=None):
    # --- 1. UNPACK ---
    image = image_input
    meta = {}
    if isinstance(image_input, tuple):
        if len(image_input) >= 2:
            image = image_input[0]
            if isinstance(image_input[1], dict): meta = image_input[1]
    
    image = _ensure_numpy(image)
    if image is None: return None

    # --- 2. INTERACTION (Wait for Draw) ---
    shapes_data = interaction

    # --- 3. YOUR TEMPLATE LOGIC STARTS HERE ---
    
    # A. Parse Time Slice
    sl_t = _parse_slice(t_crop)

    # B. Parse ROI (Y/X) - Adapted from your template
    last_shape = shapes_data[-1]
    min_coords = np.min(last_shape, axis=0)
    max_coords = np.max(last_shape, axis=0)
    
    # Napari shapes are always (..., Y, X)
    y_min, x_min = int(min_coords[-2]), int(min_coords[-1])
    y_max, x_max = int(max_coords[-2]), int(max_coords[-1])

    def _clamp_roi(y0, y1, x0, x1, h, w):
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image spatial shape: ({h}, {w})")
        y0 = int(np.clip(y0, 0, h - 1))
        x0 = int(np.clip(x0, 0, w - 1))
        y1 = int(np.clip(y1, y0 + 1, h))
        x1 = int(np.clip(x1, x0 + 1, w))
        return y0, y1, x0, x1

    # C. Smart Slicing Logic (Exact copy of your template)
    try:
        out = None
        is_rgb = False # Flag we will detect
        sl_y = None
        sl_x = None

        # Case 4D
        if image.ndim == 4:
            # Check if last dim is small (Channels) vs large (Spatial X)
            if image.shape[-1] < 10: 
                # (Time, Y, X, Channel) -> THIS IS YOUR CASE
                print("  > Detecting (Time, Y, X, C) structure")
                is_rgb = True # <--- We mark this!

                y0, y1, x0, x1 = _clamp_roi(
                    y_min, y_max, x_min, x_max, image.shape[1], image.shape[2]
                )
                sl_y = slice(y0, y1)
                sl_x = slice(x0, x1)
                
                out = image[sl_t, sl_y, sl_x, :]
                
            else:
                # (Time, Z, Y, X)
                print("  > Detecting (Time, Z, Y, X) structure")
                y0, y1, x0, x1 = _clamp_roi(
                    y_min, y_max, x_min, x_max, image.shape[2], image.shape[3]
                )
                sl_y = slice(y0, y1)
                sl_x = slice(x0, x1)
                
                out = image[sl_t, :, sl_y, sl_x]

        # Case 3D: (Time, Y, X) or (Z, Y, X)
        elif image.ndim == 3:
            # Added small heuristic for single RGB image (Y, X, C)
            if image.shape[-1] < 10:
                print("  > Detecting (Y, X, C) structure")
                is_rgb = True
                y0, y1, x0, x1 = _clamp_roi(
                    y_min, y_max, x_min, x_max, image.shape[0], image.shape[1]
                )
                sl_y = slice(y0, y1)
                sl_x = slice(x0, x1)
                out = image[sl_y, sl_x, :]
            else:
                y0, y1, x0, x1 = _clamp_roi(
                    y_min, y_max, x_min, x_max, image.shape[1], image.shape[2]
                )
                sl_y = slice(y0, y1)
                sl_x = slice(x0, x1)
                out = image[sl_t, sl_y, sl_x]

        # Case 2D: (Y, X)
        elif image.ndim == 2:
            y0, y1, x0, x1 = _clamp_roi(
                y_min, y_max, x_min, x_max, image.shape[0], image.shape[1]
            )
            sl_y = slice(y0, y1)
            sl_x = slice(x0, x1)
            out = image[sl_y, sl_x]

        else:
            out = image

        print(f"--- Cropping Input: {image.shape} ---")
        if sl_y is not None and sl_x is not None:
            print(f"  > ROI: Y[{sl_y.start}:{sl_y.stop}], X[{sl_x.start}:{sl_x.stop}]")
        else:
            print("  > ROI: Full image")

        # Safety Check
        if out is None or out.size == 0:
            raise ValueError(f"Resulting crop is empty! Shape: {out.shape}.")

        print(f"  > Success. New Shape: {out.shape}")
        
        # --- 4. PACKING THE RESULT ---
        new_meta = meta.copy()
        new_meta["name"] = f"Crop {t_crop}"
        
        # FIX THE SLIDERS: If we detected (..., C), tell Napari it is RGB
        if is_rgb:
            new_meta["rgb"] = True
            
        return (out, new_meta)
    except Exception as e:
        print(f"Crop Failed: {e}")
        raise e


# HELPERS  

def _parse_slice(s):
    try:
        if ":" in s:
            p = s.split(":")
            return slice(int(p[0]) if p[0] else None, int(p[1]) if p[1] else None)
        return int(s)
    except: return slice(None)




    # --- INTERACTIVE CROP/SLICE --- #
@register_node(
    label="Interactive Crop (Legacy)",
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
    





# PROJECT #


@register_node(
    label="Project 3D to 2D",
    category="Math",
    outputs=["projected_img"],
    params_config={
        "axis": {
            "type": "enum", 
            "options": ["0", "1", "2"], 
            "label": "Projection Axis (0=Z, 1=Y, 2=X)"
        },
        "mode": {
            "type": "enum", 
            "options": ["Max", "Mean", "Sum", "Std"], 
            "label": "Method"
        }
    }
)
def project_3d_to_2d(image_in, axis=0, mode="Max"):
    """
    Flattens a 3D volume into a 2D image layer.
    """
    if image_in is None:
        return None
    
    axis = int(axis)

    # Safety: Check dimensions
    if axis >= image_in.ndim:
        print(f"⚠️ Axis {axis} is out of bounds for {image_in.ndim}D image.")
        return None

    # Perform Projection
    if mode == 'Max':
        res = np.max(image_in, axis=axis)
    elif mode == 'Mean':
        res = np.mean(image_in, axis=axis)
    elif mode == 'Sum':
        res = np.sum(image_in, axis=axis)
    elif mode == 'Std':
        res = np.std(image_in, axis=axis)
    else:
        res = np.max(image_in, axis=axis)
        
    return res
