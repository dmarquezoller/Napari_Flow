import numpy as np
import traceback
import logging
import sys
import torch  # Critical for thread control

from .decorator import register_node

# --- SAFE IMPORTS ---
try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

@register_node(
    label="Cellpose Segmentation",
    category="Deep Learning",
    outputs=["mask_layer"],
    params_config={
        "model_type": {
            "options": ["nuclei", "cyto2", "cyto3", "tissuenet", "livecell"], 
            "value": "nuclei"
        },
        "diameter": {"min": 0.0, "max": 300.0, "step": 1.0, "value": 30.0},
        "flow_threshold": {"min": 0.0, "max": 1.0, "step": 0.1, "value": 0.4},
        "process_3d": {"type": "bool", "value": False, "label": "Process as 3D Vol"},
        "use_gpu": {"type": "bool", "value": False, "label": "Use GPU"} 
    }
)
def run_cellpose(image, 
                 model_type: str = 'nuclei', 
                 diameter: float = 30.0, 
                 flow_threshold: float = 0.4, 
                 process_3d: bool = False,
                 use_gpu: bool = False):
    
    if not CELLPOSE_AVAILABLE:
        raise ImportError("Cellpose not installed. Please run: pip install cellpose")

    # --- 1. LOGGING SETUP ---
    # Force Cellpose to print progress to stdout so the GUI captures it
    cp_logger = logging.getLogger("cellpose")
    cp_logger.setLevel(logging.INFO)
    if not cp_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        cp_logger.addHandler(handler)

    print(f"--- Cellpose Execution: {model_type} (GPU={use_gpu}) ---")

    # --- 2. INPUT UNWRAPPING ---
    # Extract the raw array from Napari layers (which might be Tuples or Lists)
    raw_img = image
    if isinstance(image, list) and len(image) > 0:
        if isinstance(image[0], tuple): # (data, meta)
            raw_img = image[0][0]
        else:
            raw_img = image[0]
    elif isinstance(image, tuple) and len(image) >= 2:
        raw_img = image[0]
    
    image = raw_img

    # --- 3. DASK / LAZY LOADING ---
    if hasattr(image, "compute"):
        print("  > Downloading data from lazy/Dask array...")
        image = image.compute()
    
    image = np.asarray(image)

    # --- 4. BOOLEAN FIX (CRITICAL FOR BLOBS) ---
    # Cellpose/OpenCV crashes on bool. Convert True/False -> 255/0
    if image.dtype == bool:
        print("  > Converting Boolean image to uint8...")
        image = image.astype(np.uint8) * 255

    # --- 5. CPU FREEZE PROTECTION ---
    # Restrict PyTorch to 1 thread to avoid deadlocking the GUI
    torch.set_num_threads(1)

    # --- 6. MODEL INITIALIZATION ---
    # Note: 'pretrained_model' is the correct argument for Cellpose v4+
    print(f"  > Loading model '{model_type}'...")
    try:
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_type)
    except TypeError:
        # Fallback for older versions
        model = models.CellposeModel(gpu=use_gpu, model_type=model_type)

    masks = None

    # --- 7. INFERENCE ---
    try:
        # We set net_avg=False for speed during testing.
        # We REMOVED 'channels' so Cellpose auto-detects grayscale.
        
        # Case A: True 3D Volume
        if process_3d and image.ndim == 3:
            print(f"  > Running 3D Inference on volume {image.shape}...")
            results = model.eval(
                image, 
                diameter=diameter, 
                flow_threshold=flow_threshold, 
                do_3D=True,
                net_avg=False # Speed up
            )

        # Case B: Batch 2D (Stack) or Single 2D
        else:
            print(f"  > Running 2D inference on {image.shape}...")
            results = model.eval(
                image, 
                diameter=diameter, 
                do_3D=False,
                flow_threshold=flow_threshold
                )

        # Unpack results (Cellpose returns tuple or array depending on version)
        if isinstance(results, tuple):
            masks = results[0]
        else:
            masks = results

    except Exception as e:
        print(f"❌ Cellpose Inference Failed: {e}")
        traceback.print_exc()
        raise e

    # --- 8. FORMAT OUTPUT ---
    masks = np.array(masks, dtype=np.uint32)

    # Return (Data, Metadata, LayerType)
    return (
        masks, 
        {"name": f"Masks ({model_type})", "opacity": 0.7}, 
        "labels"
    )