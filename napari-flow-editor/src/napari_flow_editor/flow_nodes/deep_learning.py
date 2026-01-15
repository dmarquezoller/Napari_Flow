import numpy as np
import traceback
import logging
import sys
import os
import torch 
from datetime import datetime

from .decorator import register_node

# --- SAFE IMPORTS ---
try:
    from cellpose import models, core
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

# =============================================================================
# NODE 1: RUN CELLPOSE (INFERENCE) - UPDATED with Custom Path
# =============================================================================
@register_node(
    label="Cellpose Segmentation",
    category="Deep Learning",
    outputs=["mask_layer"],
    params_config={
        "model_type": {
            "options": ["nuclei", "cyto2", "cyto3", "tissuenet", "livecell"], 
            "value": "nuclei",
            "label": "Standard Model"
        },
        "custom_path": {
            "type": "path", 
            "mode": "file"
        },
        "diameter": {"min": 0.0, "max": 300.0, "step": 1.0, "value": 30.0},
        "flow_threshold": {"min": 0.0, "max": 1.0, "step": 0.1, "value": 0.4},
        "process_3d": {"type": "bool", "value": False, "label": "Process as 3D Vol"},
        "use_gpu": {"type": "bool", "value": False, "label": "Use GPU"} 
    }
)
def run_cellpose(image, 
                 model_type: str = 'nuclei',
                 custom_path: str = "", 
                 diameter: float = 30.0, 
                 flow_threshold: float = 0.4, 
                 process_3d: bool = False,
                 use_gpu: bool = False):
    
    if not CELLPOSE_AVAILABLE:
        raise ImportError("Cellpose not installed. Please run: pip install cellpose")

    _setup_logger()

    # --- 1. DETERMINE MODEL SOURCE ---
    # If a custom path is provided (and exists), use it. Otherwise use standard type.
    final_model_arg = model_type
    if custom_path and os.path.exists(custom_path):
        print(f"  > Using Custom Model: {custom_path}")
        final_model_arg = custom_path
    elif custom_path:
        print(f"  > ⚠️ Custom path not found: '{custom_path}'. Falling back to '{model_type}'")

    print(f"--- Cellpose Inference: {final_model_arg} (GPU={use_gpu}) ---")

    # --- 2. INPUT PREP ---
    image = _ensure_numpy(image)
    if image.dtype == bool:
        print("  > Converting Boolean image to uint8...")
        image = image.astype(np.uint8) * 255

    # --- 3. EXECUTION ---
    torch.set_num_threads(1) # Prevent Freeze

    print(f"  > Loading model...")
    try:
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=final_model_arg)
    except Exception as e:
        # Fallback for older versions or path errors
        print(f"  > Error loading model specific way, trying generic: {e}")
        model = models.CellposeModel(gpu=use_gpu, model_type=model_type)

    masks = None

    try:
        if process_3d and image.ndim == 3:
            print(f"  > Running 3D Inference on {image.shape}...")
            results = model.eval(image, diameter=diameter, flow_threshold=flow_threshold, do_3D=True)
        else:
            print(f"  > Running 2D Inference on {image.shape}...")
            results = model.eval(image, diameter=diameter, flow_threshold=flow_threshold, do_3D=False)

        if isinstance(results, tuple): masks = results[0]
        else: masks = results

    except Exception as e:
        print(f"❌ Inference Failed: {e}")
        traceback.print_exc()
        raise e

    masks = np.array(masks, dtype=np.uint32)
    return (masks, {"name": f"Masks ({os.path.basename(final_model_arg)})", "opacity": 0.7}, "labels")


# =============================================================================
# NODE 2: TRAIN CELLPOSE (NEW!)
# =============================================================================
@register_node(
    label="Train Cellpose Model",
    category="Deep Learning",
    outputs=["model_path"],
    params_config={
        "base_model": {
            "options": ["nuclei", "cyto2", "tissuenet", "livecell"], 
            "value": "nuclei",
            "label": "Base Model (Start Point)"
        },
        "model_name": {"type": "str", "value": "my_custom_model", "label": "New Model Name"},
        "epochs": {"min": 10, "max": 1000, "step": 10, "value": 50, "label": "Epochs"},
        "learning_rate": {"min": 0.001, "max": 1.0, "step": 0.001, "value": 0.1, "label": "Learning Rate"},
        "use_gpu": {"type": "bool", "value": False, "label": "Use GPU"}
    }
)
def train_cellpose(image, 
                   labels, 
                   base_model: str = 'nuclei', 
                   model_name: str = 'my_custom_model',
                   epochs: int = 50,
                   learning_rate: float = 0.1,
                   use_gpu: bool = False):
    
    if not CELLPOSE_AVAILABLE:
        raise ImportError("Cellpose not installed.")
    
    _setup_logger()
    print(f"--- Cellpose Training: {model_name} (GPU={use_gpu}) ---")
    
    # 1. Prepare Data
    # Cellpose expects lists of images/masks [Img1, Img2, ...]
    train_data = [_ensure_numpy(image)]
    train_labels = [_ensure_numpy(labels)]
    
    # Ensure labels are integers (uint16/32)
    train_labels[0] = train_labels[0].astype(np.uint16)

    # 2. Check Shapes
    if train_data[0].shape != train_labels[0].shape:
        raise ValueError(f"Shape Mismatch: Image {train_data[0].shape} vs Labels {train_labels[0].shape}")

    # 3. Setup Model
    torch.set_num_threads(1) # Critical for CPU training stability
    
    print(f"  > Initializing base model '{base_model}'...")
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=base_model)
    
    # 4. Setup Save Path
    # We save in a 'models' folder next to the plugin or in user home
    save_path = os.path.join(os.path.expanduser("~"), "napari_flow_models")
    os.makedirs(save_path, exist_ok=True)
    
    print(f"  > Starting Training ({epochs} epochs)... This may freeze the UI.")
    
    try:
        # 5. Train
        # 'channels' logic: For grayscale, we can omit or pass None in v4
        new_model_path = model.train(
            train_data,
            train_labels,
            test_data=None,
            save_path=save_path,
            model_name=model_name,
            n_epochs=epochs,
            learning_rate=learning_rate,
            channels=None, # Auto-detect grayscale
            rescale=True   # Essential if object sizes vary
        )
        
        print(f"✅ Training Complete!")
        print(f"  > Saved to: {new_model_path}")
        
    except Exception as e:
        print(f"❌ Training Failed: {e}")
        traceback.print_exc()
        raise e

    # Return the path so the user can copy-paste it
    return (str(new_model_path), {"name": "Model Path"}, "text")


# =============================================================================
# HELPERS
# =============================================================================
def _ensure_numpy(image):
    """Unwraps Napari layers/Dask arrays to pure NumPy"""
    raw_img = image
    if isinstance(image, list) and len(image) > 0:
        if isinstance(image[0], tuple): raw_img = image[0][0]
        else: raw_img = image[0]
    elif isinstance(image, tuple) and len(image) >= 2:
        raw_img = image[0]
    
    if hasattr(raw_img, "compute"):
        raw_img = raw_img.compute()
        
    return np.asarray(raw_img)

def _setup_logger():
    """Hijacks Cellpose logger to print to GUI"""
    cp_logger = logging.getLogger("cellpose")
    cp_logger.setLevel(logging.INFO)
    if not cp_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        cp_logger.addHandler(handler)