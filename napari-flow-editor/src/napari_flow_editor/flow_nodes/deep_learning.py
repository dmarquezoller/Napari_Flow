import numpy as np
import sys
import os
import tempfile
import subprocess
import torch
import logging
import traceback

from .decorator import register_node

# --- SAFE IMPORTS ---
try:
    from cellpose import models, core
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

# =============================================================================
# CELLPOSE NODES
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
# STARDIST NODES
# =============================================================================


@register_node(
    label="StarDist Segmentation",
    category="Deep Learning",
    outputs=["mask_layer"],
    params_config={
        "model_type": {
            "options": ["2D_versatile_fluo", "2D_versatile_he", "3D_demo"], 
            "value": "2D_versatile_fluo",
            "label": "Model"
        },
        "scale": {"min": 0.1, "max": 2.0, "step": 0.1, "value": 1.0, "label": "Scale"},
        "prob_thresh": {"min": 0.0, "max": 1.0, "step": 0.05, "value": 0.5},
        "nms_thresh": {"min": 0.0, "max": 1.0, "step": 0.05, "value": 0.3},
        "norm_pmin": {"min": 0.0, "max": 100.0, "step": 1.0, "value": 1.0},
        "norm_pmax": {"min": 0.0, "max": 100.0, "step": 1.0, "value": 99.8}
    }
)
def run_stardist(image, 
                 model_type: str = '2D_versatile_fluo', 
                 scale: float = 1.0,
                 prob_thresh: float = 0.5, 
                 nms_thresh: float = 0.3,
                 norm_pmin: float = 1.0,
                 norm_pmax: float = 99.8):
    
    # 1. Prepare Paths
    image = _ensure_numpy(image)
    if image.dtype == bool: image = image.astype(np.float32)
    
    # Locate the worker script
    # Assumes stardist_worker.py is in the parent folder of flow_nodes
    current_dir = os.path.dirname(os.path.abspath(__file__)) # .../flow_nodes
    parent_dir = os.path.dirname(current_dir)                # .../napari_flow_editor
    worker_script = os.path.join(parent_dir, "stardist_worker.py")
    
    if not os.path.exists(worker_script):
        # Fallback for different install structures
        worker_script = os.path.join(current_dir, "..", "stardist_worker.py")
        if not os.path.exists(worker_script):
             raise FileNotFoundError(f"Could not find stardist_worker.py at {worker_script}")

    # 2. Create Temp Files
    # We use a temp directory to pass data safely
    with tempfile.TemporaryDirectory() as tmp_dir:
        in_path = os.path.join(tmp_dir, "input.npy")
        out_path = os.path.join(tmp_dir, "output.npy")
        
        # 3. Save Image for Worker
        print(f"--- Launching StarDist Subprocess ---")
        np.save(in_path, image)
        
        # 4. Build Command
        # We use sys.executable to ensure we use the SAME python environment (conda env)
        cmd = [
            sys.executable, worker_script,
            "--input", in_path,
            "--output", out_path,
            "--model", model_type,
            "--prob", str(prob_thresh),
            "--nms", str(nms_thresh),
            "--pmin", str(norm_pmin),
            "--pmax", str(norm_pmax),
            "--scale", str(scale)
        ]
        
        # 5. Execute
        print(f"  > Running: {' '.join(cmd)}")
        
        # Check for environment issues
        env = os.environ.copy()
        # Force Python to not buffer output so we see errors
        env["PYTHONUNBUFFERED"] = "1"

        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        # 6. Check Results
        if result.returncode != 0:
            print("❌ Worker Failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"StarDist crashed: {result.stderr}")
        else:
            print("  > Worker finished successfully.")
            # Print worker logs for debugging
            print(result.stdout)
            
        # 7. Load Result
        if not os.path.exists(out_path):
            raise RuntimeError("Worker finished but output file is missing.")
            
        masks = np.load(out_path)

    return (
        masks.astype(np.uint32), 
        {"name": f"StarDist ({model_type})", "opacity": 0.7}, 
        "labels"
    )



@register_node(
    label="Train StarDist Model",
    category="Deep Learning",
    outputs=[], # No output data, it produces a file on disk
    params_config={
        "model_name": {"type": "str", "value": "my_custom_nuclei", "label": "New Model Name"},
        "epochs": {"min": 10, "max": 500, "value": 50, "label": "Epochs"},
        "patch_size": {"options": [128, 256, 512], "value": 256, "label": "Patch Size"},
    }
)
def train_stardist(image, labels, model_name="my_custom_nuclei", epochs=50, patch_size=256):
    
    # 1. Validation
    image = _ensure_numpy(image)
    labels = _ensure_numpy(labels)
    
    # Check 1: Exact match (Ideal for Grayscale)
    if image.shape == labels.shape:
        pass
    
    # Check 2: Image has extra channel dim (e.g. RGB or Multi-channel)
    # Checks if Image is (T, Y, X, C) and Labels is (T, Y, X)
    elif image.ndim == labels.ndim + 1 and image.shape[:-1] == labels.shape:
        pass
        
    # Check 3: Image is (Y, X, C) and Labels is (Y, X)
    elif image.ndim == labels.ndim + 1 and image.shape[:-1] == labels.shape:
        pass

    else:
        # If none of the above, it's a real error
        raise ValueError(f"Shape mismatch! Image {image.shape} vs Labels {labels.shape}. "
                         "Ensure frames match and labels are single-channel.")
    
    # 2. Locate Worker
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    worker_script = os.path.join(parent_dir, "stardist_train_worker.py")
    
    if not os.path.exists(worker_script):
        # Fallback check
        worker_script = os.path.join(current_dir, "..", "stardist_train_worker.py")
    
    # 3. Define Model Output Folder
    # We save models to a "models" folder inside your napari-flow-editor directory
    # so they are easy to find later.
    models_dir = os.path.join(parent_dir, "custom_models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # 4. Run Training via Subprocess
    with tempfile.TemporaryDirectory() as tmp_dir:
        print("--- Prepare Training Data ---")
        img_path = os.path.join(tmp_dir, "train_img.npy")
        lbl_path = os.path.join(tmp_dir, "train_lbl.npy")
        
        np.save(img_path, image)
        np.save(lbl_path, labels)
        
        print(f"--- Launching Training Job: {model_name} ---")
        cmd = [
            sys.executable, worker_script,
            "--img", img_path,
            "--lbl", lbl_path,
            "--name", model_name,
            "--outdir", models_dir,
            "--epochs", str(epochs),
            "--patch", str(patch_size)
        ]
        
        # We allow output to stream to console so you can see the progress bar
        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        proc.wait()
        
        if proc.returncode != 0:
            raise RuntimeError("Training Failed! Check terminal for errors.")
            
        print(f"✅ Training Complete!")
        print(f"📁 Model saved to: {os.path.join(models_dir, model_name)}")
        print("💡 Restart Napari or Refresh to see it in the Segmentation node options.")

    return None













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