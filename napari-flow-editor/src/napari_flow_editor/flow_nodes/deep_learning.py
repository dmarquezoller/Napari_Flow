import numpy as np
import sys
import os
import tempfile
import subprocess
import logging
import traceback
import pandas as pd
from pathlib import Path
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
    description="Instance segmentation using Cellpose. Accepts pretrained or custom models. Runs on CPU or GPU (PyTorch).",
    outputs=["mask_layer"],
    input_types={"image": "image"},
    output_types={"mask_layer": "labels"},
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

    import torch

    if not CELLPOSE_AVAILABLE:
        raise ImportError("Cellpose not installed. Please run: pip install cellpose")

    _setup_logger()

    # --- 1. DETERMINE MODEL SOURCE ---
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
    torch.set_num_threads(1)

    print(f"  > Loading model...")
    try:
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=final_model_arg)
    except Exception as e:
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

        if isinstance(results, tuple):
            masks = results[0]
        else:
            masks = results

    except Exception as e:
        print(f"❌ Inference Failed: {e}")
        traceback.print_exc()
        raise e

    masks = np.array(masks, dtype=np.uint32)
    return (masks, {"name": f"Masks ({os.path.basename(final_model_arg)})", "opacity": 0.7, "layer_type": "labels"})


@register_node(
    label="Train Cellpose Model",
    category="Deep Learning",
    description="Fine-tune a Cellpose model on labelled images. Saves model to ~/napari_flow_models/. Path is printed to console on completion.",
    outputs=[],
    input_types={"image": "image", "labels": "labels"},
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

    import torch

    if not CELLPOSE_AVAILABLE:
        raise ImportError("Cellpose not installed.")

    _setup_logger()
    print(f"--- Cellpose Training: {model_name} (GPU={use_gpu}) ---")

    train_data = [_ensure_numpy(image)]
    train_labels = [_ensure_numpy(labels)]
    train_labels[0] = train_labels[0].astype(np.uint16)

    if train_data[0].shape != train_labels[0].shape:
        raise ValueError(f"Shape Mismatch: Image {train_data[0].shape} vs Labels {train_labels[0].shape}")

    torch.set_num_threads(1)

    print(f"  > Initializing base model '{base_model}'...")
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=base_model)

    save_path = os.path.join(os.path.expanduser("~"), "napari_flow_models")
    os.makedirs(save_path, exist_ok=True)

    print(f"  > Starting Training ({epochs} epochs)... This may freeze the UI.")

    try:
        new_model_path = model.train(
            train_data,
            train_labels,
            test_data=None,
            save_path=save_path,
            model_name=model_name,
            n_epochs=epochs,
            learning_rate=learning_rate,
            channels=None,
            rescale=True
        )
        print(f"✅ Training Complete! Model saved to: {new_model_path}")

    except Exception as e:
        print(f"❌ Training Failed: {e}")
        traceback.print_exc()
        raise e

    return None


# =============================================================================
# STARDIST NODES
# =============================================================================

@register_node(
    label="StarDist Segmentation",
    category="Deep Learning",
    description="Instance segmentation using StarDist. Runs in an isolated subprocess to prevent TF/Keras memory conflicts.",
    outputs=["mask_layer"],
    input_types={"image": "image"},
    output_types={"mask_layer": "labels"},
    params_config={
        "model_type": {
            "options": ["2D_versatile_fluo", "2D_versatile_he", "3D_demo"],
            "value": "2D_versatile_fluo",
            "label": "Pretrained Model"
        },
        "custom_path": {
            "type": "path",
            "mode": "directory",
            "label": "Custom Model Folder (Optional)"
        },
        "scale": {"min": 0.1, "max": 2.0, "step": 0.1, "value": 1.0, "label": "Scale"},
        "prob_thresh": {"min": 0.0, "max": 1.0, "step": 0.05, "value": 0.5},
        "nms_thresh": {"min": 0.0, "max": 1.0, "step": 0.05, "value": 0.3},
        "norm_pmin": {"min": 0.0, "max": 100.0, "step": 1.0, "value": 1.0},
        "norm_pmax": {"min": 0.0, "max": 100.0, "step": 1.0, "value": 99.8},
        "use_gpu": {"type": "bool", "value": False, "label": "Use GPU"},
    }
)
def run_stardist(image,
                 model_type: str = '2D_versatile_fluo',
                 custom_path: str = "",
                 scale: float = 1.0,
                 prob_thresh: float = 0.5,
                 nms_thresh: float = 0.3,
                 norm_pmin: float = 1.0,
                 norm_pmax: float = 99.8,
                 use_gpu: bool = False):

    final_model_id = model_type

    if custom_path and isinstance(custom_path, str) and os.path.isdir(custom_path):
        print(f"🔹 Using Custom Model: {custom_path}")
        final_model_id = custom_path
    else:
        print(f"🔹 Using Pretrained Model: {final_model_id}")

    image = _ensure_numpy(image)
    if image.dtype == bool:
        image = image.astype(np.float32)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    worker_script = os.path.join(parent_dir, "stardist_worker.py")

    if not os.path.exists(worker_script):
        worker_script = os.path.join(current_dir, "..", "stardist_worker.py")
        if not os.path.exists(worker_script):
            raise FileNotFoundError(f"Could not find stardist_worker.py at {worker_script}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        in_path = os.path.join(tmp_dir, "input.npy")
        out_path = os.path.join(tmp_dir, "output.npy")

        print(f"--- Launching StarDist Subprocess ---")
        np.save(in_path, image)

        cmd = [
            sys.executable, worker_script,
            "--input", in_path,
            "--output", out_path,
            "--model", final_model_id,
            "--prob", str(prob_thresh),
            "--nms", str(nms_thresh),
            "--pmin", str(norm_pmin),
            "--pmax", str(norm_pmax),
            "--scale", str(scale),
            "--gpu", "1" if use_gpu else "0",
        ]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        if result.returncode != 0:
            print("❌ Worker Failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"StarDist crashed: {result.stderr}")
        else:
            print("  > Worker finished successfully.")
            print(result.stdout)

        if not os.path.exists(out_path):
            raise RuntimeError("Worker finished but output file is missing.")

        masks = np.load(out_path)

    return (
        masks.astype(np.uint32),
        {"name": f"StarDist ({os.path.basename(final_model_id)})", "opacity": 0.7, "layer_type": "labels"},
    )


@register_node(
    label="Train StarDist Model",
    category="Deep Learning",
    description="Train a StarDist 2D model from scratch. Saves model to napari_flow_editor/custom_models/.",
    outputs=[],
    input_types={"image": "image", "labels": "labels"},
    params_config={
        "model_name": {"type": "str", "value": "my_custom_nuclei", "label": "New Model Name"},
        "epochs": {"min": 10, "max": 500, "value": 50, "label": "Epochs"},
        "patch_size": {"options": [128, 256, 512], "value": 256, "label": "Patch Size"},
        "use_gpu": {"type": "bool", "value": False, "label": "Use GPU"},
    }
)
def train_stardist(image, labels, model_name="my_custom_nuclei", epochs=50, patch_size=256, use_gpu=False):

    image = _ensure_numpy(image)
    labels = _ensure_numpy(labels)

    # Shape validation
    if image.shape == labels.shape:
        pass
    elif image.ndim == labels.ndim + 1 and image.shape[:-1] == labels.shape:
        pass
    else:
        raise ValueError(f"Shape mismatch! Image {image.shape} vs Labels {labels.shape}. "
                         "Ensure frames match and labels are single-channel.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    worker_script = os.path.join(parent_dir, "stardist_train_worker.py")

    if not os.path.exists(worker_script):
        worker_script = os.path.join(current_dir, "..", "stardist_train_worker.py")

    models_dir = os.path.join(parent_dir, "custom_models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

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
            "--patch", str(patch_size),
            "--gpu", "1" if use_gpu else "0",
        ]

        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError("Training Failed! Check terminal for errors.")

        print(f"✅ Training Complete!")
        print(f"📁 Model saved to: {os.path.join(models_dir, model_name)}")

    return None


@register_node(
    label="Refine StarDist Model",
    category="Deep Learning",
    description="Fine-tune an existing StarDist model on new labelled data.",
    outputs=[],
    input_types={"image": "image", "labels": "labels"},
    params_config={
        "base_model_path": {"type": "path", "label": "Base Model Folder", "mode": "directory"},
        "new_model_name": {"type": "text", "label": "New Model Name", "value": "refined_model_v1"},
        "epochs": {"min": 1, "max": 1000, "value": 50, "label": "Epochs"},
        "learning_rate": {"min": 0.00001, "max": 0.01, "value": 0.0001, "step": 0.00001, "label": "Learning Rate"},
        "use_gpu": {"type": "bool", "value": False, "label": "Use GPU"},
    }
)
def refine_stardist_model(image, labels, base_model_path="", new_model_name="refined_model_v1",
                          epochs=50, learning_rate=0.0001, use_gpu=False):

    if image is None or labels is None:
        raise ValueError("Image and Labels are required.")

    image = _ensure_numpy(image)
    labels = _ensure_numpy(labels)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    worker_script = os.path.join(parent_dir, "stardist_refine_worker.py")

    if not os.path.exists(worker_script):
        worker_script = os.path.join(current_dir, "..", "stardist_refine_worker.py")

    print(f"--- Refine Node: Preparing Worker ---")

    with tempfile.TemporaryDirectory() as tmp_dir:
        img_path = os.path.join(tmp_dir, "train_img.npy")
        lbl_path = os.path.join(tmp_dir, "train_lbl.npy")

        np.save(img_path, image)
        np.save(lbl_path, labels)

        models_dir = os.path.join(parent_dir, "custom_models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        print(f"   > Launching Refinement Job: {new_model_name}")

        cmd = [
            sys.executable, worker_script,
            "--img", img_path,
            "--lbl", lbl_path,
            "--base_model", str(base_model_path),
            "--name", str(new_model_name),
            "--outdir", models_dir,
            "--epochs", str(epochs),
            "--lr", str(learning_rate),
            "--gpu", "1" if use_gpu else "0",
        ]

        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError("Refinement Worker Failed! Check terminal for errors.")

        final_path = os.path.join(models_dir, new_model_name)
        print(f"✅ Refinement Complete!")
        print(f"📁 Model saved to: {final_path}")

    return None


# =============================================================================
# ULTRACK NODES
# =============================================================================
@register_node(
    label="Ultrack Tracking",
    category="Deep Learning",
    description="Cell tracking via Ultrack. Input must be a label mask time-series (T,Y,X) or (T,Z,Y,X).",
    outputs=["detection_layer", "edges_layer", "tracks_layer"],
    input_types={"labels": "labels"},
    output_types={
        "detection_layer": "image",
        "edges_layer": "image",
        "tracks_layer": "tracks",
    },
    params_config={
        "config_path": {
            "type": "path",
            "mode": "file",
            "label": "Config File (Optional, .toml)",
            "value": ""
        },
        "time_limit": {
            "type": "int",
            "min": 30, "max": 36000, "step": 30,
            "value": 600,
            "label": "Time Limit (s)"
        },
        "solution_gap": {
            "type": "float",
            "min": 0.0001, "max": 0.1, "step": 0.001,
            "value": 0.001,
            "label": "Solution Gap"
        },
        "window_size": {
            "type": "int",
            "min": 0, "max": 500, "step": 1,
            "value": 0,
            "label": "Window Size (0 = off)"
        },
        "max_distance": {
            "type": "float",
            "min": 0.0, "max": 500.0, "step": 1.0,
            "value": 0.0,
            "label": "Max Distance px (0 = auto)"
        },
        "min_area": {
            "type": "int",
            "min": 0, "max": 10000, "step": 10,
            "value": 0,
            "label": "Min Area px (0 = auto)"
        },
        "max_area": {
            "type": "int",
            "min": 0, "max": 500000, "step": 100,
            "value": 0,
            "label": "Max Area px (0 = auto)"
        },
    }
)
def run_ultrack_node(labels, config_path="", time_limit: int = 600,
                     solution_gap: float = 0.001, window_size: int = 0,
                     max_distance: float = 0.0, min_area: int = 0, max_area: int = 0):
    print("\n--- Ultrack: Initializing Pipeline ---")

    try:
        import ultrack
        from ultrack.config import MainConfig, load_config
        from ultrack import to_tracks_layer

        try:
            from ultrack.utils import estimate_parameters_from_labels
        except ImportError:
            try:
                from ultrack.utils.estimation import estimate_parameters_from_labels
            except Exception:
                estimate_parameters_from_labels = None

        try:
            from ultrack.utils import labels_to_contours
        except ImportError:
            try:
                from ultrack.core.segmentation.processing import labels_to_contours
            except Exception:
                if hasattr(ultrack, "labels_to_contours"):
                    labels_to_contours = ultrack.labels_to_contours
                else:
                    raise ImportError("Could not find labels_to_contours")

    except ImportError as e:
        raise ImportError(f"Ultrack import failed: {e}")

    if hasattr(labels, "compute"):
        labels = labels.compute()
    labels = np.asarray(labels)
    if labels.ndim not in [3, 4]:
        raise ValueError("Input must be (T, Y, X) or (T, Z, Y, X)")
    print(f"Data Shape: {labels.shape}")

    print("Step 1: Configuring...")

    # Safe fallback estimates (overwritten below if auto-estimation succeeds)
    estimated_max_dist = 15.0
    estimated_min_area = 100
    estimated_max_area = 1_000_000

    if config_path and os.path.exists(config_path):
        print(f"   > Loading config from: {config_path}")
        try:
            cfg = load_config(config_path)
        except Exception as e:
            print(f"   ❌ Error loading config file: {e}. Falling back to defaults.")
            cfg = MainConfig()
    else:
        print("   > No config selected. Auto-estimating from labels...")
        cfg = MainConfig()
        try:
            if estimate_parameters_from_labels:
                est_df = estimate_parameters_from_labels(labels, is_timelapse=True)
                if 'distance' in est_df.columns:
                    estimated_max_dist = float(est_df['distance'].quantile(0.95) * 1.5)
                if 'area' in est_df.columns:
                    estimated_min_area = max(1, int(est_df['area'].quantile(0.05)))
                    estimated_max_area = int(est_df['area'].quantile(0.95) * 2)
                print(f"   > Estimated: max_dist={estimated_max_dist:.1f}  "
                      f"min_area={estimated_min_area}  max_area={estimated_max_area}")
            else:
                print("   > estimate_parameters_from_labels unavailable, using defaults.")
        except Exception as e:
            print(f"   > Estimation failed ({e}), using fallback defaults.")

    import os as _os
    n_cpu = _os.cpu_count() or 1
    cfg.linking_config.n_workers = n_cpu
    cfg.segmentation_config.n_workers = n_cpu

    cfg.linking_config.max_distance   = max_distance if max_distance > 0 else estimated_max_dist
    cfg.segmentation_config.min_area  = min_area if min_area > 0 else estimated_min_area
    cfg.segmentation_config.max_area  = max_area if max_area > 0 else estimated_max_area

    cfg.tracking_config.time_limit   = time_limit
    cfg.tracking_config.solution_gap = solution_gap
    cfg.tracking_config.window_size  = window_size if window_size > 0 else None

    print(f"   > Linking:  max_distance={cfg.linking_config.max_distance:.1f}  n_workers={n_cpu}")
    print(f"   > Segmentation: min_area={cfg.segmentation_config.min_area}  "
          f"max_area={cfg.segmentation_config.max_area}")
    print(f"   > Solver: time_limit={time_limit}s  gap={solution_gap}  window={window_size or 'off'}")

    print("Step 2: Computing contours (labels_to_contours)...")
    detection, edges = labels_to_contours(labels, sigma=0.0)

    print("Step 3: Solving tracking...")
    ultrack.track(
        cfg,
        detection=detection,
        edges=edges,
        overwrite=True
    )

    print("Step 4: Formatting layers...")
    try:
        tracks_df, graph = to_tracks_layer(cfg)

        tracks_data = tracks_df.copy()

        clean_graph = {child: list(parents) if isinstance(parents, (list, tuple)) else [parents]
                       for child, parents in graph.items()}

    except Exception as e:
        print(f"⚠️ No tracks found: {e}")
        tracks_data = pd.DataFrame(columns=["track_id", "t", "y", "x"])
        clean_graph = {}

    print("--- Finished ---")

    return (
        (detection, {"name": "Ultrack Detection", "layer_type": "image", "colormap": "gray", "contrast_limits": [0, 1]}),
        (edges, {"name": "Ultrack Edges", "layer_type": "image", "colormap": "magenta", "contrast_limits": [0, 1]}),
        (tracks_data, {"name": "Ultrack Lineages", "graph": clean_graph, "layer_type": "tracks", "colormap": "turbo"}),
    )


# =============================================================================
# HELPERS
# =============================================================================

def _ensure_numpy(image):
    """Unwrap Napari layers/Dask arrays to pure NumPy."""
    raw_img = image
    if isinstance(image, list) and len(image) > 0:
        if isinstance(image[0], tuple):
            raw_img = image[0][0]
        else:
            raw_img = image[0]
    elif isinstance(image, tuple) and len(image) >= 2:
        raw_img = image[0]

    if hasattr(raw_img, "compute"):
        raw_img = raw_img.compute()

    return np.asarray(raw_img)


def _setup_logger():
    """Route Cellpose logger output to stdout."""
    cp_logger = logging.getLogger("cellpose")
    cp_logger.setLevel(logging.INFO)
    if not cp_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        cp_logger.addHandler(handler)
