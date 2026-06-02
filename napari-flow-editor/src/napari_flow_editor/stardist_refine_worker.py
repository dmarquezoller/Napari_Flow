import sys
import os
import argparse
import numpy as np
import warnings
import shutil
import random
from pathlib import Path

# --- FORCE CPU SETTINGS (Matched to your working Train worker) ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
if "--gpu" not in sys.argv or sys.argv[sys.argv.index("--gpu") + 1] == "0":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

warnings.filterwarnings("ignore")

def refine_model(img_path, lbl_path, base_model_path, model_name, output_dir, epochs, learning_rate, use_gpu=False):
    print(f"REFINE WORKER: Starting Job (PID: {os.getpid()})")
    
    # --- 1. LOAD DATA (Identical to Train Worker) ---
    try:
        X_raw = np.load(img_path)
        Y_raw = np.load(lbl_path)
        print(f"REFINE WORKER: Loaded Data - Image {X_raw.shape}, Labels {Y_raw.shape}")
    except Exception as e:
        print(f"REFINE WORKER ERROR: Could not load data: {e}")
        sys.exit(1)

    # --- 2. PREPARE DATA (Identical to Train Worker) ---
    X_train_all = []
    Y_train_all = []

    # Handle 4D (Time/Z, Y, X, C)
    if X_raw.ndim == 4 and Y_raw.ndim == 3:
        print("REFINE WORKER: Processing 4D Stack (Multichannel/RGB)...")
        for i in range(X_raw.shape[0]):
            if np.max(Y_raw[i]) > 0: # Only use annotated frames
                X_train_all.append(X_raw[i])
                Y_train_all.append(Y_raw[i].astype(int))
                
    # Handle Standard 3D (Z, Y, X)
    elif X_raw.ndim == 3 and Y_raw.ndim == 3:
        print("REFINE WORKER: Processing 3D Stack (Grayscale)...")
        for i in range(X_raw.shape[0]):
            if np.max(Y_raw[i]) > 0:
                X_train_all.append(X_raw[i])
                Y_train_all.append(Y_raw[i].astype(int))

    # Handle Single 2D
    elif X_raw.ndim == 2:
        X_train_all = [X_raw]
        Y_train_all = [Y_raw.astype(int)]

    else:
        print(f"REFINE WORKER ERROR: Shape combination not supported. X:{X_raw.shape}, Y:{Y_raw.shape}")
        sys.exit(1)

    if len(X_train_all) == 0:
        print("REFINE WORKER ERROR: No labeled frames found.")
        sys.exit(1)

    print(f"REFINE WORKER: Found {len(X_train_all)} annotated frames.")

    # --- 3. IMPORT LIBRARIES ---
    try:
        from stardist import fill_label_holes
        from stardist.models import StarDist2D
        from csbdeep.utils import normalize
    except ImportError:
        print("REFINE WORKER ERROR: StarDist not installed.")
        sys.exit(1)

    # --- 4. PREPROCESS (Identical to Train Worker) ---
    print("REFINE WORKER: Normalizing...")
    X_norm = []
    for x in X_train_all:
        # Normalize spatially (axis 0,1), keeping channels (axis 2) independent
        axis_norm = (0,1) 
        X_norm.append(normalize(x, 1, 99.8, axis=axis_norm))

    Y_clean = [fill_label_holes(y) for y in Y_train_all]

    # --- 5. INITIALIZE MODEL (The Refine Logic) ---
    base_path = Path(base_model_path)
    if not base_path.exists():
        print(f"REFINE WORKER ERROR: Base model path not found: {base_path}")
        sys.exit(1)

    print(f"REFINE WORKER: Loading base model '{base_path.name}'...")
    try:
        # Load the OLD model to access its config and weights
        old_model = StarDist2D(None, name=base_path.name, basedir=str(base_path.parent))
    except Exception as e:
        print(f"REFINE WORKER ERROR: Failed to load base model. {e}")
        sys.exit(1)

    # Clean up destination if it exists
    target_dir = os.path.join(output_dir, model_name)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # Clone config from old model
    config = old_model.config
    
    # IMPORTANT: Update Learning Rate in the config before creating new model
    config.train_learning_rate = learning_rate
    config.train_epochs = epochs
    config.use_gpu = use_gpu

    print(f"REFINE WORKER: Creating new model '{model_name}' with updated config...")
    new_model = StarDist2D(config, name=model_name, basedir=output_dir)

    print("REFINE WORKER: Transferring weights from base model...")
    new_model.keras_model.set_weights(old_model.keras_model.get_weights())

    # --- 6. MANUAL VALIDATION SPLIT (Identical to Train Worker) ---
    print("REFINE WORKER: Splitting Validation Data...")
    
    # Create indices and shuffle them
    indices = np.arange(len(X_norm))
    np.random.shuffle(indices)
    
    # Calculate split (15% validation, minimum 1 frame)
    n_val = max(1, int(len(X_norm) * 0.15))
    
    # Special Case for Refinement: 
    # If we have only 1 image, we MUST use it for both train and val to avoid crashes
    if len(X_norm) == 1: 
        print("REFINE WORKER: Single image mode -> Using training data for validation.")
        n_val = 0 # No split
        X_t, Y_t = X_norm, Y_clean
        X_v, Y_v = X_norm, Y_clean
        validation_data = (X_v, Y_v)
    else:
        train_idx = indices[:-n_val]
        val_idx   = indices[-n_val:]
        
        X_t = [X_norm[i] for i in train_idx]
        Y_t = [Y_clean[i] for i in train_idx]
        
        X_v = [X_norm[i] for i in val_idx]
        Y_v = [Y_clean[i] for i in val_idx]
        
        validation_data = (X_v, Y_v)
        print(f"REFINE WORKER: Training on {len(X_t)} frames, Validating on {len(X_v)} frames.")

    # --- 7. TRAIN (Identical to Train Worker) ---
    print(f"REFINE WORKER: Starting loop ({epochs} epochs)...")
    
    # We always pass validation_data (even if it's the same as train data in 1-image case)
    # This prevents the "missing argument" error you saw earlier.
    new_model.train(X_t, Y_t, validation_data=validation_data, epochs=epochs)

    # --- 8. OPTIMIZE ---
    print("REFINE WORKER: Optimizing thresholds...")
    new_model.optimize_thresholds(X_v, Y_v)

    print("REFINE WORKER: Done. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Arguments matched to the new Node logic
    parser.add_argument("--img", required=True)
    parser.add_argument("--lbl", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    refine_model(args.img, args.lbl, args.base_model, args.name, args.outdir, args.epochs, args.lr, bool(args.gpu))