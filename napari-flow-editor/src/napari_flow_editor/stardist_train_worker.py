import sys
import os
import argparse
import numpy as np
import warnings
import shutil
import random

# Force CPU settings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

warnings.filterwarnings("ignore")

def train_model(img_path, lbl_path, model_name, output_dir, epochs, patch_size):
    print(f"TRAINER: Starting Training Job (PID: {os.getpid()})")
    
    # --- 1. LOAD DATA ---
    try:
        X_raw = np.load(img_path)
        Y_raw = np.load(lbl_path)
        print(f"TRAINER: Loaded Data - Image {X_raw.shape}, Labels {Y_raw.shape}")
    except Exception as e:
        print(f"TRAINER ERROR: Could not load data: {e}")
        sys.exit(1)

    # --- 2. PREPARE DATA ---
    X_train_all = []
    Y_train_all = []

    # Handle 4D (Time/Z, Y, X, C) - Your specific case
    if X_raw.ndim == 4 and Y_raw.ndim == 3:
        print("TRAINER: Processing 4D Stack (Multichannel/RGB)...")
        for i in range(X_raw.shape[0]):
            if np.max(Y_raw[i]) > 0: # Only use annotated frames
                X_train_all.append(X_raw[i])
                Y_train_all.append(Y_raw[i].astype(int))
                
    # Handle Standard 3D (Z, Y, X)
    elif X_raw.ndim == 3 and Y_raw.ndim == 3:
        print("TRAINER: Processing 3D Stack (Grayscale)...")
        for i in range(X_raw.shape[0]):
            if np.max(Y_raw[i]) > 0:
                X_train_all.append(X_raw[i])
                Y_train_all.append(Y_raw[i].astype(int))

    # Handle Single 2D
    elif X_raw.ndim == 2:
        X_train_all = [X_raw]
        Y_train_all = [Y_raw.astype(int)]

    else:
        print(f"TRAINER ERROR: Shape combination not supported. X:{X_raw.shape}, Y:{Y_raw.shape}")
        sys.exit(1)

    if len(X_train_all) == 0:
        print("TRAINER ERROR: No labeled frames found.")
        sys.exit(1)

    print(f"TRAINER: Found {len(X_train_all)} annotated frames.")

    # --- 3. IMPORT LIBRARIES ---
    try:
        from stardist import fill_label_holes
        from stardist.models import Config2D, StarDist2D
        from csbdeep.utils import normalize
    except ImportError:
        print("TRAINER ERROR: StarDist not installed.")
        sys.exit(1)

    # --- 4. PREPROCESS ---
    print("TRAINER: Normalizing...")
    X_norm = []
    for x in X_train_all:
        # Normalize spatially (axis 0,1), keeping channels (axis 2) independent if they exist
        axis_norm = (0,1) 
        X_norm.append(normalize(x, 1, 99.8, axis=axis_norm))

    Y_clean = [fill_label_holes(y) for y in Y_train_all]

    # --- 5. CONFIGURATION ---
    sample_img = X_norm[0]
    n_channels = 1
    if sample_img.ndim == 3:
        n_channels = sample_img.shape[-1]
    
    print(f"TRAINER: Architecture set for {n_channels} channel(s).")

    conf = Config2D(
        n_rays       = 32,
        grid         = (2, 2),
        n_channel_in = n_channels,
        train_epochs = epochs,
        train_steps_per_epoch = max(10, len(X_norm)*2), # Dynamic steps based on data size
        train_patch_size = (patch_size, patch_size), 
        use_gpu      = False
    )

    # Cleanup old model
    if os.path.exists(os.path.join(output_dir, model_name)):
        shutil.rmtree(os.path.join(output_dir, model_name))

    print(f"TRAINER: Initializing Model '{model_name}'...")
    model = StarDist2D(conf, name=model_name, basedir=output_dir)

    # --- 6. MANUAL VALIDATION SPLIT (FIX FOR CRASH) ---
    print("TRAINER: Splitting Validation Data...")
    
    # Create indices and shuffle them
    indices = np.arange(len(X_norm))
    np.random.shuffle(indices)
    
    # Calculate split (15% validation, minimum 1 frame)
    n_val = max(1, int(len(X_norm) * 0.15))
    if len(X_norm) == 1: n_val = 0 # Cannot validate if only 1 image
    
    train_idx = indices[:-n_val] if n_val > 0 else indices
    val_idx   = indices[-n_val:] if n_val > 0 else []

    X_t = [X_norm[i] for i in train_idx]
    Y_t = [Y_clean[i] for i in train_idx]
    
    if n_val > 0:
        X_v = [X_norm[i] for i in val_idx]
        Y_v = [Y_clean[i] for i in val_idx]
        validation_data = (X_v, Y_v)
        print(f"TRAINER: Training on {len(X_t)} frames, Validating on {len(X_v)} frames.")
    else:
        validation_data = None
        print("TRAINER: Warning - Not enough data for validation. Training on all frames.")

    # --- 7. TRAIN ---
    print(f"TRAINER: Starting loop ({epochs} epochs)...")
    
    # Pass the manual tuple 'validation_data' instead of 'validation_split'
    if validation_data:
        history = model.train(X_t, Y_t, validation_data=validation_data)
    else:
        history = model.train(X_t, Y_t)

    # --- 8. OPTIMIZE ---
    print("TRAINER: Optimizing thresholds...")
    if validation_data:
        model.optimize_thresholds(X_v, Y_v)
    else:
        model.optimize_thresholds(X_t, Y_t)

    print("TRAINER: Done. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--lbl", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patch", type=int, default=256)
    
    args = parser.parse_args()
    train_model(args.img, args.lbl, args.name, args.outdir, args.epochs, args.patch)