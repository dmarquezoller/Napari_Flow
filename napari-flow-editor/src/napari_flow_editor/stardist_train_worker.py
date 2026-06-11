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
if "--gpu" not in sys.argv or sys.argv[sys.argv.index("--gpu") + 1] == "0":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

warnings.filterwarnings("ignore")

def train_model(img_path, lbl_path, model_name, output_dir, epochs, patch_size, use_gpu=False):
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
    # One coherent rule: the LABELS define the samples; each image sample is
    # (H, W) [grayscale] or (H, W, C) [multichannel/RGB]. This covers single 2D
    # and stacks, grayscale and multichannel, uniformly -- no per-shape special cases.
    def _as_samples(X, Y):
        if Y.ndim == 2:
            Xs, Ys = [X], [Y]                       # single frame; X is (H,W) or (H,W,C)
        elif Y.ndim == 3:
            n = Y.shape[0]
            if X.shape[0] != n:
                raise ValueError(f"Sample-count mismatch: image {X.shape} vs labels {Y.shape}")
            Xs = [X[i] for i in range(n)]           # X[i] is (H,W) or (H,W,C)
            Ys = [Y[i] for i in range(n)]
        else:
            raise ValueError(f"Labels must be 2D or 3D (got {Y.shape}).")

        out_X, out_Y = [], []
        for x, y in zip(Xs, Ys):
            if x.shape[:2] != y.shape:              # spatial dims must align
                raise ValueError(f"Spatial mismatch: image {x.shape} vs label {y.shape}")
            if np.max(y) > 0:                       # keep only annotated samples
                out_X.append(x)
                out_Y.append(y.astype(int))
        return out_X, out_Y

    try:
        X_train_all, Y_train_all = _as_samples(X_raw, Y_raw)
    except ValueError as e:
        print(f"TRAINER ERROR: {e}")
        sys.exit(1)

    if len(X_train_all) == 0:
        print("TRAINER ERROR: No labeled frames found.")
        sys.exit(1)

    _ch = X_train_all[0].shape[-1] if X_train_all[0].ndim == 3 else 1
    print(f"TRAINER: {len(X_train_all)} annotated sample(s), {_ch} channel(s).")

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
        use_gpu      = use_gpu,
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
        print(f"TRAINER: Training on {len(X_t)} frames, Validating on {len(X_v)} frames.")
    else:
        # StarDist2D.train() requires validation_data. With a single annotated sample
        # there's nothing to hold out, so reuse the training sample for validation.
        # The validation metric is then meaningless (it's the training data) -- annotate
        # more frames for a model that actually generalises.
        X_v, Y_v = X_t, Y_t
        print("TRAINER: Warning - only one annotated sample; reusing it for validation "
              "(metric not meaningful). Annotate more frames for a usable model.")

    validation_data = (X_v, Y_v)

    # --- 7. TRAIN ---
    print(f"TRAINER: Starting loop ({epochs} epochs)...")
    history = model.train(X_t, Y_t, validation_data=validation_data)

    # --- 8. OPTIMIZE ---
    print("TRAINER: Optimizing thresholds...")
    model.optimize_thresholds(X_v, Y_v)

    print("TRAINER: Done. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--lbl", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    train_model(args.img, args.lbl, args.name, args.outdir, args.epochs, args.patch, bool(args.gpu))