# src/napari_flow_editor/stardist_worker.py
import sys
import os
import argparse
import numpy as np
import warnings
import shutil

# 1. LOCK THREADS & FORCE CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

warnings.filterwarnings("ignore")

def run_inference(img_path, out_path, model_input, prob_thresh, nms_thresh, pmin, pmax, scale):
    print(f"WORKER: Starting StarDist (PID: {os.getpid()})...")
    
    # --- 1. SETUP ---
    model_input = model_input.strip().strip('"').strip("'")
    is_custom_path = os.path.isdir(model_input) or os.path.sep in model_input
    IS_3D_MODEL = "3d" in model_input.lower()
    
    # --- 2. IMPORT LIBRARIES ---
    try:
        from csbdeep.utils import normalize
        if IS_3D_MODEL:
            from stardist.models import StarDist3D
            ModelClass = StarDist3D
        else:
            from stardist.models import StarDist2D
            ModelClass = StarDist2D
    except ImportError:
        print("WORKER ERROR: StarDist library not installed.")
        sys.exit(1)

    # --- 3. LOAD IMAGE ---
    try:
        img = np.load(img_path)
        img = np.squeeze(img)
        print(f"WORKER: Raw Image Shape: {img.shape}")
    except Exception as e:
        print(f"WORKER ERROR: Could not load image: {e}")
        sys.exit(1)

    # --- 4. LOAD MODEL ---
    print(f"WORKER: Loading network weights...")
    model = None
    try:
        if is_custom_path:
            model_name = os.path.basename(model_input)
            base_dir = os.path.dirname(model_input)
            # Handle trailing slash edge case
            if not model_name: 
                model_name = os.path.basename(base_dir)
                base_dir = os.path.dirname(base_dir)
                
            model = ModelClass(None, name=model_name, basedir=base_dir)
        else:
            if model_input.lower() == '3d_demo': model_input = '3D_demo'
            model = ModelClass.from_pretrained(model_input)
    except Exception as e:
        print(f"WORKER ERROR: Failed to load model '{model_input}'. {e}")
        sys.exit(1)

    # --- 5. SMART CHANNEL MATCHING (The Fix) ---
    # Check what the model expects
    expected_channels = 1
    if hasattr(model.config, 'n_channel_in'):
        expected_channels = model.config.n_channel_in
    
    print(f"WORKER: Model expects {expected_channels} channel(s).")

    # 4D Case: (Time, Y, X, C)
    if img.ndim == 4:
        if expected_channels == 1:
            print("WORKER: Converting 4D (T,Y,X,C) -> 3D (T,Y,X) [Grayscale] to match model.")
            img = np.mean(img, axis=-1)
        else:
            print("WORKER: Keeping 4D (T,Y,X,C) [Color] to match model.")

    # 3D Case: Could be (Time, Y, X) OR (Y, X, C)
    elif img.ndim == 3:
        # Heuristic: If last dim is tiny (<=4), it's probably channels (Y, X, C)
        if img.shape[-1] <= 4:
            if expected_channels == 1:
                print("WORKER: Converting 3D (Y,X,C) -> 2D (Y,X) [Grayscale] to match model.")
                img = np.mean(img, axis=-1)
            else:
                print("WORKER: Keeping 3D (Y,X,C) [Color] to match model.")
        else:
            # It's (Time, Y, X)
            print("WORKER: Interpreting as Stack (Time, Y, X).")

    # --- 6. PREDICT ---
    print(f"WORKER: Running Prediction (Scale={scale})...")
    
    try:
        # 3D Volumetric Model
        if IS_3D_MODEL:
            img_norm = normalize(img, pmin=pmin, pmax=pmax, axis=(0,1,2))
            tiles = None
            if img.shape[0] > 128: tiles = (2, 2, 2)
            masks, _ = model.predict_instances(img_norm, prob_thresh=prob_thresh, nms_thresh=nms_thresh, n_tiles=tiles, scale=scale)

        # 2D / Stack Model
        else:
            # We process slice-by-slice to handle both Time-Lapse and Single Images consistently
            
            # Helper to normalize and predict a single frame
            def process_frame(frame):
                # Normalize spatially (axis 0,1)
                # If frame is (Y, X), axis=(0,1). If (Y, X, C), axis=(0,1) preserves channel ratios.
                frame_norm = normalize(frame, pmin=pmin, pmax=pmax, axis=(0,1))
                res, _ = model.predict_instances(frame_norm, prob_thresh=prob_thresh, nms_thresh=nms_thresh, scale=scale)
                return res

            # Case A: Stack of Frames (T, Y, X) or (T, Y, X, C)
            if img.ndim == 3 and expected_channels == 1: 
                # (T, Y, X)
                print(f"WORKER: Processing {img.shape[0]} grayscale frames...")
                results = [process_frame(img[i]) for i in range(img.shape[0])]
                masks = np.array(results)
                
            elif img.ndim == 4:
                # (T, Y, X, C)
                print(f"WORKER: Processing {img.shape[0]} color frames...")
                results = [process_frame(img[i]) for i in range(img.shape[0])]
                masks = np.array(results)

            # Case B: Single Frame (Y, X) or (Y, X, C)
            else:
                print("WORKER: Processing single frame...")
                masks = process_frame(img)

    except Exception as e:
        print(f"WORKER CRASH: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 7. SAVE ---
    print(f"WORKER: Saving output...")
    np.save(out_path, masks)
    print("WORKER: Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prob", type=float, default=0.5)
    parser.add_argument("--nms", type=float, default=0.3)
    parser.add_argument("--pmin", type=float, default=1.0)
    parser.add_argument("--pmax", type=float, default=99.8)
    parser.add_argument("--scale", type=float, default=1.0)
    
    args = parser.parse_args()
    
    run_inference(args.input, args.output, args.model, args.prob, args.nms, args.pmin, args.pmax, args.scale)