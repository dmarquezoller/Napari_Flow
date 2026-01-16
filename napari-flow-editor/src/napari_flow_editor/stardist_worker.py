import sys
import os
import argparse
import numpy as np
import warnings

# Force CPU settings for stability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

warnings.filterwarnings("ignore")

def run_inference(img_path, out_path, model_type, prob_thresh, nms_thresh, pmin, pmax, scale):
    print(f"WORKER: Starting StarDist (Model: {model_type})...")
    
    # --- 1. DETECT 2D vs 3D MODE ---
    # The standard pretrained 3D model is called '3D_demo'
    IS_3D_MODEL = "3D" in model_type
    
    # --- 2. LOAD IMAGE ---
    try:
        img = np.load(img_path)
    except Exception as e:
        print(f"WORKER ERROR: Could not load image: {e}")
        sys.exit(1)

    # Clean up shape (remove singleton dims like 1, 1, 512, 512)
    img = np.squeeze(img)

    # Handle RGB (if user accidentally passed color image)
    if img.ndim >= 3 and img.shape[-1] in [3, 4] and img.shape[-1] < 20:
        img = np.mean(img, axis=-1)

    print(f"WORKER: Input Shape: {img.shape} | Mode: {'3D Volumetric' if IS_3D_MODEL else '2D/Stack'}")

    # --- 3. DYNAMIC IMPORT ---
    try:
        from csbdeep.utils import normalize
        if IS_3D_MODEL:
            from stardist.models import StarDist3D
            ModelClass = StarDist3D
        else:
            from stardist.models import StarDist2D
            ModelClass = StarDist2D
            
    except ImportError:
        print("WORKER ERROR: StarDist not installed.")
        sys.exit(1)

    # --- 4. NORMALIZE ---
    print(f"WORKER: Normalizing ({pmin}% - {pmax}%)...")
    # For True 3D, we normalize the whole volume together (axis=None usually works well, 
    # but (0,1,2) is explicit for 3D).
    if IS_3D_MODEL:
        img = normalize(img, pmin=pmin, pmax=pmax, axis=(0,1,2))
    else:
        # For 2D, we normalize each slice independently
        axis = (0,1) if img.ndim == 2 else (1,2)
        img = normalize(img, pmin=pmin, pmax=pmax, axis=axis)

    # --- 5. LOAD & PREDICT ---
    print(f"WORKER: Loading {model_type}...")
    try:
        model = ModelClass.from_pretrained(model_type)
    except Exception as e:
        print(f"WORKER ERROR: Failed to load model '{model_type}': {e}")
        sys.exit(1)

    print("WORKER: Running Prediction (Scale={scale})...")
    masks = None

    if IS_3D_MODEL:
        # --- TRUE 3D EXECUTION ---
        if img.ndim != 3:
            print(f"WORKER ERROR: 3D Model requires 3D input (Z,Y,X). Got {img.ndim}D.")
            sys.exit(1)
            
        # n_tiles helps prevent running out of RAM on big 3D stacks
        # We try to infer reasonable tiles if image is huge
        tiles = None
        if img.shape[1] > 1024 or img.shape[2] > 1024:
            tiles = (1, 2, 2) 
            print(f"WORKER: Large image detected, using tiling {tiles}")

        masks, _ = model.predict_instances(img, prob_thresh=prob_thresh, nms_thresh=nms_thresh, n_tiles=tiles, scale=scale)
        
    else:
        # --- 2D / STACK EXECUTION ---
        if img.ndim == 2:
            masks, _ = model.predict_instances(img, prob_thresh=prob_thresh, nms_thresh=nms_thresh, scale=scale)
        elif img.ndim == 3:
            # Loop through Z-stack (Slice by Slice)
            print(f"WORKER: Processing stack of {img.shape[0]} slices...")
            results = []
            for i in range(img.shape[0]):
                res, _ = model.predict_instances(img[i], prob_thresh=prob_thresh, nms_thresh=nms_thresh, scale=scale)
                results.append(res)
            masks = np.array(results)
        elif img.ndim == 4:
             # Flatten Hyperstack (T*Z, Y, X)
            original = img.shape
            img_flat = img.reshape(-1, img.shape[2], img.shape[3])
            results = [model.predict_instances(slice)[0] for slice in img_flat]
            masks = np.array(results).reshape(original)

    # --- 6. SAVE ---
    print(f"WORKER: Saving output...")
    np.save(out_path, masks)
    print("WORKER: Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default='2D_versatile_fluo')
    parser.add_argument("--prob", type=float, default=0.5)
    parser.add_argument("--nms", type=float, default=0.3)
    parser.add_argument("--pmin", type=float, default=1.0)
    parser.add_argument("--pmax", type=float, default=99.8)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    
    run_inference(args.input, args.output, args.model, args.prob, args.nms, args.pmin, args.pmax, args.scale)