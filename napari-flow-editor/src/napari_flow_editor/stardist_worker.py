# src/napari_flow_editor/stardist_worker.py
import sys
import os
import argparse
import numpy as np
import time

# 1. LOCK THREADS (Crucial for the worker process too)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def run_inference(img_path, out_path, model_type, prob_thresh, nms_thresh, pmin, pmax):
    print(f"WORKER: Starting StarDist on {os.getpid()}...")
    
    # Load Image
    try:
        img = np.load(img_path)
        print(f"WORKER: Loaded image {img.shape}")
    except Exception as e:
        print(f"WORKER ERROR: Could not load image: {e}")
        sys.exit(1)

    # Import Libraries (Only happens in this isolated process)
    try:
        from stardist.models import StarDist2D
        from csbdeep.utils import normalize
    except ImportError:
        print("WORKER ERROR: StarDist/CSBDeep not installed.")
        sys.exit(1)

    # Normalize
    print(f"WORKER: Normalizing ({pmin}-{pmax})...")
    axis = (0,1) if img.ndim == 2 else (1,2)
    img = normalize(img, pmin=pmin, pmax=pmax, axis=axis)

    # Load Model
    print(f"WORKER: Loading model '{model_type}'...")
    # Suppress GPU for stability
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    try:
        model = StarDist2D.from_pretrained(model_type)
    except Exception as e:
        print(f"WORKER ERROR: Model load failed: {e}")
        sys.exit(1)

    # Predict
    print("WORKER: Running prediction...")
    masks = None
    if img.ndim == 3:
        results = []
        for i in range(img.shape[0]):
            res, _ = model.predict_instances(img[i], prob_thresh=prob_thresh, nms_thresh=nms_thresh)
            results.append(res)
        masks = np.array(results)
    else:
        masks, _ = model.predict_instances(img, prob_thresh=prob_thresh, nms_thresh=nms_thresh)

    # Save Result
    print(f"WORKER: Saving mask to {out_path}...")
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
    
    args = parser.parse_args()
    
    run_inference(
        args.input, args.output, args.model, 
        args.prob, args.nms, args.pmin, args.pmax
    )