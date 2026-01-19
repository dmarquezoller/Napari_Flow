import sys
import os
import argparse
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

def run_ultrack(labels_path, out_dir, config_path=None):
    # Enable unbuffered output so you see the progress bars in Napari
    sys.stdout.reconfigure(line_buffering=True)
    print(f"WORKER: Starting Ultrack Replica (PID {os.getpid()})...")

    # 1. Imports
    try:
        import ultrack
        from ultrack.config import MainConfig, load_config
        # This function creates the colored segments layer from the tracks
        from ultrack.utils import labels_to_contours
    except ImportError:
        print("WORKER FATAL: 'ultrack' not installed.")
        sys.exit(1)

    # 2. Load Data
    try:
        labels = np.load(labels_path)
        print(f"WORKER: Loaded labels {labels.shape}")
    except Exception as e:
        print(f"WORKER FATAL: Load failed: {e}")
        sys.exit(1)

    # 3. Configure (Replicating Plugin Logic)
    # If you provide a config file (optional), we use it. 
    # Otherwise, we use the EXACT defaults the library provides.
    if config_path and os.path.exists(config_path):
        try:
            cfg = load_config(config_path)
            print(f"WORKER: Loaded custom config from {config_path}")
        except Exception as e:
             print(f"WORKER ERROR: Could not load config: {e}. Falling back to defaults.")
             cfg = MainConfig()
    else:
        print("WORKER: Using default configuration (Same as Plugin defaults)...")
        cfg = MainConfig()
        
    # Ensure multi-processing doesn't conflict with Napari
    cfg.linking_config.n_workers = 1 

    # 4. Run Tracking
    print("WORKER: Running track()... (This may take a while)")
    try:
        # This single call does EVERYTHING: Detection -> Linking -> Solving
        # It handles the "0 tracks" logic internally much better than we can
        detection, edges = labels_to_contours(labels, sigma=4.0)
        result = ultrack.track(
            detection=detection,
            edges=edges,
            config=cfg,
            overwrite=True
        )

        tracks_df, graph = result
        print(f"WORKER: Tracking finished. Found {len(tracks_df['track_id'].unique())} tracks.")

        # 5. Save Tracks
        tracks_path = os.path.join(out_dir, "tracks.csv")
        tracks_df.to_csv(tracks_path, index=False)

        print("WORKER: Done.")

    except Exception as e:
        print(f"WORKER CRASH: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config", default=None)
    
    args = parser.parse_args()
    
    run_ultrack(args.labels, args.out_dir, args.config)