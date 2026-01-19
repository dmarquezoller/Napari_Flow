from .decorator import register_node
from .deep_learning import _ensure_numpy
import numpy as np
import pandas as pd
import skimage.measure
from tqdm import tqdm  

@register_node(
    label="Region Properties",
    category="Measure",
    outputs=["measurements_table"],
    params_config={
        "properties": {
            "options": ["Basic (Area, Perim)", "Shape (Eccentricity, Solidity)", "All"],
            "value": "All",
            "label": "Metrics to Calculate"
        },
        "mode": {
            "options": ["2D Slice-by-Slice (Time/Z)", "3D Volumetric"],
            "value": "2D Slice-by-Slice (Time/Z)",
            "label": "Analysis Mode"
        }
    }
)
def measure_labels(labels, image, properties: str = "All", mode: str = "2D Slice-by-Slice (Time/Z)"):
    """
    Calculates properties for labeled regions.
    Supports processing stack slice-by-slice to prevent RAM crashes.
    """
    labels = _ensure_numpy(labels)
    
    if labels is None:
        return None
    
    # --- 1. PREPARE IMAGE ---
    intensity_img = None
    if image is not None:
        intensity_img = _ensure_numpy(image)
        # Handle RGB/Multi-channel intensity (Take Mean)
        # If Image has 1 more dim than Labels, it's likely channels
        if intensity_img.ndim == labels.ndim + 1:
            intensity_img = np.mean(intensity_img, axis=-1)
    
    print(f"--- Measuring Objects (Mode: {mode}) ---")

    # --- 2. DEFINE PROPERTIES ---
    props_list = ['label', 'area', 'perimeter', 'centroid']
    if properties in ["Shape (Eccentricity, Solidity)", "All"]:
        props_list += ['eccentricity', 'solidity', 'axis_major_length', 'axis_minor_length', 'orientation']
    if image is not None:
        props_list += ['mean_intensity', 'max_intensity', 'min_intensity']

    # --- 3. EXECUTE MEASUREMENT ---
    dfs = []
    
    try:
        # MODE A: PROCESS STACK FRAME-BY-FRAME (Prevents Memory Crash)
        # Use this for Time-Lapse or large Z-Stacks where you analyze 2D planes
        if mode == "2D Slice-by-Slice (Time/Z)" and labels.ndim == 3:
            total_frames = labels.shape[0]
            print(f"  > Processing {total_frames} frames individually...")
            
            for t in range(total_frames):
                lab_slice = labels[t]
                
                # Skip empty frames to save speed
                if np.max(lab_slice) == 0:
                    continue
                
                img_slice = intensity_img[t] if intensity_img is not None else None
                
                # Measure single frame
                data = skimage.measure.regionprops_table(
                    lab_slice, 
                    intensity_image=img_slice, 
                    properties=props_list
                )
                
                df_slice = pd.DataFrame(data)
                
                # Add a Time Column so we know which frame these cells belong to
                df_slice["Frame"] = t
                
                # Rename Centroids 2D
                if 'centroid-0' in df_slice.columns:
                    df_slice.rename(columns={'centroid-0': 'Y', 'centroid-1': 'X'}, inplace=True)
                
                dfs.append(df_slice)

        # MODE B: SINGLE 2D IMAGE OR TRUE 3D VOLUMETRIC
        else:
            print("  > Processing as single volume...")
            data = skimage.measure.regionprops_table(
                labels, 
                intensity_image=intensity_img, 
                properties=props_list
            )
            df_frame = pd.DataFrame(data)
            
            # Handle 3D Centroids
            if labels.ndim == 3 and 'centroid-0' in df_frame.columns:
                df_frame.rename(columns={'centroid-0': 'Z', 'centroid-1': 'Y', 'centroid-2': 'X'}, inplace=True)
            elif 'centroid-0' in df_frame.columns:
                df_frame.rename(columns={'centroid-0': 'Y', 'centroid-1': 'X'}, inplace=True)
                
            dfs.append(df_frame)

        # --- 4. COMBINE RESULTS ---
        if not dfs:
            print("  > No objects found in any frame.")
            return pd.DataFrame()
            
        final_df = pd.concat(dfs, ignore_index=True)
        
        # Reorder columns to put Frame/Label first
        cols = final_df.columns.tolist()
        if "Frame" in cols:
            cols.insert(0, cols.pop(cols.index("Frame")))
        if "label" in cols:
            cols.insert(1, cols.pop(cols.index("label")))
        final_df = final_df[cols]

        print(f"  > Success! Calculated {final_df.shape[1]} metrics for {final_df.shape[0]} total objects.")
        return final_df

    except Exception as e:
        print(f"Measurement Failed: {e}")
        # Hint for OOM
        if "MemoryError" in str(e) or "killed" in str(e).lower():
            print("  HINT: Try reducing 'Metrics to Calculate' or ensure 'Slice-by-Slice' mode is on.")
        raise e