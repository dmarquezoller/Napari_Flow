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
    


def _ensure_numpy(image):
    if hasattr(image, "compute"): return image.compute()
    return np.asarray(image)

def _ensure_dataframe(data):
    """
    Unwraps input to ensure we have a pandas DataFrame.
    Handles cases where data might be passed as (data, meta, type) tuple.
    """
    if isinstance(data, pd.DataFrame):
        return data
    
    # If it's a tuple (data, meta, type), grab the first element
    if isinstance(data, tuple) and len(data) > 0:
        if isinstance(data[0], pd.DataFrame):
            return data[0]
            
    # If it's a list (sometimes happens with multi-inputs), try the first item
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], pd.DataFrame):
            return data[0]

    raise ValueError(f"Input is not a valid DataFrame. Got: {type(data)}")



@register_node(
    label="Filter Labels by CSV",
    category="Measure",
    outputs=["filtered_labels"],
    params_config={
        "filters": {
            "type": "table",  
            "label": "Filter Rules",
            "columns": [
                {"name": "col", "label": "Filter Property", "type": "enum",
                 "options": ["area", "mean_intensity", "max_intensity", "min_intensity", "solidity", "eccentricity", "circularity", "perimeter", "major_axis_length", "minor_axis_length", "label"]},
                {"name": "op", "label": "Operation", "type": "enum",
                 "options": [">", "<", ">=", "<=", "==", "!="]},
                {"name": "val", "label": "Threshold", "type": "float", "value": 100.00}
            ],
            "value": [{"col": "area", "op": ">", "val": 100}]
        },
        "frame_col": {"type": "text", "value": "frame"},
        "id_col": {"type": "text", "value": "label"}
    }
)
def filter_labels_by_csv(labels_layer, csv_data, filters=[{"col": "area", "op": ">", "val": 100}], frame_column="frame", id_column="label"):
    
    print(f"--- Filtering Labels (Multi-Rule) ---")
    
    # 1. Load Inputs
    labels = _ensure_numpy(labels_layer)
    try:
        df = _ensure_dataframe(csv_data)
    except Exception as e:
        raise ValueError(f"Invalid CSV input. Please connect a 'Load CSV' node. ({e})")

    # 2. Validation
    if id_column not in df.columns:
        raise ValueError(f"ID Column '{id_column}' not found in CSV.")

    # 3. Determine Mode
    is_time_series = False
    if frame_column in df.columns and labels.ndim >= 3:
        is_time_series = True
        print(f"   > Mode: Time-Series (Found '{frame_column}')")
    else:
        print(f"   > Mode: Global")

    # 4. Build Query from Table (THE NEW PART)
    # We convert the list of dicts into a string like: "area > 100 and solidity < 0.9"
    try:
        query_parts = []
        for row in filters:
            col = row['col']
            op = row['op']
            val = row['val']
            
            # Basic validation
            if col not in df.columns:
                print(f"   ⚠️ Warning: Column '{col}' not found. Skipping rule.")
                continue
                
            query_parts.append(f"{col} {op} {val}")
        
        if not query_parts:
            print("   > No valid filters found. Returning original.")
            filtered_df = df
        else:
            full_query_string = " and ".join(query_parts)
            print(f"   > Applying Query: {full_query_string}")
            filtered_df = df.query(full_query_string)

    except Exception as e:
        raise ValueError(f"Filter Logic Error: {e}")

    print(f"   > Result: {len(df)} -> {len(filtered_df)} objects kept.")

    # 5. Apply to Image (Standard Frame-Aware Logic)
    output_labels = np.zeros_like(labels)

    if is_time_series:
        n_frames = labels.shape[0]
        # Optimization: Group dataframe by frame once
        grouped = filtered_df.groupby(frame_column)[id_column].apply(set).to_dict()
        
        for t in range(n_frames):
            valid_ids = grouped.get(t, set())
            if not valid_ids: continue
            
            frame_slice = labels[t]
            # Fast numpy mask
            mask = np.isin(frame_slice, list(valid_ids))
            output_labels[t] = np.where(mask, frame_slice, 0)
    else:
        valid_ids = filtered_df[id_column].unique()
        if len(valid_ids) > 0:
            mask = np.isin(labels, valid_ids)
            output_labels = np.where(mask, labels, 0).astype(labels.dtype)

    return (output_labels, {"name": "Filtered Labels"}, "labels")