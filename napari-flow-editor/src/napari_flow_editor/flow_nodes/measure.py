from .decorator import register_node
from .deep_learning import _ensure_numpy
import numpy as np
import pandas as pd
import skimage.measure

def _build_props(ndim_obj, properties, has_image):
    """Pick regionprops_table properties valid for the measured array's ndim.
    perimeter / eccentricity / axis lengths / orientation are 2D-only in skimage."""
    if ndim_obj == 3:
        props = ['label', 'area', 'centroid']
        if properties in ("Shape (Eccentricity, Solidity)", "All"):
            props.append('solidity')
    else:
        props = ['label', 'area', 'perimeter', 'centroid']
        if properties in ("Shape (Eccentricity, Solidity)", "All"):
            props += ['eccentricity', 'solidity', 'axis_major_length',
                      'axis_minor_length', 'orientation']
    if has_image:
        props += ['mean_intensity', 'max_intensity', 'min_intensity']
    return props


def _rename_centroids(df):
    """Rename centroid-N columns to spatial axis names based on dimensionality."""
    if 'centroid-2' in df.columns:
        return df.rename(columns={'centroid-0': 'Z', 'centroid-1': 'Y', 'centroid-2': 'X'})
    if 'centroid-1' in df.columns:
        return df.rename(columns={'centroid-0': 'Y', 'centroid-1': 'X'})
    return df


def _measure_array(lab, img, properties, has_image):
    """Measure a single 2D or 3D label array, returning a DataFrame."""
    props = _build_props(lab.ndim, properties, has_image)
    data = skimage.measure.regionprops_table(lab, intensity_image=img, properties=props)
    return _rename_centroids(pd.DataFrame(data))


@register_node(
    label="Region Properties",
    category="Measure",
    outputs=["measurements_table"],
    input_types={"image": "image", "labels": "labels"},
    output_types={"measurements_table": "table"},
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
    """Region properties table. One row per (frame, label) with a string cell_uid.

    Slice-by-Slice iterates the leading axis (time / Z planes); 3D Volumetric
    measures the whole volume as one. Property set adapts to the measured array's
    dimensionality so 3D measurements never request 2D-only metrics.
    """
    labels = _ensure_numpy(labels)
    if labels is None:
        return None
    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int32)

    has_image = image is not None
    intensity = None
    if has_image:
        intensity = _ensure_numpy(image)
        if intensity.ndim == labels.ndim + 1:   # trailing channel axis -> collapse
            intensity = intensity.mean(axis=-1)

    slice_mode = (mode == "2D Slice-by-Slice (Time/Z)")
    # Iterate the leading axis for 4D (per-timepoint volume) or 3D time-stacks.
    iterate = (labels.ndim == 4) or (labels.ndim == 3 and slice_mode)

    dfs = []
    if iterate:
        print(f"--- Region Properties: {labels.shape[0]} frames (mode={mode}) ---")
        for t in range(labels.shape[0]):
            lab = labels[t]
            if lab.max() == 0:                   # skip empty frames
                continue
            img_t = intensity[t] if intensity is not None else None
            df = _measure_array(lab, img_t, properties, has_image)
            df.insert(0, 'frame', t)
            df.insert(0, 'cell_uid', [f"t{t}_l{int(v)}" for v in df['label']])
            dfs.append(df)
    else:
        print(f"--- Region Properties: single volume (mode={mode}) ---")
        df = _measure_array(labels, intensity, properties, has_image)
        df.insert(0, 'cell_uid', [f"l{int(v)}" for v in df['label']])
        dfs.append(df)

    if not dfs:
        print("  > No objects found.")
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    lead = [c for c in ('cell_uid', 'frame', 'label') if c in result.columns]
    result = result[lead + [c for c in result.columns if c not in lead]]
    print(f"  > {len(result)} objects, {result.shape[1]} columns.")
    return result



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
            "max_rows": 2,
            "columns": [
                {"name": "col", "label": "Filter Property", "type": "enum",
                 "options": ["area", "mean_intensity", "max_intensity", "min_intensity", "solidity", "eccentricity", "circularity", "perimeter", "major_axis_length", "minor_axis_length", "label"]},
                {"name": "op", "label": "Operation", "type": "enum",
                 "options": [">", "<", ">=", "<=", "==", "!="]},
                {"name": "val", "label": "Threshold", "type": "float", "value": 100.00}
            ],
            "value": [{"col": "area", "op": ">", "val": 100}]
        },
        "frame_column": {"type": "text", "value": "Frame"},
        "id_column":    {"type": "text", "value": "label"}
    }
)
def filter_labels_by_csv(labels_layer, csv_data, filters=[{"col": "area", "op": ">", "val": 100}], frame_column="Frame", id_column="label"):
    
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

    # 3. Determine Mode — case-insensitive column match for robustness
    is_time_series = False
    frame_col_actual = next(
        (c for c in df.columns if c.lower() == frame_column.lower()), None
    )
    if frame_col_actual and labels.ndim >= 3:
        is_time_series = True
        frame_column = frame_col_actual  # use the exact casing from the CSV
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


@register_node(
    label="Track Statistics",
    category="Measure",
    description="Computes per-track statistics from a tracks layer (track_id, t, y, x). Outputs an enriched table with length, duration, displacement, speed, and division info. Attach a Save Table node to write it to disk.",
    outputs=["stats_table"],
    input_types={"tracks": "tracks"},
    output_types={"stats_table": "table"},
)
def track_statistics(tracks):
    print("--- Track Statistics ---")

    if tracks is None:
        raise ValueError("No tracks data received.")

    if isinstance(tracks, np.ndarray):
        ncols = tracks.shape[1] if tracks.ndim == 2 else 4
        cols = (["track_id", "t", "z", "y", "x"] if ncols == 5
                else ["track_id", "t", "y", "x"])[:ncols]
        df = pd.DataFrame(tracks, columns=cols)
    else:
        df = pd.DataFrame(tracks)

    required = {"track_id", "t", "y", "x"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tracks data missing columns: {missing}")

    has_z = "z" in df.columns

    def _track_stats(g):
        g = g.sort_values("t")
        length   = len(g)
        duration = int(g["t"].max() - g["t"].min())

        if has_z:
            coords = g[["z", "y", "x"]].to_numpy().astype(float)
        else:
            coords = g[["y", "x"]].to_numpy().astype(float)

        diffs              = np.diff(coords, axis=0)
        step_distances     = np.linalg.norm(diffs, axis=1)
        total_displacement = float(step_distances.sum())
        net_displacement   = float(np.linalg.norm(coords[-1] - coords[0]))
        mean_speed         = total_displacement / duration if duration > 0 else 0.0

        return pd.Series({
            "length":             length,
            "duration":           duration,
            "total_displacement": round(total_displacement, 3),
            "net_displacement":   round(net_displacement, 3),
            "mean_speed":         round(mean_speed, 3),
            "t_start":            int(g["t"].min()),
            "t_end":              int(g["t"].max()),
        })

    stats = df.groupby("track_id").apply(_track_stats).reset_index()

    if "parent_track_id" in df.columns:
        divisions = (
            df[df["parent_track_id"].notna() & (df["parent_track_id"] != 0)]
            [["track_id", "parent_track_id"]]
            .drop_duplicates()
        )
        stats = stats.merge(divisions, on="track_id", how="left")
        stats["has_division"] = stats["parent_track_id"].notna()
    else:
        stats["has_division"] = False

    # Merge per-track stats back onto per-detection rows, preserving original structure.
    # Drop parent_track_id from stats since df already has it.
    stats_to_merge = stats.drop(columns=["parent_track_id"], errors="ignore")
    enriched = df.merge(stats_to_merge, on="track_id", how="left")

    print(f"  Tracks: {len(stats)}  |  Rows: {len(enriched)}  |  Columns: {list(enriched.columns)}")

    return (enriched, {"layer_type": "table"})


def _load_table(data):
    """Coerce a node input into a DataFrame.

    Accepts a DataFrame, a tuple/list (first element, recursively), or a string
    (an existing file path -> read_csv; otherwise treated as raw CSV text).
    """
    import io
    import os

    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, (tuple, list)) and len(data) > 0:
        return _load_table(data[0])
    if isinstance(data, str):
        if os.path.exists(data):
            return pd.read_csv(data)
        return pd.read_csv(io.StringIO(data))
    raise ValueError(f"Could not interpret input as a table. Got: {type(data)}")


@register_node(
    label="Join Tables",
    category="Measure",
    description="Spatially join two tables. 'exact' columns must match (they block the search); "
                "'approximate' columns are matched together by nearest-neighbor within each block, "
                "rejecting matches beyond Max Distance.",
    outputs=["joined_table"],
    input_types={"csv1": "table", "csv2": "table"},
    output_types={"joined_table": "table"},
    params_config={
        "rules": {
            "type": "table",
            "label": "Join Keys",
            "columns": [
                {"name": "c1",     "label": "CSV 1 Column", "type": "text"},
                {"name": "c2",     "label": "CSV 2 Column", "type": "text"},
                {"name": "method", "label": "Match",        "type": "enum",
                 "options": ["exact", "approximate"]},
            ],
            "value": [
                {"c1": "t", "c2": "frame", "method": "exact"},
                {"c1": "y", "c2": "Y",     "method": "approximate"},
                {"c1": "x", "c2": "X",     "method": "approximate"},
            ],
        },
        "max_distance": {"type": "float", "min": 0.0, "max": 1e6, "step": 1.0,
                         "value": 5.0, "label": "Max Distance (0 = no limit)"},
        "base": {"type": "enum",
                 "options": ["CSV 1 (keep all)", "CSV 2 (keep all)", "Only matched (inner)"],
                 "value": "CSV 1 (keep all)", "label": "Base Table"},
    }
)
def join_tables(csv1, csv2,
                rules=[{"c1": "t", "c2": "frame", "method": "exact"},
                       {"c1": "y", "c2": "Y", "method": "approximate"},
                       {"c1": "x", "c2": "X", "method": "approximate"}],
                max_distance: float = 5.0,
                base: str = "CSV 1 (keep all)"):
    from scipy.spatial import cKDTree

    print("--- Join Tables ---")
    df1 = _load_table(csv1)
    df2 = _load_table(csv2)

    # 1. Parse rules into (base_col, other_col) pairs depending on which side is the base.
    keep_csv2 = base.startswith("CSV 2")
    inner = base.startswith("Only matched")
    base_df, other_df = (df2, df1) if keep_csv2 else (df1, df2)

    exact, approx = [], []
    for row in rules:
        c1, c2, method = row.get("c1"), row.get("c2"), row.get("method", "exact")
        if not c1 or not c2:
            continue
        # base_col / other_col follow whichever table is the base
        base_col, other_col = (c2, c1) if keep_csv2 else (c1, c2)
        (exact if method == "exact" else approx).append((base_col, other_col))

    if not exact and not approx:
        raise ValueError("No valid join rules provided.")

    # 2. Validate columns exist.
    for bc, oc in exact + approx:
        if bc not in base_df.columns:
            raise ValueError(f"Column '{bc}' not in base table. Available: {list(base_df.columns)}")
        if oc not in other_df.columns:
            raise ValueError(f"Column '{oc}' not in other table. Available: {list(other_df.columns)}")

    base_exact = [bc for bc, _ in exact]
    other_exact = [oc for _, oc in exact]
    base_approx = [bc for bc, _ in approx]
    other_approx = [oc for _, oc in approx]

    # 3. Block 'other' by its exact-key columns; build a KDTree per block over approx coords.
    other_reset = other_df.reset_index(drop=True)
    if other_exact:
        groups = other_reset.groupby(other_exact, sort=False)
        block_items = groups
    else:
        block_items = [((), other_reset)]   # single global block

    blocks = {}
    for key, sub in block_items:
        key_t = key if isinstance(key, tuple) else (key,)
        idx = sub.index.to_numpy()
        tree = cKDTree(sub[other_approx].to_numpy(dtype=float)) if other_approx else None
        blocks[key_t] = (idx, tree)

    # 4. Match each base row to its nearest 'other' row within the same exact block.
    base_reset = base_df.reset_index(drop=True)
    n = len(base_reset)
    matched_other_idx = np.full(n, -1, dtype=np.int64)
    match_dist = np.full(n, np.nan, dtype=float)

    base_exact_vals = base_reset[base_exact].to_numpy() if base_exact else None
    base_approx_vals = base_reset[base_approx].to_numpy(dtype=float) if base_approx else None

    for i in range(n):
        key_t = tuple(base_exact_vals[i]) if base_exact else ()
        block = blocks.get(key_t)
        if block is None:
            continue
        idx, tree = block
        if tree is not None:
            dist, pos = tree.query(base_approx_vals[i])
            if max_distance > 0 and dist > max_distance:
                continue
            matched_other_idx[i] = idx[pos]
            match_dist[i] = dist
        else:
            matched_other_idx[i] = idx[0]   # exact-only: take first in block
            match_dist[i] = 0.0

    # 5. Assemble output: base columns + matched other columns.
    # Drop other's join-key columns (redundant with base keys); suffix remaining collisions.
    join_key_cols = set(other_exact) | set(other_approx)
    attach_cols = [c for c in other_reset.columns if c not in join_key_cols]

    has_match = matched_other_idx >= 0
    attached = other_reset.loc[matched_other_idx.clip(min=0), attach_cols].reset_index(drop=True)
    attached[~has_match] = np.nan   # null out rows with no match

    rename = {c: (f"{c}_2" if c in base_reset.columns else c) for c in attach_cols}
    attached = attached.rename(columns=rename)

    joined = pd.concat([base_reset, attached], axis=1)
    joined["match_distance"] = match_dist

    if inner:
        joined = joined[has_match].reset_index(drop=True)

    n_matched = int(has_match.sum())
    finite = match_dist[np.isfinite(match_dist)]
    mean_d = float(np.mean(finite)) if finite.size else float("nan")
    med_d = float(np.median(finite)) if finite.size else float("nan")
    print(f"  Matched {n_matched}/{n} base rows | mean dist={mean_d:.3f} median={med_d:.3f}")
    print(f"  Output: {len(joined)} rows, {joined.shape[1]} columns")

    return (joined, {"layer_type": "table"})