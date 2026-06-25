import os
from .decorator import register_node
from skimage.io import imsave
from .deep_learning import _ensure_numpy


@register_node(
    label="Save Table (CSV)",
    category="Outputs",
    outputs=[],
    input_types={"table": "table"},
    params_config={
        "filename": {"type": "text", "value": "results.csv", "label": "Filename"},
        "folder": {"type": "path", "mode": "directory", "label": "Save Folder"}
    }
)
def save_table(table, folder: str = "", filename: str = "results.csv"):
    import numpy as np
    import pandas as pd

    if table is None:
        print("Save Table: No data received.")
        return

    if not folder or not os.path.isdir(folder):
        raise ValueError("Please select a valid folder.")

    full_path = os.path.join(folder, filename)
    if not full_path.endswith(".csv"):
        full_path += ".csv"

    if isinstance(table, np.ndarray):
        if table.ndim == 2:
            ncols = table.shape[1]
            if ncols == 4:
                cols = ["track_id", "t", "y", "x"]
            elif ncols == 5:
                cols = ["track_id", "t", "z", "y", "x"]
            else:
                cols = [f"col_{i}" for i in range(ncols)]
            table = pd.DataFrame(table, columns=cols)
        else:
            table = pd.DataFrame(table)

    table.to_csv(full_path, index=False)
    print(f"Saved table to: {full_path}")



@register_node(
    label="Save Image",
    category="Outputs",
    outputs=[],  # No outputs, it's a sink
    input_types={"image": "any"},
    params_config={
        "folder": {"type": "path", "mode": "directory", "label": "Save Folder"},
        "base_name": {"type": "text", "label": "Filename Prefix", "value": "crop_output"},
        "format": {"type": "enum", "options": [".tif", ".png", ".jpg"], "label": "Format"}
    }
)
def save_image_node(image, folder="", base_name="output", format=".tif"):
    if image is None: return
    if not folder or not os.path.isdir(folder):
        print(">> Save Node: Invalid folder.")
        return

    # 1. Smart Naming: Find the next available index
    # (So we get crop_output_001.tif, crop_output_002.tif, etc.)
    idx = 1
    while True:
        filename = f"{base_name}_{idx:03d}{format}"
        full_path = os.path.join(folder, filename)
        if not os.path.exists(full_path):
            break
        idx += 1
    
    # 2. Save
    # Ensure it's on CPU and proper format
    image = _ensure_numpy(image)
    
    # Simple normalization for PNG/JPG if needed, but TIF handles floats well
    try:
        imsave(full_path, image, check_contrast=False)
        print(f">> Saved: {filename}")
    except Exception as e:
        print(f"!! Save Failed: {e}")

    return None


@register_node(
    label="Export Ultrack Tracks",
    category="Outputs",
    outputs=[],
    input_types={"tracks": "tracks"},
    params_config={
        "folder":   {"type": "path", "mode": "directory", "label": "Save Folder"},
        "filename": {"type": "text", "value": "tracks",   "label": "Filename (no extension)"},
        "format":   {"type": "enum", "options": ["csv", "parquet"], "label": "Format"},
    }
)
def export_ultrack_tracks(tracks, folder: str = "", filename: str = "tracks", format: str = "csv"):
    import numpy as np
    import pandas as pd

    if tracks is None:
        print("Export Ultrack Tracks: No data received.")
        return
    if not folder or not os.path.isdir(folder):
        raise ValueError("Please select a valid folder.")

    if isinstance(tracks, np.ndarray):
        if tracks.ndim == 2:
            ncols = tracks.shape[1]
            if ncols == 5:
                cols = ["track_id", "t", "z", "y", "x"]
            else:
                cols = ["track_id", "t", "y", "x"]
            tracks = pd.DataFrame(tracks, columns=cols[:ncols])
        else:
            tracks = pd.DataFrame(tracks)

    base = filename.rsplit(".", 1)[0] if "." in filename else filename
    if format == "parquet":
        full_path = os.path.join(folder, base + ".parquet")
        tracks.to_parquet(full_path, index=False)
    else:
        full_path = os.path.join(folder, base + ".csv")
        tracks.to_csv(full_path, index=False)

    print(f"Tracks saved to: {full_path}")
    print(f"  Rows: {len(tracks)}  |  Columns: {list(tracks.columns)}")