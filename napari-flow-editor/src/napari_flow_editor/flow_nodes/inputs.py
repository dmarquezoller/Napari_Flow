from .decorator import register_node
import dask.array as da
import pandas as pd
from pathlib import Path
import zarr
import napari
from ome_zarr.reader import Reader
from ome_zarr.io import parse_url
from napari.plugins.io import read_data_with_plugins
import os

@register_node(
    label="Get Layer",
    category="Input",
    outputs=["data_out"],
    # We define it as an enum so the JSON knows it's a dropdown.
    # We leave options empty [] because the GUI fills them in real-time.
    params_config={
        "layer_name": {"type": "enum", "options": []},
        "axis_map": {"type": "table", "label": "Dimensions (e.g. T, Z, Z, Y, X)",
                     "max_rows": 1,
                     "row_unique": True,
                     "allow_duplicates": ["-"],
                     "columns": [
                         {"name": "d0", "label": "Dim 0", "type": "enum", "options": ["-","T","Z","C","Y","X"]},
                         {"name": "d1", "label": "Dim 1", "type": "enum", "options": ["-","T","Z","C","Y","X"]},
                         {"name": "d2", "label": "Dim 2", "type": "enum", "options": ["-","T","Z","C","Y","X"]},
                         {"name": "d3", "label": "Dim 3", "type": "enum", "options": ["-","T","Z","C","Y","X"]},
                         {"name": "d4", "label": "Dim 4", "type": "enum", "options": ["-","T","Z","C","Y","X"]},
                     ],
                     "value": [{"d0": "Y", "d1": "X", "d2": "-", "d3": "-", "d4": "-"}]
                    }
    }
)
def get_layer(layer_name: str = "", axis_map: list = [{"d0": "Y", "d1": "X", "d2": "-", "d3": "-", "d4": "-"}]):
    # The argument 'layer_name' creates the parameter entry in the JSON.
    return layer_name




@register_node(
    label="Open Ome-Zarr",
    category="Input",
    outputs=["data"],
    params_config={
        "path": {"type": "path", "mode": "directory"},
    }
)
def open_ome_zarr(path: str = ""):
    import os
    if not path or not os.path.exists(path):
        raise ValueError("Path does not exist")

    # 1) Ask napari-ome-zarr for a reader that matches this path
    try:
        from napari_ome_zarr import napari_get_reader
    except Exception as e:
        raise RuntimeError(
            "napari-ome-zarr is not installed in this environment."
        ) from e

    reader = napari_get_reader(path)
    if reader is None:
        raise ValueError("napari-ome-zarr did not return a reader for this path.")

    # 2) Call the reader: this is the plugin’s real output contract
    layers = reader(path)

    if not layers:
        raise ValueError("Reader returned no layers")

    # At this point, layers should be a list of (data, meta, layer_type)
    # Return exactly that; your UI can iterate and add each.
    return layers




# LOAD CSV NODE #

@register_node(
    label="Load CSV",
    category="Input",
    outputs=["csv_data"],
    params_config={
        "file_path": {
            "type": "path", 
            "label": "CSV File", 
            "mode": "file", 
            "filter": "*.csv"
        }
    }
)
def load_csv(file_path=""):
    """
    Loads a CSV file and passes it to the next node.
    """
    if not file_path:
        raise ValueError("Please select a CSV file.")
    
    try:
        df = pd.read_csv(file_path)
        print(f"--- Load CSV ---")
        print(f"   > Loaded {len(df)} rows from {file_path.split('/')[-1]}")
        
        # We return the dataframe packaged as a "dataframe" type.
        # Even if Napari doesn't visualize 'dataframe' layers, the flow editor 
        # passes this object to the next node.
        return (df, {"name": "Loaded Data"}, "dataframe")
        
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")
    


@register_node(
    label="Select Layer",
    category="Input",
    outputs=["image_out"],
    params_config={
        # Text input (your UI will render a normal text field if type is unknown)
        "layer_name": {"type": "text", "label": "Layer name (exact preferred)"},
        # Optional safety: keep this to disambiguate image vs labels when names collide
        "layer_type": {"options": ["image", "labels"]},
    }
)
def select_layer(layers, layer_name: str = "", layer_type: str = "image"):
    """
    Select a layer by *name* from a list of napari LayerDataTuples:
      layers: [(data, meta, layer_type), ...]

    Returns (data, meta) so downstream nodes receive just the data (engine will unpack),
    while metadata stays available.
    """
    import dask.array as da

    def is_ldt(x):
        return isinstance(x, tuple) and len(x) == 3 and isinstance(x[1], dict)

    if not isinstance(layers, list) or len(layers) == 0:
        raise ValueError("Select Layer: input is empty or not a list of layers.")

    # Build searchable list
    available = []
    for item in layers:
        if not is_ldt(item):
            continue
        data, meta, lt = item
        name = str(meta.get("name", ""))
        available.append({"name": name, "type": str(lt), "data": data, "meta": meta})

    if not available:
        raise ValueError("Select Layer: no valid LayerDataTuples found in input.")

    # If no name provided, print available and fail (for testing)
    if not layer_name.strip():
        print("\n🧩 SELECT_LAYER (by name) — available layers:")
        for i, a in enumerate(available):
            print(f"  [{i}] name={a['name']!r} type={a['type']}")
        raise ValueError("Select Layer: please provide a layer_name.")

    target = layer_name.strip()

    # Match priority: exact -> case-insensitive exact -> contains (case-insensitive)
    matches = [a for a in available if a["type"] == layer_type and a["name"] == target]
    if not matches:
        matches = [a for a in available if a["type"] == layer_type and a["name"].lower() == target.lower()]
    if not matches:
        matches = [a for a in available if a["type"] == layer_type and target.lower() in a["name"].lower()]

    # If still no match with requested type, fallback: try name match across any type
    if not matches:
        matches = [a for a in available if a["name"] == target]
    if not matches:
        matches = [a for a in available if a["name"].lower() == target.lower()]
    if not matches:
        matches = [a for a in available if target.lower() in a["name"].lower()]

    if not matches:
        print("\n🧩 SELECT_LAYER (by name) — no match for:", target)
        print("Available layers:")
        for i, a in enumerate(available):
            print(f"  [{i}] name={a['name']!r} type={a['type']}")
        raise ValueError(f"Select Layer: no layer matched name={target!r}")

    if len(matches) > 1:
        print(f"⚠️ Select Layer: multiple matches for {target!r} (type={layer_type}). Using the first.")
        for i, m in enumerate(matches[:10]):
            print(f"  match[{i}] name={m['name']!r} type={m['type']}")

    chosen = matches[0]
    data = chosen["data"]
    meta = chosen["meta"]

    # --- TEST PRINT: what this node will output ---
    print("\n✅ SELECT_LAYER OUTPUT")
    print("  selected name:", chosen["name"])
    print("  selected type:", chosen["type"])
    print("  data type:", type(data))
    print("  is multiscale list:", isinstance(data, list))
    if isinstance(data, list) and len(data) > 0:
        print("  levels:", len(data))
        print("  level0 type:", type(data[0]))
        print("  level0 is dask:", isinstance(data[0], da.Array))
    else:
        print("  is dask:", isinstance(data, da.Array))
    print("  meta keys (sample):", list(meta.keys())[:15])

    return (data, meta)

