from .decorator import register_node
import dask.array as da
from pathlib import Path
import zarr
import napari
from ome_zarr.reader import Reader
from ome_zarr.io import parse_url
from napari.plugins.io import read_data_with_plugins

@register_node(
    label="Get Layer",
    category="Input",
    outputs=["data_out"],
    # We define it as an enum so the JSON knows it's a dropdown.
    # We leave options empty [] because the GUI fills them in real-time.
    params_config={
        "layer_name": {"type": "enum", "options": []} 
    }
)
def get_layer(layer_name: str = ""):
    # The argument 'layer_name' creates the parameter entry in the JSON.
    return layer_name


# flow_nodes/inputs.py
from .decorator import register_node
import os
# Import the internal Napari reader function
from napari.plugins.io import read_data_with_plugins

# flow_nodes/inputs.py
from .decorator import register_node
import os
from napari.plugins.io import read_data_with_plugins

@register_node(
    label="Open Ome-Zarr",
    category="Input",
    outputs=["data"],
    params_config={
        "path": {"type": "path", "mode": "directory"},
    }
)
def open_ome_zarr(path: str = ""):
    """
    Uses the official napari-ome-zarr plugin logic to read the file.
    Returns a LIST of LayerDataTuples: [(data, meta, layer_type), ...]
    """
    if not path or not os.path.exists(path):
        return None

    print(f"🔌 Invoking native napari-ome-zarr plugin for: {path}")

    # 1. READ: We use the internal function that 'viewer.open()' uses.
    # CRITICAL: We pass [path] as a list, otherwise it crashes.
    try:
        layers = read_data_with_plugins([path], plugin="napari-ome-zarr")
    except Exception as e:
        print(f"Plugin Read Error: {e}")
        return None

    if not layers:
        raise ValueError("The plugin could not read this file.")

    # 2. RETURN: We pass the exact list of layers (images, labels, etc.) 
    # to the Main Thread.
    return layers