from .decorator import register_node
import dask.array as da
from pathlib import Path
import zarr
import napari

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



@register_node(
    label="Open Zarr",
    category="Input",
    outputs=["data"],
    params_config={
        # This tells your UI to use the folder picker we designed
        "path": {
            "type": "path", 
            "mode": "directory" 
        },
        # Allow user to name the layer in Napari
        "layer_name": {
            "type": "string",
            "default": "zarr_layer"
        }
    }
)
def open_zarr(path: str = "", layer_name: str = "zarr_layer"):
    """
    Loads a Zarr dataset lazily. 
    """
    if not path:
        return None
        
    z_group = zarr.open(str(path), mode='r')
        
    # 2. Create a list of Dask arrays for each level
    # (This assumes standard OME-Zarr structure where keys are numbers)
    pyramid = []
    for i in range(len(z_group)):
        try:
            d = da.from_zarr(str(path), component=str(i))
            pyramid.append(d)
        except:
            break
                
    return pyramid



