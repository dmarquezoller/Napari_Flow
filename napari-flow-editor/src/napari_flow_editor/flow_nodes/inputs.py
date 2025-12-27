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
        
    try:
        print(f"📂 [Open Zarr] Pointing to: {path}")
        
        # Load Metadata Only
        lazy_image = da.from_zarr(str(path), component='0')
        
        print(f"✅ [Open Zarr] Success! Shape: {lazy_image.shape}")
        return lazy_image

    except Exception as e:
        print(f"❌ [Open Zarr] Error: {e}")
        return None


