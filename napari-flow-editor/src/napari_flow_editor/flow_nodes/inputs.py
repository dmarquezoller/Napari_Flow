from .decorator import register_node
import numpy as np

# We use a special flag or just handle it in the execution engine
# But for now, we'll wrap a helper that we assume gets injected with the image
@register_node(
    label="Get Active Layer",
    category="Input",
    outputs=["image"],
    params_config={} 
)
def get_active_layer(image_from_viewer):
    # This function is a placeholder. 
    # The Execution Engine will inject the actual Napari layer data here.
    return image_from_viewer