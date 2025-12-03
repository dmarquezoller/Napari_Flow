from .decorator import register_node

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