from .decorator import register_node
import skimage.transform

# --- ROTATE IMAGE ---
@register_node(
    label="Rotate",
    category="Transform",
    outputs=["rotated"],
    params_config={
        "angle": {"min": -180, "max": 180, "step": 1},
        "resize": {"type": "bool", "default": True}
    }
)
def rotate_image(image, angle: float = 90.0, resize: bool = True):
    return skimage.transform.rotate(image, angle, resize=resize)

# --- RESCALE IMAGE ---
@register_node(
    label="Rescale",
    category="Transform",
    outputs=["rescaled"],
    params_config={
        "scale": {"min": 0.1, "max": 4.0, "step": 0.1}
    }
)
def rescale_image(image, scale: float = 0.5):
    # Anti-aliasing is good practice when shrinking
    return skimage.transform.rescale(image, scale, anti_aliasing=True, channel_axis=None)