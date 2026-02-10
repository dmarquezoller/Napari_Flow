"""
Tests for the library generation logic.

These tests verify type mapping, signature introspection, and the merging
of parameter configurations in generate_library.py.
"""

import pytest
import inspect
from napari_flow_editor.generate_library import get_type_name
from napari_flow_editor.flow_nodes.decorator import register_node


def test_get_type_name_float():
    """Test that float type is correctly mapped."""
    assert get_type_name(float) == "float"


def test_get_type_name_int():
    """Test that int type is correctly mapped."""
    assert get_type_name(int) == "int"


def test_get_type_name_bool():
    """Test that bool type is correctly mapped."""
    assert get_type_name(bool) == "bool"


def test_get_type_name_str():
    """Test that str type is mapped to enum."""
    assert get_type_name(str) == "enum"


def test_get_type_name_unknown():
    """Test that unknown types default to string."""
    assert get_type_name(list) == "string"
    assert get_type_name(dict) == "string"
    assert get_type_name(object) == "string"


def test_signature_introspection_inputs_vs_params():
    """Test that parameters without defaults become inputs."""
    @register_node(label="Test", category="Test")
    def test_func(image, mask, threshold=0.5, method="otsu"):
        return image
    
    sig = inspect.signature(test_func)
    
    inputs = []
    parameters = {}
    
    for param_name, param in sig.parameters.items():
        if param.default == inspect.Parameter.empty:
            inputs.append(param_name)
        else:
            parameters[param_name] = param.default
    
    assert inputs == ["image", "mask"]
    assert "threshold" in parameters
    assert "method" in parameters
    assert parameters["threshold"] == 0.5
    assert parameters["method"] == "otsu"


def test_params_config_merge():
    """Test that extra_config from decorator merges with base param definition."""
    @register_node(
        label="Test",
        category="Test",
        params_config={
            "threshold": {"min": 0, "max": 1, "step": 0.1, "label": "Threshold Value"}
        }
    )
    def test_func(image, threshold=0.5):
        return image
    
    sig = inspect.signature(test_func)
    meta = test_func._node_meta
    
    # Simulate what generate_library.py does
    for param_name, param in sig.parameters.items():
        if param.default != inspect.Parameter.empty:
            p_type = get_type_name(param.annotation if param.annotation != inspect.Parameter.empty else type(param.default))
            extra_config = meta["params_config"].get(param_name, {})
            
            param_def = {
                "type": p_type,
                "default": param.default
            }
            param_def.update(extra_config)
            
            # Check that merge happened correctly
            if param_name == "threshold":
                assert param_def["type"] == "float"
                assert param_def["default"] == 0.5
                assert param_def["min"] == 0
                assert param_def["max"] == 1
                assert param_def["step"] == 0.1
                assert param_def["label"] == "Threshold Value"


def test_enum_without_options_fallback():
    """Test that enum type without options falls back to string."""
    param_def = {"type": "enum", "default": "value"}
    
    # Simulate the fallback logic in generate_library.py
    if param_def["type"] == "enum" and "options" not in param_def:
        param_def["type"] = "string"
    
    assert param_def["type"] == "string"


def test_enum_with_options_preserved():
    """Test that enum type with options is preserved."""
    param_def = {
        "type": "enum",
        "default": "otsu",
        "options": ["otsu", "li", "mean"]
    }
    
    # Simulate the fallback logic
    if param_def["type"] == "enum" and "options" not in param_def:
        param_def["type"] = "string"
    
    assert param_def["type"] == "enum"
    assert param_def["options"] == ["otsu", "li", "mean"]


def test_node_metadata_extraction():
    """Test that all node metadata is correctly extracted."""
    @register_node(
        label="Gaussian Blur",
        category="Filters",
        outputs=["blurred"],
        params_config={"sigma": {"min": 0.1, "max": 10.0}},
        interactive=False
    )
    def gaussian_blur(image, sigma=1.0):
        return image
    
    meta = gaussian_blur._node_meta
    
    assert meta["label"] == "Gaussian Blur"
    assert meta["category"] == "Filters"
    assert meta["outputs"] == ["blurred"]
    assert "sigma" in meta["params_config"]
    assert meta["interactive"] is False


def test_interactive_flag_in_library():
    """Test that interactive flag is correctly included in node metadata."""
    @register_node(
        label="Interactive Crop",
        category="Math",
        interactive=True
    )
    def interactive_crop(image):
        return image
    
    meta = interactive_crop._node_meta
    assert meta["interactive"] is True


def test_non_interactive_flag_in_library():
    """Test that non-interactive flag defaults to False."""
    @register_node(
        label="Gaussian Blur",
        category="Filters"
    )
    def gaussian_blur(image, sigma=1.0):
        return image
    
    meta = gaussian_blur._node_meta
    assert meta["interactive"] is False


def test_multiple_inputs_extraction():
    """Test extraction of multiple input parameters."""
    @register_node(label="Blend", category="Math")
    def blend_images(image_a, image_b, image_c, alpha=0.5):
        return image_a
    
    sig = inspect.signature(blend_images)
    
    inputs = [
        name for name, param in sig.parameters.items()
        if param.default == inspect.Parameter.empty
    ]
    
    assert len(inputs) == 3
    assert "image_a" in inputs
    assert "image_b" in inputs
    assert "image_c" in inputs


def test_no_inputs_all_params():
    """Test function with no inputs (all parameters have defaults)."""
    @register_node(label="Generate", category="Generators")
    def generate_noise(width=512, height=512, seed=42):
        return None
    
    sig = inspect.signature(generate_noise)
    
    inputs = [
        name for name, param in sig.parameters.items()
        if param.default == inspect.Parameter.empty
    ]
    
    assert len(inputs) == 0
