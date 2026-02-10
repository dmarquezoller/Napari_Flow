"""
Tests for the @register_node decorator.

These tests verify that the decorator correctly marks functions as flow nodes,
stores metadata, and maintains function behavior.
"""

import pytest
from napari_flow_editor.flow_nodes.decorator import register_node


def test_decorator_marks_function_as_flow_node():
    """Test that the decorator adds the _is_flow_node flag."""
    @register_node(label="Test Node", category="Test")
    def test_func():
        return 42
    
    assert hasattr(test_func, '_is_flow_node')
    assert test_func._is_flow_node is True


def test_decorator_stores_metadata():
    """Test that the decorator stores all metadata correctly."""
    @register_node(
        label="Test Node",
        category="Test Category",
        outputs=["output1", "output2"],
        params_config={"param1": {"type": "float"}},
        interactive=True
    )
    def test_func():
        return 1, 2
    
    assert hasattr(test_func, '_node_meta')
    meta = test_func._node_meta
    
    assert meta['label'] == "Test Node"
    assert meta['category'] == "Test Category"
    assert meta['outputs'] == ["output1", "output2"]
    assert meta['params_config'] == {"param1": {"type": "float"}}
    assert meta['interactive'] is True


def test_decorator_default_outputs():
    """Test that outputs default to ['out'] if not specified."""
    @register_node(label="Test Node", category="Test")
    def test_func():
        return 42
    
    assert test_func._node_meta['outputs'] == ["out"]


def test_decorator_default_params_config():
    """Test that params_config defaults to empty dict if not specified."""
    @register_node(label="Test Node", category="Test")
    def test_func():
        return 42
    
    assert test_func._node_meta['params_config'] == {}


def test_decorator_default_interactive():
    """Test that interactive defaults to False if not specified."""
    @register_node(label="Test Node", category="Test")
    def test_func():
        return 42
    
    assert test_func._node_meta['interactive'] is False


def test_decorated_function_still_works():
    """Test that the decorated function can still be called and returns correct values."""
    @register_node(label="Add", category="Math")
    def add_func(a, b):
        return a + b
    
    result = add_func(5, 3)
    assert result == 8


def test_decorated_function_with_params():
    """Test that decorated function works with parameters."""
    @register_node(
        label="Scale",
        category="Math",
        params_config={"scale": {"type": "float", "default": 1.0}}
    )
    def scale_func(value, scale=1.0):
        return value * scale
    
    assert scale_func(10) == 10
    assert scale_func(10, scale=2.5) == 25


def test_decorator_preserves_function_name():
    """Test that functools.wraps preserves the function name."""
    @register_node(label="Test", category="Test")
    def my_special_function():
        return 42
    
    assert my_special_function.__name__ == "my_special_function"


def test_decorator_preserves_docstring():
    """Test that functools.wraps preserves the function docstring."""
    @register_node(label="Test", category="Test")
    def documented_function():
        """This is a test function with documentation."""
        return 42
    
    assert "test function with documentation" in documented_function.__doc__


def test_multiple_outputs():
    """Test decorator with multiple outputs."""
    @register_node(
        label="Split",
        category="Math",
        outputs=["first", "second", "third"]
    )
    def split_func(value):
        return value, value * 2, value * 3
    
    assert test_func._node_meta['outputs'] == ["first", "second", "third"]
    result = split_func(10)
    assert result == (10, 20, 30)


def test_complex_params_config():
    """Test decorator with complex params configuration."""
    params = {
        "threshold": {"type": "float", "min": 0, "max": 1, "step": 0.1},
        "method": {"type": "enum", "options": ["otsu", "li", "mean"]},
        "invert": {"type": "bool", "default": False}
    }
    
    @register_node(
        label="Threshold",
        category="Segmentation",
        params_config=params
    )
    def threshold_func(image, threshold=0.5, method="otsu", invert=False):
        return image > threshold
    
    assert test_func._node_meta['params_config'] == params
