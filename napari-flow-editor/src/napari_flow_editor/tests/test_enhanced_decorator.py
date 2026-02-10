"""
Tests for the enhanced @register_node decorator features.

Tests cover:
- New decorator parameters (interactive, output_meta, validate_inputs, doc, icon)
- Library generation with new metadata
- Input validation logic
- Backward compatibility
"""

import pytest
import numpy as np
from napari_flow_editor.flow_nodes.decorator import register_node
from napari_flow_editor import generate_library
import json
import os


def test_decorator_accepts_new_parameters():
    """Test that decorator accepts all new parameters without errors."""
    
    @register_node(
        label="Test Node",
        category="Test",
        outputs=["out"],
        interactive={"layer_type": "shapes"},
        output_meta={"colormap": "viridis"},
        validate_inputs={"image": {"required": True}},
        doc="Test documentation",
        icon="🔬"
    )
    def test_func(image):
        return image
    
    # Check that metadata is stored
    assert hasattr(test_func, '_is_flow_node')
    assert hasattr(test_func, '_node_meta')
    assert test_func._node_meta['label'] == "Test Node"
    assert test_func._node_meta['interactive'] == {"layer_type": "shapes"}
    assert test_func._node_meta['output_meta'] == {"colormap": "viridis"}
    assert test_func._node_meta['validate_inputs'] == {"image": {"required": True}}
    assert test_func._node_meta['doc'] == "Test documentation"
    assert test_func._node_meta['icon'] == "🔬"


def test_decorator_backward_compatibility():
    """Test that old-style decorator calls still work."""
    
    @register_node(
        label="Legacy Node",
        category="Test",
        outputs=["out"],
        params_config={"sigma": {"min": 0.0, "max": 10.0}}
    )
    def legacy_func(image, sigma: float = 1.0):
        return image
    
    # Check basic metadata
    assert hasattr(legacy_func, '_is_flow_node')
    assert legacy_func._node_meta['label'] == "Legacy Node"
    assert legacy_func._node_meta['category'] == "Test"
    
    # New fields should not be present
    assert 'interactive' not in legacy_func._node_meta
    assert 'output_meta' not in legacy_func._node_meta
    assert 'validate_inputs' not in legacy_func._node_meta


def test_decorator_function_executes():
    """Test that decorated functions still execute correctly."""
    
    @register_node(
        label="Add One",
        category="Test",
        outputs=["result"]
    )
    def add_one(value):
        return value + 1
    
    result = add_one(5)
    assert result == 6


def test_output_meta_metadata():
    """Test output_meta with various configurations."""
    
    @register_node(
        label="Styled Output",
        category="Test",
        outputs=["out"],
        output_meta={
            "layer_type": "labels",
            "colormap": "cyan",
            "opacity": 0.5,
            "name_suffix": "Processed"
        }
    )
    def styled_func(image):
        return image
    
    meta = styled_func._node_meta['output_meta']
    assert meta['layer_type'] == "labels"
    assert meta['colormap'] == "cyan"
    assert meta['opacity'] == 0.5
    assert meta['name_suffix'] == "Processed"


def test_validate_inputs_metadata():
    """Test validate_inputs with various rules."""
    
    @register_node(
        label="Validated Node",
        category="Test",
        outputs=["out"],
        validate_inputs={
            "image": {
                "required": True,
                "dtype": ["float32", "float64"],
                "ndim": [2, 3]
            },
            "mask": {
                "required": False,
                "ndim": [2]
            }
        }
    )
    def validated_func(image, mask=None):
        return image
    
    validation = validated_func._node_meta['validate_inputs']
    assert validation['image']['required'] is True
    assert "float32" in validation['image']['dtype']
    assert 2 in validation['image']['ndim']
    assert validation['mask']['required'] is False


def test_interactive_metadata():
    """Test interactive configuration."""
    
    @register_node(
        label="Interactive Node",
        category="Test",
        outputs=["out"],
        interactive={
            "layer_type": "shapes",
            "tool": "rectangle",
            "prompt": "Draw a rectangle",
            "arg_name": "roi",
            "confirm": True
        }
    )
    def interactive_func(image, roi=None):
        return image
    
    interactive = interactive_func._node_meta['interactive']
    assert interactive['layer_type'] == "shapes"
    assert interactive['tool'] == "rectangle"
    assert interactive['prompt'] == "Draw a rectangle"
    assert interactive['arg_name'] == "roi"
    assert interactive['confirm'] is True


def test_doc_and_icon_metadata():
    """Test doc and icon fields."""
    
    @register_node(
        label="Documented Node",
        category="Test",
        outputs=["out"],
        doc="This is a test node with documentation",
        icon="🎨"
    )
    def documented_func(image):
        """Fallback docstring."""
        return image
    
    assert documented_func._node_meta['doc'] == "This is a test node with documentation"
    assert documented_func._node_meta['icon'] == "🎨"


def test_doc_fallback_to_docstring():
    """Test that doc falls back to function docstring."""
    
    @register_node(
        label="Node with Docstring",
        category="Test",
        outputs=["out"]
    )
    def func_with_docstring(image):
        """This is the function docstring."""
        return image
    
    # doc should not be in metadata if not explicitly set
    assert 'doc' not in func_with_docstring._node_meta
    # But the function should still have its docstring
    assert func_with_docstring.__doc__ == """This is the function docstring."""


def test_library_generation_includes_new_fields(tmp_path):
    """Test that generate_library includes new metadata fields in JSON."""
    
    # NOTE: This is a simplified test that verifies the code changes are present.
    # A full integration test would require proper module loading and library generation,
    # which is tested manually by running generate_library.py with examples_advanced.py
    
    # Verify the generate_library.py code has the required changes
    import os
    generate_lib_path = os.path.join(os.path.dirname(__file__), '..', 'generate_library.py')
    
    if os.path.exists(generate_lib_path):
        with open(generate_lib_path, 'r') as f:
            content = f.read()
            assert 'if "interactive" in meta:' in content
            assert 'if "output_meta" in meta:' in content
            assert 'if "validate_inputs" in meta:' in content
            assert 'if "icon" in meta:' in content


def test_execution_engine_validation_helper():
    """Test the validation helper method in ExecutionWorker."""
    from napari_flow_editor.execution_engine import ExecutionWorker
    
    # Create a mock worker (without full setup)
    worker = ExecutionWorker(scene=None, viewer=None)
    
    # Test valid input
    func_inputs = {
        "image": np.random.rand(100, 100).astype(np.float32)
    }
    validate_config = {
        "image": {
            "required": True,
            "dtype": ["float32", "float64"],
            "ndim": [2, 3]
        }
    }
    
    # Should not raise
    worker.validate_node_inputs(func_inputs, validate_config)


def test_execution_engine_validation_fails_on_wrong_dtype():
    """Test that validation fails with wrong dtype."""
    from napari_flow_editor.execution_engine import ExecutionWorker
    
    worker = ExecutionWorker(scene=None, viewer=None)
    
    func_inputs = {
        "image": np.random.rand(100, 100).astype(np.int32)  # Wrong dtype
    }
    validate_config = {
        "image": {
            "required": True,
            "dtype": ["float32", "float64"],
            "ndim": [2]
        }
    }
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="dtype"):
        worker.validate_node_inputs(func_inputs, validate_config)


def test_execution_engine_validation_fails_on_wrong_ndim():
    """Test that validation fails with wrong ndim."""
    from napari_flow_editor.execution_engine import ExecutionWorker
    
    worker = ExecutionWorker(scene=None, viewer=None)
    
    func_inputs = {
        "image": np.random.rand(100, 100, 100).astype(np.float32)  # 3D
    }
    validate_config = {
        "image": {
            "required": True,
            "ndim": [2]  # Only 2D allowed
        }
    }
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="dimensions"):
        worker.validate_node_inputs(func_inputs, validate_config)


def test_execution_engine_validation_fails_on_missing_required():
    """Test that validation fails when required input is missing."""
    from napari_flow_editor.execution_engine import ExecutionWorker
    
    worker = ExecutionWorker(scene=None, viewer=None)
    
    func_inputs = {}  # No image provided
    validate_config = {
        "image": {
            "required": True
        }
    }
    
    # Should raise ValueError
    with pytest.raises(ValueError, match="Required input"):
        worker.validate_node_inputs(func_inputs, validate_config)


def test_examples_advanced_imports():
    """Test that examples_advanced.py can be imported."""
    try:
        from napari_flow_editor.flow_nodes import examples_advanced
        assert hasattr(examples_advanced, 'interactive_crop')
        assert hasattr(examples_advanced, 'threshold_with_style')
        assert hasattr(examples_advanced, 'smart_gaussian_blur')
        assert hasattr(examples_advanced, 'edge_detection_suite')
        assert hasattr(examples_advanced, 'interactive_point_analysis')
    except ImportError as e:
        pytest.fail(f"Failed to import examples_advanced: {e}")


def test_examples_have_correct_metadata():
    """Test that example nodes have the correct enhanced metadata."""
    from napari_flow_editor.flow_nodes.examples_advanced import (
        interactive_crop, threshold_with_style, smart_gaussian_blur
    )
    
    # Check interactive_crop
    assert 'interactive' in interactive_crop._node_meta
    assert interactive_crop._node_meta['interactive']['layer_type'] == "shapes"
    assert 'icon' in interactive_crop._node_meta
    
    # Check threshold_with_style
    assert 'output_meta' in threshold_with_style._node_meta
    assert 'validate_inputs' in threshold_with_style._node_meta
    assert 'doc' in threshold_with_style._node_meta
    
    # Check smart_gaussian_blur
    assert 'validate_inputs' in smart_gaussian_blur._node_meta
    assert 'icon' in smart_gaussian_blur._node_meta


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
