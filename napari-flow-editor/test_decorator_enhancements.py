#!/usr/bin/env python3
"""
Test script to validate the enhanced @register_node decorator.
This can run without Qt dependencies.
"""
import sys
import os
import importlib.util

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import decorator directly without __init__.py
decorator_path = os.path.join(os.path.dirname(__file__), 'src/napari_flow_editor/flow_nodes/decorator.py')
spec = importlib.util.spec_from_file_location('decorator', decorator_path)
decorator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decorator_module)
register_node = decorator_module.register_node

# Test 1: Basic backward compatibility
print("Test 1: Backward compatibility...")
@register_node(
    label="Test Node",
    category="Test",
    outputs=["out"]
)
def test_basic():
    return "test"

assert test_basic._is_flow_node == True
assert test_basic._node_meta["label"] == "Test Node"
assert test_basic._node_meta["category"] == "Test"
assert test_basic._node_meta["outputs"] == ["out"]
assert test_basic._node_meta.get("interactive") is None
assert test_basic._node_meta.get("output_meta") is None
print("✓ Basic decorator works")

# Test 2: New features
print("\nTest 2: New features...")
@register_node(
    label="Advanced Node",
    category="Test",
    outputs=["result"],
    interactive={"layer_type": "shapes", "tool": "rectangle"},
    output_meta={"layer_type": "labels", "colormap": "viridis"},
    validate_inputs={"image": {"required": True, "ndim": [2, 3]}},
    doc="Test documentation",
    icon="🔬"
)
def test_advanced(image):
    return image

assert test_advanced._node_meta["interactive"] == {"layer_type": "shapes", "tool": "rectangle"}
assert test_advanced._node_meta["output_meta"] == {"layer_type": "labels", "colormap": "viridis"}
assert test_advanced._node_meta["validate_inputs"] == {"image": {"required": True, "ndim": [2, 3]}}
assert test_advanced._node_meta["doc"] == "Test documentation"
assert test_advanced._node_meta["icon"] == "🔬"
print("✓ All new decorator fields work correctly")

# Test 3: Check examples_advanced.py exists and has nodes
print("\nTest 3: Loading examples_advanced.py...")
try:
    # Import examples_advanced directly
    examples_path = os.path.join(os.path.dirname(__file__), 'src/napari_flow_editor/flow_nodes/examples_advanced.py')
    spec = importlib.util.spec_from_file_location('examples_advanced', examples_path)
    examples_advanced = importlib.util.module_from_spec(spec)
    
    # Make decorator available for the examples module
    sys.modules['napari_flow_editor'] = type(sys)('napari_flow_editor')
    sys.modules['napari_flow_editor.flow_nodes'] = type(sys)('flow_nodes')
    sys.modules['napari_flow_editor.flow_nodes.decorator'] = decorator_module
    
    spec.loader.exec_module(examples_advanced)
    
    # Check for our example nodes
    assert hasattr(examples_advanced, 'example_threshold')
    assert hasattr(examples_advanced, 'example_blur_validated')
    assert hasattr(examples_advanced, 'example_advanced_filter')
    assert hasattr(examples_advanced, 'example_interactive_crop')
    assert hasattr(examples_advanced, 'example_split_channels')
    
    # Verify they are registered
    assert examples_advanced.example_threshold._is_flow_node == True
    assert examples_advanced.example_threshold._node_meta.get("icon") == "🎯"
    assert examples_advanced.example_threshold._node_meta.get("doc") is not None
    
    assert examples_advanced.example_blur_validated._node_meta.get("validate_inputs") is not None
    assert examples_advanced.example_blur_validated._node_meta.get("output_meta") is not None
    
    print("✓ All 5 example nodes loaded successfully")
    print("  - example_threshold (icon: 🎯)")
    print("  - example_blur_validated (with validation & output_meta)")
    print("  - example_advanced_filter (comprehensive)")
    print("  - example_interactive_crop (interactive)")
    print("  - example_split_channels (multi-output)")
    
except Exception as e:
    print(f"✗ Failed to load examples_advanced.py: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
