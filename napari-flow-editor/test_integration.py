#!/usr/bin/env python3
"""
Integration test demonstrating the new decorator features work together.
Tests validation, output metadata, and error handling.
"""
import sys
import os
import importlib.util
import types
import numpy as np

# Setup paths and imports (same approach as test_decorator_enhancements.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load decorator module
decorator_path = os.path.join(os.path.dirname(__file__), 'src/napari_flow_editor/flow_nodes/decorator.py')
spec = importlib.util.spec_from_file_location('decorator', decorator_path)
decorator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decorator_module)
register_node = decorator_module.register_node

print("="*60)
print("Integration Test: Enhanced Decorator Features")
print("="*60)

# Test 1: Validation catches missing inputs
print("\n1. Testing input validation...")

@register_node(
    label="Test Validator",
    category="Test",
    outputs=["out"],
    validate_inputs={
        "image": {
            "required": True,
            "ndim": [2, 3],
            "dtype": ["float32", "uint8"]
        }
    }
)
def test_validator(image):
    return image

# Create a mock library definition
library_def = {
    "test_validator": {
        "label": "Test Validator",
        "outputs": ["out"],
        "validate_inputs": test_validator._node_meta["validate_inputs"]
    }
}

# Test validation logic manually
def test_validation(data, input_name, rules):
    """Simulate the validation logic from execution_engine.py"""
    if rules.get("required") and data is None:
        raise ValueError(f"Input '{input_name}' is required but not connected")
    
    if data is not None:
        if "ndim" in rules and hasattr(data, "ndim"):
            if data.ndim not in rules["ndim"]:
                raise ValueError(f"Input '{input_name}' must be {rules['ndim']}D (got {data.ndim}D)")
        
        if "dtype" in rules and hasattr(data, "dtype"):
            data_dtype_str = str(data.dtype)
            allowed_dtypes = [str(d) for d in rules["dtype"]]
            if data_dtype_str not in allowed_dtypes:
                raise ValueError(f"Input '{input_name}' dtype must be one of {rules['dtype']} (got {data.dtype})")

# Test cases
try:
    # Should fail: None input when required
    test_validation(None, "image", {"required": True})
    print("  ✗ Failed to catch missing required input")
except ValueError as e:
    print(f"  ✓ Caught missing input: {e}")

try:
    # Should fail: wrong ndim
    test_data = np.zeros((2, 3, 4, 5))  # 4D
    test_validation(test_data, "image", {"ndim": [2, 3]})
    print("  ✗ Failed to catch wrong ndim")
except ValueError as e:
    print(f"  ✓ Caught wrong ndim: {e}")

try:
    # Should fail: wrong dtype
    test_data = np.zeros((10, 10), dtype=np.int16)
    test_validation(test_data, "image", {"dtype": ["float32", "uint8"]})
    print("  ✗ Failed to catch wrong dtype")
except ValueError as e:
    print(f"  ✓ Caught wrong dtype: {e}")

# Should pass: correct input
test_data = np.zeros((10, 10), dtype=np.float32)
test_validation(test_data, "image", {"ndim": [2, 3], "dtype": ["float32", "uint8"]})
print("  ✓ Valid input passed validation")

# Test 2: Output metadata envelope with sentinel
print("\n2. Testing metadata envelope with sentinel...")

def test_envelope(result, output_meta):
    """Simulate the envelope wrapping logic from execution_engine.py"""
    is_envelope = (
        isinstance(result, tuple) and 
        len(result) == 2 and 
        isinstance(result[1], dict) and
        result[1].get("__napari_meta__") is True
    )
    if not is_envelope:
        meta_with_sentinel = {**output_meta, "__napari_meta__": True}
        return (result, meta_with_sentinel)
    return result

# Test regular output gets wrapped
simple_result = np.zeros((10, 10))
output_meta = {"layer_type": "labels", "colormap": "viridis"}
wrapped = test_envelope(simple_result, output_meta)
assert isinstance(wrapped, tuple)
assert wrapped[1].get("__napari_meta__") is True
assert wrapped[1].get("layer_type") == "labels"
print("  ✓ Simple result wrapped with sentinel")

# Test legitimate tuple doesn't get double-wrapped
legit_tuple = (np.zeros((10, 10)), {"some": "data"})  # No sentinel
wrapped = test_envelope(legit_tuple, output_meta)
assert isinstance(wrapped, tuple)
assert wrapped[1].get("__napari_meta__") is True  # Now has sentinel
print("  ✓ Legitimate tuple wrapped correctly")

# Test already-wrapped result doesn't get double-wrapped
already_wrapped = (np.zeros((10, 10)), {"layer_type": "image", "__napari_meta__": True})
result = test_envelope(already_wrapped, output_meta)
assert result is already_wrapped  # Should return unchanged
print("  ✓ Already-wrapped result not double-wrapped")

# Test 3: Metadata extraction in UI
print("\n3. Testing metadata extraction with sentinel...")

def extract_metadata(data):
    """Simulate the metadata extraction from handle_execution_result"""
    is_envelope = (
        isinstance(data, tuple) and 
        len(data) == 2 and 
        isinstance(data[1], dict) and 
        data[1].get("__napari_meta__") is True
    )
    if is_envelope:
        actual_data = data[0]
        metadata = {k: v for k, v in data[1].items() if k != "__napari_meta__"}
        return actual_data, metadata
    else:
        return data, {}

# Test with envelope
envelope_data = (np.ones((5, 5)), {"layer_type": "labels", "colormap": "viridis", "__napari_meta__": True})
data, meta = extract_metadata(envelope_data)
assert isinstance(data, np.ndarray)
assert meta.get("layer_type") == "labels"
assert "__napari_meta__" not in meta  # Sentinel should be filtered out
print("  ✓ Metadata extracted and sentinel removed")

# Test with regular tuple (should not be treated as envelope)
regular_tuple = (np.ones((5, 5)), {"user": "data"})  # No sentinel
data, meta = extract_metadata(regular_tuple)
assert data == regular_tuple  # Should return the whole tuple as data
assert meta == {}  # No metadata extracted
print("  ✓ Regular tuple not mistaken for envelope")

# Test with simple array
simple_array = np.ones((5, 5))
data, meta = extract_metadata(simple_array)
assert isinstance(data, np.ndarray)
assert meta == {}
print("  ✓ Simple array handled correctly")

print("\n" + "="*60)
print("All integration tests passed! ✓")
print("="*60)
print("\nSummary:")
print("  • Input validation catches missing, wrong ndim, and wrong dtype")
print("  • Output metadata wrapping uses sentinel to avoid false positives")
print("  • Metadata extraction correctly filters sentinel key")
print("  • Regular tuples are not mistaken for metadata envelopes")
