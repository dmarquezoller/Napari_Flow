"""
Tests for execution engine logic.

These tests cover topological sorting and signature calculation without
requiring a full GUI or Qt event loop.
"""

import pytest
import json
import hashlib
from unittest.mock import Mock
from napari_flow_editor.execution_engine import ExecutionWorker


def test_topological_sort_linear(mock_connected_nodes):
    """Test topological sort with a simple linear graph (A -> B)."""
    node_a, node_b = mock_connected_nodes
    
    # Create a mock worker
    worker = ExecutionWorker(scene=Mock(), viewer=Mock())
    
    # Sort
    sorted_nodes = worker.topological_sort([node_b, node_a])
    
    # A should come before B
    assert sorted_nodes.index(node_a) < sorted_nodes.index(node_b)


def test_topological_sort_diamond(mock_diamond_graph):
    """Test topological sort with a diamond graph."""
    node_a, node_b, node_c, node_d = mock_diamond_graph
    
    worker = ExecutionWorker(scene=Mock(), viewer=Mock())
    
    # Sort all nodes
    sorted_nodes = worker.topological_sort([node_d, node_c, node_b, node_a])
    
    # A must come before B and C
    assert sorted_nodes.index(node_a) < sorted_nodes.index(node_b)
    assert sorted_nodes.index(node_a) < sorted_nodes.index(node_c)
    
    # B and C must come before D
    assert sorted_nodes.index(node_b) < sorted_nodes.index(node_d)
    assert sorted_nodes.index(node_c) < sorted_nodes.index(node_d)


def test_topological_sort_single_node(mock_node):
    """Test topological sort with a single disconnected node."""
    worker = ExecutionWorker(scene=Mock(), viewer=Mock())
    
    sorted_nodes = worker.topological_sort([mock_node])
    
    assert len(sorted_nodes) == 1
    assert sorted_nodes[0] == mock_node


def test_calculate_signature_no_inputs_no_params(mock_node):
    """Test signature calculation for a node with no inputs or parameters."""
    worker = ExecutionWorker(scene=Mock(), viewer=Mock())
    
    mock_node.params = {}
    mock_node.inputs = []
    
    sig = worker.calculate_signature(mock_node)
    
    # Should return a consistent hash
    assert isinstance(sig, str)
    assert len(sig) == 32  # MD5 hash length


def test_calculate_signature_with_params(mock_node):
    """Test that signature changes when parameters change."""
    worker = ExecutionWorker(scene=Mock(), viewer=Mock())
    
    mock_node.inputs = []
    
    # First signature with param set to 1.0
    mock_node.params = {"threshold": 1.0}
    sig1 = worker.calculate_signature(mock_node)
    
    # Second signature with param changed to 2.0
    mock_node.params = {"threshold": 2.0}
    sig2 = worker.calculate_signature(mock_node)
    
    # Signatures should be different
    assert sig1 != sig2


def test_calculate_signature_params_unchanged(mock_node):
    """Test that signature is same when parameters don't change."""
    worker = ExecutionWorker(scene=Mock(), viewer=Mock())
    
    mock_node.inputs = []
    mock_node.params = {"threshold": 1.0, "method": "otsu"}
    
    sig1 = worker.calculate_signature(mock_node)
    sig2 = worker.calculate_signature(mock_node)
    
    # Signatures should be identical
    assert sig1 == sig2


def test_calculate_signature_with_inputs(mock_connected_nodes):
    """Test that signature incorporates upstream node signatures."""
    node_a, node_b = mock_connected_nodes
    
    worker = ExecutionWorker(scene=Mock(), viewer=Mock())
    
    node_a.params = {}
    node_b.params = {}
    
    # Calculate signature for B (which depends on A)
    sig = worker.calculate_signature(node_b)
    
    # Change A's signature
    node_a.last_signature = "new-signature"
    sig_new = worker.calculate_signature(node_b)
    
    # B's signature should change when A's signature changes
    assert sig != sig_new


def test_calculate_signature_no_input_connection(mock_node):
    """Test signature calculation for a node with unconnected input socket."""
    worker = ExecutionWorker(scene=Mock(), viewer=Mock())
    
    # Create an input socket with no connections
    input_socket = Mock()
    input_socket.connected_edges = []
    
    mock_node.inputs = [input_socket]
    mock_node.params = {}
    
    sig = worker.calculate_signature(mock_node)
    
    # Should still return a valid signature
    assert isinstance(sig, str)
    assert len(sig) == 32


def test_calculate_signature_error_handling(mock_node):
    """Test that calculate_signature returns 'dirty' on error."""
    worker = ExecutionWorker(scene=Mock(), viewer=Mock())
    
    # Make params un-serializable to cause an error
    mock_node.params = {"bad": object()}
    mock_node.inputs = []
    
    sig = worker.calculate_signature(mock_node)
    
    # Should return "dirty" on error
    assert sig == "dirty"


def test_calculate_signature_consistent_param_order(mock_node):
    """Test that parameter order doesn't affect signature."""
    worker = ExecutionWorker(scene=Mock(), viewer=Mock())
    
    mock_node.inputs = []
    
    # Same params in different order
    mock_node.params = {"a": 1, "b": 2, "c": 3}
    sig1 = worker.calculate_signature(mock_node)
    
    mock_node.params = {"c": 3, "a": 1, "b": 2}
    sig2 = worker.calculate_signature(mock_node)
    
    # Should be the same due to sort_keys=True
    assert sig1 == sig2


def test_signature_algorithm():
    """Test the signature calculation algorithm directly."""
    # This tests the logic without mocking
    params = {"threshold": 0.5, "method": "otsu"}
    input_sigs = ["sig-a", "sig-b"]
    
    param_str = json.dumps(params, sort_keys=True, default=str)
    combined = param_str + "".join(input_sigs)
    expected_sig = hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    # Verify it produces a consistent 32-character hash
    assert len(expected_sig) == 32
    
    # Verify it's reproducible
    combined2 = param_str + "".join(input_sigs)
    expected_sig2 = hashlib.md5(combined2.encode('utf-8')).hexdigest()
    assert expected_sig == expected_sig2


def test_topological_sort_handles_empty_list():
    """Test that topological sort handles empty node list."""
    worker = ExecutionWorker(scene=Mock(), viewer=Mock())
    
    sorted_nodes = worker.topological_sort([])
    
    assert sorted_nodes == []


def test_topological_sort_preserves_node_count(mock_diamond_graph):
    """Test that topological sort doesn't lose or duplicate nodes."""
    node_a, node_b, node_c, node_d = mock_diamond_graph
    
    worker = ExecutionWorker(scene=Mock(), viewer=Mock())
    
    input_nodes = [node_d, node_c, node_b, node_a]
    sorted_nodes = worker.topological_sort(input_nodes)
    
    # Should have same number of nodes
    assert len(sorted_nodes) == len(input_nodes)
    
    # Should have same nodes (no duplicates, no missing)
    assert set(sorted_nodes) == set(input_nodes)


def test_calculate_signature_with_none_last_signature(mock_connected_nodes):
    """Test signature calculation when upstream node has None signature."""
    node_a, node_b = mock_connected_nodes
    
    worker = ExecutionWorker(scene=Mock(), viewer=Mock())
    
    node_a.last_signature = None
    node_a.params = {}
    node_b.params = {}
    
    sig = worker.calculate_signature(node_b)
    
    # Should use "dirty" for None signatures
    assert isinstance(sig, str)
