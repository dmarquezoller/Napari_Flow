"""
Shared pytest fixtures for napari-flow-editor tests.

This module provides common fixtures used across multiple test files,
including mock Node objects and mock scenes for testing execution logic.
"""

import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def mock_node():
    """
    Create a mock Node object for testing.
    
    Returns a node with basic properties (uid, title, params, inputs, outputs)
    and caching attributes (last_signature, cached_results).
    """
    node = Mock()
    node.uid = "test-node-123"
    node.title = "Test Node"
    node.node_type = "test_node"
    node.params = {}
    node.inputs = []
    node.outputs = []
    node.last_signature = None
    node.cached_results = {}
    return node


@pytest.fixture
def mock_socket():
    """Create a mock Socket object for testing."""
    socket = Mock()
    socket.name = "test_socket"
    socket.connected_edges = []
    socket.node = None
    return socket


@pytest.fixture
def mock_connected_nodes():
    """
    Create a simple connected graph for topological sort testing.
    
    Returns a tuple of (node_a, node_b) where node_a feeds into node_b.
    """
    # Create two nodes
    node_a = Mock()
    node_a.uid = "node-a"
    node_a.title = "Node A"
    node_a.inputs = []
    node_a.outputs = []
    node_a.last_signature = "sig-a"
    node_a.cached_results = {}
    
    node_b = Mock()
    node_b.uid = "node-b"
    node_b.title = "Node B"
    node_b.last_signature = "sig-b"
    node_b.cached_results = {}
    
    # Create sockets
    output_socket = Mock()
    output_socket.name = "out"
    output_socket.node = node_a
    output_socket.connected_edges = []
    
    input_socket = Mock()
    input_socket.name = "in"
    input_socket.node = node_b
    
    # Create edge connecting them
    edge = Mock()
    edge.start_socket = output_socket
    edge.end_socket = input_socket
    
    input_socket.connected_edges = [edge]
    output_socket.connected_edges = [edge]
    
    node_a.outputs = [output_socket]
    node_b.inputs = [input_socket]
    node_b.outputs = []
    
    return node_a, node_b


@pytest.fixture
def mock_diamond_graph():
    """
    Create a diamond-shaped graph for testing complex topological sorts.
    
    Structure:
        A
       / \\
      B   C
       \\ /
        D
    
    Returns (node_a, node_b, node_c, node_d)
    """
    # Create nodes
    nodes = {}
    for name in ['A', 'B', 'C', 'D']:
        node = Mock()
        node.uid = f"node-{name}"
        node.title = f"Node {name}"
        node.last_signature = f"sig-{name}"
        node.cached_results = {}
        node.inputs = []
        node.outputs = []
        nodes[name] = node
    
    # Helper to connect two nodes
    def connect(source, target):
        out_socket = Mock()
        out_socket.name = f"{source.title}_out"
        out_socket.node = source
        out_socket.connected_edges = []
        
        in_socket = Mock()
        in_socket.name = f"{target.title}_in"
        in_socket.node = target
        
        edge = Mock()
        edge.start_socket = out_socket
        edge.end_socket = in_socket
        
        in_socket.connected_edges = [edge]
        out_socket.connected_edges = [edge]
        
        source.outputs.append(out_socket)
        target.inputs.append(in_socket)
    
    # Connect: A->B, A->C, B->D, C->D
    connect(nodes['A'], nodes['B'])
    connect(nodes['A'], nodes['C'])
    connect(nodes['B'], nodes['D'])
    connect(nodes['C'], nodes['D'])
    
    return nodes['A'], nodes['B'], nodes['C'], nodes['D']


@pytest.fixture
def mock_scene():
    """Create a mock FlowScene for testing."""
    scene = Mock()
    scene.items = Mock(return_value=[])
    return scene


@pytest.fixture
def simple_node_function():
    """A simple function to use for testing the decorator."""
    def add_numbers(a, b, scale=1.0):
        """Add two numbers and scale the result."""
        return (a + b) * scale
    return add_numbers
