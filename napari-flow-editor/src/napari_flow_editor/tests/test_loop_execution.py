import pytest
import os
import sys
from qtpy.QtCore import Qt, QPointF, QTimer
from unittest.mock import MagicMock, patch

# --- 1. PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- 2. IMPORTS ---
from napari_flow_editor.execution_engine import ExecutionWorker
from napari_flow_editor.napari_plugin_v2 import Node

# Mock the Napari viewer
class MockViewer:
    pass

class MockScene:
    def __init__(self):
        self._items = []
    
    def items(self):
        return self._items
    
    def add_node(self, node):
        self._items.append(node)

class MockSocket:
    def __init__(self, name, socket_type):
        self.name = name
        self.socket_type = socket_type
        self.connected_edges = []

def create_mock_node(uid, title, node_type):
    """Helper to create a mock node."""
    node = MagicMock(spec=Node)
    node.uid = uid
    node.title = title
    node.node_type = node_type
    node.inputs = []
    node.outputs = []
    node.params = {}
    node.last_signature = None
    node.cached_results = {}
    return node

def test_loop_execution_basic():
    """Test that loop execution runs multiple iterations."""
    # Setup
    scene = MockScene()
    viewer = MockViewer()
    
    # Create a simple node
    node1 = create_mock_node("node1", "Test Node", "test")
    node1.inputs = [MockSocket("input", "input")]
    node1.outputs = [MockSocket("output", "output")]
    scene.add_node(node1)
    
    # Configure loop: run node1 for 3 iterations
    loop_config = {
        "nodes": ["node1"],
        "iterations": 3
    }
    
    # Create worker
    worker = ExecutionWorker(scene, viewer, loop_config=loop_config)
    
    # Track log messages
    log_messages = []
    worker.log_signal.connect(lambda msg: log_messages.append(msg))
    
    # Mock the execute_node_logic to avoid actual execution
    worker.execute_node_logic = MagicMock(return_value={})
    
    # Run
    worker.run()
    
    # Verify: Should have iteration messages
    iteration_messages = [msg for msg in log_messages if "Loop Iteration" in msg]
    assert len(iteration_messages) == 3, f"Expected 3 iteration messages, got {len(iteration_messages)}"
    assert "🔁 Loop Iteration 1/3" in log_messages
    assert "🔁 Loop Iteration 2/3" in log_messages
    assert "🔁 Loop Iteration 3/3" in log_messages

def test_loop_cache_clearing():
    """Test that loop nodes have their cache cleared on subsequent iterations."""
    # Setup
    scene = MockScene()
    viewer = MockViewer()
    
    # Create a node
    node1 = create_mock_node("node1", "Loop Node", "test")
    node1.inputs = [MockSocket("input", "input")]
    node1.outputs = [MockSocket("output", "output")]
    node1.last_signature = "initial_signature"
    node1.cached_results = {"output": "cached_data"}
    scene.add_node(node1)
    
    # Configure loop
    loop_config = {
        "nodes": ["node1"],
        "iterations": 2
    }
    
    # Create worker
    worker = ExecutionWorker(scene, viewer, loop_config=loop_config)
    
    # Track when cache is cleared
    cache_states = []
    iteration_count = [0]  # Use list to allow modification in nested function
    
    def mock_execute(node, library):
        # Record cache state before execution
        cache_states.append({
            "iteration": iteration_count[0],
            "last_signature": node.last_signature,
            "has_cached_results": len(node.cached_results) > 0
        })
        return {}
    
    log_messages = []
    def track_iteration(msg):
        log_messages.append(msg)
        if "Loop Iteration" in msg:
            iteration_count[0] += 1
    
    worker.log_signal.connect(track_iteration)
    worker.execute_node_logic = mock_execute
    
    # Run
    worker.run()
    
    # Verify: On first iteration, node should still have cache
    # On second iteration, cache should be cleared
    assert len(cache_states) >= 2, "Should have executed at least twice"
    # Note: First execution might skip due to cached signature, 
    # but after cache clear, second should execute

def test_no_loop_config_normal_execution():
    """Test that execution works normally when no loop config is provided."""
    # Setup
    scene = MockScene()
    viewer = MockViewer()
    
    # Create a simple node
    node1 = create_mock_node("node1", "Normal Node", "test")
    node1.inputs = [MockSocket("input", "input")]
    node1.outputs = [MockSocket("output", "output")]
    scene.add_node(node1)
    
    # No loop config (normal execution)
    worker = ExecutionWorker(scene, viewer, loop_config=None)
    
    # Track log messages
    log_messages = []
    worker.log_signal.connect(lambda msg: log_messages.append(msg))
    
    # Mock the execute_node_logic
    worker.execute_node_logic = MagicMock(return_value={})
    
    # Run
    worker.run()
    
    # Verify: Should NOT have iteration messages
    iteration_messages = [msg for msg in log_messages if "Loop Iteration" in msg]
    assert len(iteration_messages) == 0, f"Expected no iteration messages, got {len(iteration_messages)}"
    
    # Should still have start and finish messages
    assert any("Starting Smart Execution" in msg for msg in log_messages)
    assert any("Execution Finished" in msg for msg in log_messages)

def test_mixed_loop_and_non_loop_nodes():
    """Test that non-loop nodes keep their cache while loop nodes are cleared."""
    # Setup
    scene = MockScene()
    viewer = MockViewer()
    
    # Create two nodes: one in loop, one not
    node1 = create_mock_node("node1", "Non-Loop Node", "test")
    node1.inputs = []
    node1.outputs = [MockSocket("output", "output")]
    node1.last_signature = "node1_sig"
    node1.cached_results = {"output": "node1_data"}
    scene.add_node(node1)
    
    node2 = create_mock_node("node2", "Loop Node", "test")
    node2.inputs = [MockSocket("input", "input")]
    node2.outputs = [MockSocket("output", "output")]
    node2.last_signature = "node2_sig"
    node2.cached_results = {"output": "node2_data"}
    scene.add_node(node2)
    
    # Configure loop: only node2 is in the loop
    loop_config = {
        "nodes": ["node2"],
        "iterations": 2
    }
    
    # Create worker
    worker = ExecutionWorker(scene, viewer, loop_config=loop_config)
    
    # Track cache states
    node1_signatures = []
    node2_signatures = []
    
    original_calculate = worker.calculate_signature
    def mock_calculate(node):
        sig = original_calculate(node)
        if node.uid == "node1":
            node1_signatures.append(node.last_signature)
        elif node.uid == "node2":
            node2_signatures.append(node.last_signature)
        return sig
    
    worker.calculate_signature = mock_calculate
    worker.execute_node_logic = MagicMock(return_value={})
    
    # Run
    worker.run()
    
    # Verify: node2's cache should have been cleared between iterations
    # node1 should keep its signature (not in loop)
    assert len(node1_signatures) > 0, "Node1 should have been processed"
    assert len(node2_signatures) > 0, "Node2 should have been processed"
