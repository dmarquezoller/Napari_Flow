import pytest
import os
import sys
from qtpy.QtCore import Qt, QPointF

# --- 1. PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- 2. IMPORTS ---
# Import from your ACTUAL main file name: napari_plugin_v2
from napari_flow_editor.napari_plugin_v2 import FlowEditor, Node, Connection

# Mock the Napari viewer
class MockViewer:
    pass

@pytest.fixture
def flow_editor(qtbot):
    """Fixture to create the GUI before every test."""
    viewer = MockViewer()
    widget = FlowEditor(viewer)
    qtbot.addWidget(widget)
    return widget

def test_add_node(flow_editor):
    """Test adding a node programmatically."""
    scene = flow_editor.scene
    
    # Verify scene is empty initially (or has default items)
    initial_count = len(scene.items())
    
    # Add a node
    flow_editor.add_node("gaussian_blur", pos=QPointF(0, 0))
    
    # Verify item count increased (Node + Sockets)
    # 1 Node + 1 Input + 1 Output = 3 new items
    assert len(scene.items()) == initial_count + 3
    
    # Check the node properties
    nodes = [i for i in scene.items() if isinstance(i, Node)]
    assert len(nodes) > 0
    assert "Gaussian Blur" in nodes[0].title

def test_connection_logic(flow_editor):
    """Test connecting two nodes."""
    scene = flow_editor.scene
    
    # 1. Create two nodes
    node1 = flow_editor.add_node("gaussian_blur", pos=QPointF(0, 0))
    node2 = flow_editor.add_node("gaussian_blur", pos=QPointF(200, 0))
    
    # 2. Get Output of Node 1 and Input of Node 2
    out_socket = node1.outputs[0]
    in_socket = node2.inputs[0]
    
    # 3. Create Connection
    conn = Connection(out_socket, scene)
    conn.finalize(in_socket)
    scene.addItem(conn)
    
    # 4. Assert connection is registered
    assert conn in out_socket.connected_edges
    assert conn in in_socket.connected_edges