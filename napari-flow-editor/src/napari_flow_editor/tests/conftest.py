import numpy as np
import pytest


class FakeLayer:
    def __init__(self, data, metadata=None):
        self.data = data
        self.metadata = metadata or {}


class FakeViewer:
    def __init__(self, layers=None):
        self.layers = layers or {}


class FakeSocket:
    def __init__(self, name):
        self.name = name
        self.connected_edges = []


class FakeEdge:
    def __init__(self, start_socket):
        self.start_socket = start_socket


class FakeNode:
    def __init__(self, node_type, title="Node", params=None, inputs=None, outputs=None):
        self.node_type = node_type
        self.title = title
        self.uid = "fake-uid"
        self.params = params or {}
        self.parameters = self.params.copy()
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.cached_results = {}
        self.last_signature = None


@pytest.fixture
def viewer():
    data = np.arange(12).reshape(3, 4)
    layer = FakeLayer(data, metadata={"name": "LayerA"})
    return FakeViewer(layers={"LayerA": layer})


@pytest.fixture
def worker(viewer):
    from napari_flow_editor.execution_engine import ExecutionWorker
    return ExecutionWorker(scene=None, viewer=viewer)