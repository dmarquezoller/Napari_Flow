import numpy as np
from .conftest import FakeNode, FakeSocket, FakeEdge


def test_execute_node_logic_wraps_result(worker):
    def dummy_func():
        return np.ones((2, 2))

    node = FakeNode(node_type="dummy", params={})
    library_def = {
        "dummy": {
            "outputs": ["out"],
            "executable": dummy_func,
        }
    }

    result = worker.execute_node_logic(node, library_def)
    data, meta = result["out"]

    assert data.shape == (2, 2)
    assert isinstance(meta, dict)


def test_execute_node_logic_unwraps_input_envelope(worker):
    # Upstream node with cached envelope
    upstream = FakeNode("upstream")
    upstream.cached_results = {
        "out": (np.zeros((3, 3)), {"axes": "ZYX"})
    }

    # Downstream node expects input
    input_socket = FakeSocket("image")
    input_socket.connected_edges = [FakeEdge(start_socket=type("S", (), {"node": upstream, "name": "out"})())]

    node = FakeNode(node_type="dummy", inputs=[input_socket])

    def dummy_func(image):
        assert image.shape == (3, 3)
        return image + 1

    library_def = {
        "dummy": {
            "outputs": ["out"],
            "executable": dummy_func,
        }
    }

    result = worker.execute_node_logic(node, library_def)
    _, meta = result["out"]
    assert meta["axes"] == "ZYX"