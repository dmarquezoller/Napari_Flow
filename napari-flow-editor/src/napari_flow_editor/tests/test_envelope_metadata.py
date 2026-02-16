import numpy as np
from .conftest import FakeNode, FakeSocket, FakeEdge


def test_metadata_merge_priority(worker):
    # Upstream meta
    upstream = FakeNode("upstream")
    upstream.cached_results = {
        "out": (np.zeros((2, 2)), {"axes": "ZYX", "name": "Input"})
    }

    # Downstream node
    input_socket = FakeSocket("image")
    input_socket.connected_edges = [FakeEdge(start_socket=type("S", (), {"node": upstream, "name": "out"})())]
    node = FakeNode(node_type="dummy", inputs=[input_socket])

    def dummy_func(image):
        return (image, {"name": "Processed"})

    library_def = {
        "dummy": {
            "outputs": ["out"],
            "executable": dummy_func,
        }
    }

    result = worker.execute_node_logic(node, library_def)
    _, meta = result["out"]
    assert meta["axes"] == "ZYX"
    assert meta["name"] == "Processed"
    