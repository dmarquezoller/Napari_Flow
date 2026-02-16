import numpy as np
from .conftest import FakeNode, FakeSocket, FakeEdge


def test_simple_pipeline(worker):
    # Node A (source)
    node_a = FakeNode("dummy_source")
    node_a.cached_results = {"out": (np.ones((2, 2)), {"axes": "YX"})}

    # Node B (consumer)
    input_socket = FakeSocket("image")
    input_socket.connected_edges = [FakeEdge(start_socket=type("S", (), {"node": node_a, "name": "out"})())]
    node_b = FakeNode("dummy_consumer", inputs=[input_socket])

    def func(image):
        return image * 2

    library_def = {
        "dummy_consumer": {
            "outputs": ["out"],
            "executable": func,
        }
    }

    result = worker.execute_node_logic(node_b, library_def)
    data, meta = result["out"]

    assert np.all(data == 2)
    assert meta["axes"] == "YX"