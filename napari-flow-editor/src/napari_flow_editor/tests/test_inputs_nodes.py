import numpy as np
from .conftest import FakeNode


def test_get_layer_axes_injected(worker):
    node = FakeNode(
        node_type="get_layer",
        title="Get Layer",
        params={
            "layer_name": "LayerA",
            "axis_map": [{"d0": "Z", "d1": "Y", "d2": "X", "d3": "-", "d4": "-"}],
        },
    )

    result = worker.execute_node_logic(node, library_def={})
    data, meta = result["data_out"]

    assert isinstance(data, np.ndarray)
    assert meta["axes"] == "ZYX"
    assert meta["source_layer"] == "LayerA"


def test_get_layer_without_axes_map(worker):
    node = FakeNode(
        node_type="get_layer",
        title="Get Layer",
        params={"layer_name": "LayerA", "axis_map": []},
    )

    result = worker.execute_node_logic(node, library_def={})
    _, meta = result["data_out"]

    assert "axes" not in meta
    assert meta["source_layer"] == "LayerA"