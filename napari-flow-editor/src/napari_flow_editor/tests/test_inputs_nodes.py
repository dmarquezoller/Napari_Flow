import numpy as np
from napari_flow_editor.execution_engine import infer_layout_kind
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
    assert meta["layout_kind"] == "3d_image"
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
    assert "layout_kind" not in meta
    assert meta["source_layer"] == "LayerA"


def test_infer_layout_kind_mappings():
    assert infer_layout_kind("YX") == "2d_image"
    assert infer_layout_kind("YXC") == "2d_image_channels"
    assert infer_layout_kind("ZYX") == "3d_image"
    assert infer_layout_kind("ZYXC") == "3d_image_channels"
    assert infer_layout_kind("TYX") == "3d_timeline"
    assert infer_layout_kind("TYXC") == "3d_timeline_channels"
    assert infer_layout_kind("TZYX") == "4d_timeline"
    assert infer_layout_kind("TZYXC") == "4d_timeline_channels"
    assert infer_layout_kind("ABC") == "unknown_layout"
