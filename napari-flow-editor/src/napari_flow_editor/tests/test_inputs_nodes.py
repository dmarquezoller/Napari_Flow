import numpy as np
from napari_flow_editor.execution_engine import ExecutionWorker, infer_layout_kind
from .conftest import FakeLayer, FakeNode, FakeViewer


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


def test_get_layer_normalizes_multiscale_wrapper():
    class MultiScaleData:
        __module__ = "napari.layers._multiscale_data"

        def __init__(self, levels):
            self._levels = list(levels)

        @property
        def shape(self):
            return self._levels[0].shape

        @property
        def shapes(self):
            return tuple(level.shape for level in self._levels)

        def __len__(self):
            return len(self._levels)

        def __getitem__(self, idx):
            return self._levels[idx]

    levels = [
        np.ones((64, 64), dtype=np.float32),
        np.ones((32, 32), dtype=np.float32),
        np.ones((16, 16), dtype=np.float32),
    ]
    viewer = FakeViewer(
        layers={
            "LayerA": FakeLayer(MultiScaleData(levels), metadata={"name": "LayerA"})
        }
    )
    worker = ExecutionWorker(scene=None, viewer=viewer)
    node = FakeNode(
        node_type="get_layer",
        title="Get Layer",
        params={"layer_name": "LayerA", "axis_map": []},
    )

    result = worker.execute_node_logic(node, library_def={})
    data, meta = result["data_out"]

    assert isinstance(data, list)
    assert len(data) == 3
    assert tuple(data[0].shape) == (64, 64)
    assert meta["source_layer"] == "LayerA"


def test_get_layer_includes_visual_metadata_from_layer_data_tuple():
    class FakeNapariImageLayer:
        def __init__(self):
            self.data = np.ones((16, 16), dtype=np.float32)
            self.metadata = {"custom_tag": "kept"}

        def as_layer_data_tuple(self):
            return (
                self.data,
                {
                    "name": "Blue Channel",
                    "colormap": "blue",
                    "contrast_limits": [0.0, 1.0],
                    "gamma": 1.0,
                },
                "image",
            )

    viewer = FakeViewer(layers={"LayerA": FakeNapariImageLayer()})
    worker = ExecutionWorker(scene=None, viewer=viewer)
    node = FakeNode(
        node_type="get_layer",
        title="Get Layer",
        params={"layer_name": "LayerA", "axis_map": []},
    )

    result = worker.execute_node_logic(node, library_def={})
    _, meta = result["data_out"]

    assert meta["name"] == "Blue Channel"
    assert meta["colormap"] == "blue"
    assert meta["custom_tag"] == "kept"
    assert meta["source_layer"] == "LayerA"


def test_get_layer_default_yx_does_not_override_existing_3d_axes_metadata():
    class FakeNapariImageLayer:
        def __init__(self):
            self.data = np.ones((10, 64, 64), dtype=np.float32)
            self.metadata = {}

        def as_layer_data_tuple(self):
            return (
                self.data,
                {
                    "name": "OME Layer",
                    "axes": "TYX",
                    "layout_kind": "3d_timeline",
                },
                "image",
            )

    viewer = FakeViewer(layers={"LayerA": FakeNapariImageLayer()})
    worker = ExecutionWorker(scene=None, viewer=viewer)
    node = FakeNode(
        node_type="get_layer",
        title="Get Layer",
        params={
            "layer_name": "LayerA",
            # UI default row; should not clobber existing TYX metadata
            "axis_map": [{"d0": "Y", "d1": "X", "d2": "-", "d3": "-", "d4": "-"}],
        },
    )

    result = worker.execute_node_logic(node, library_def={})
    _, meta = result["data_out"]

    assert meta["axes"] == "TYX"
    assert meta["layout_kind"] == "3d_timeline"


def test_get_layer_default_yx_infers_tyx_when_3d_axes_missing():
    class FakeNapariImageLayer:
        def __init__(self):
            self.data = np.ones((10, 64, 64), dtype=np.float32)
            self.metadata = {}

        def as_layer_data_tuple(self):
            return (
                self.data,
                {
                    "name": "NoAxesLayer",
                    # no "axes" provided by source metadata
                },
                "image",
            )

    viewer = FakeViewer(layers={"LayerA": FakeNapariImageLayer()})
    worker = ExecutionWorker(scene=None, viewer=viewer)
    node = FakeNode(
        node_type="get_layer",
        title="Get Layer",
        params={
            "layer_name": "LayerA",
            "axis_map": [{"d0": "Y", "d1": "X", "d2": "-", "d3": "-", "d4": "-"}],
        },
    )

    result = worker.execute_node_logic(node, library_def={})
    _, meta = result["data_out"]

    assert meta["axes"] == "TYX"
    assert meta["layout_kind"] == "3d_timeline"
    assert tuple(meta["axis_labels"]) == ("t", "y", "x")


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
