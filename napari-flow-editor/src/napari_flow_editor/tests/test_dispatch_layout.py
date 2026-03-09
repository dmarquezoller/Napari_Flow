import numpy as np
import dask.array as da

from napari_flow_editor.flow_nodes.decorator import (
    dispatch,
    pop_dispatch_context,
    push_dispatch_context,
)
from .conftest import FakeEdge, FakeNode, FakeSocket


def _run_with_metadata(metadata, fn):
    token = push_dispatch_context({"metadata": metadata})
    try:
        return fn()
    finally:
        pop_dispatch_context(token)


def test_dispatch_tyx_runs_per_time_slice():
    calls = []
    image = np.arange(3 * 8 * 9, dtype=np.float32).reshape(3, 8, 9)

    def default_func(x):
        calls.append(x.shape)
        assert x.ndim == 2
        return x + 1

    out = _run_with_metadata(
        {"axes": "TYX", "layout_kind": "3d_timeline"},
        lambda: dispatch(default=default_func, args=(image,), kwargs={}),
    )

    assert out.shape == image.shape
    assert len(calls) == 3
    assert all(shape == (8, 9) for shape in calls)


def test_dispatch_tyxc_runs_per_time_and_channel_slice():
    calls = []
    image = np.arange(2 * 6 * 7 * 3, dtype=np.float32).reshape(2, 6, 7, 3)

    def default_func(x):
        calls.append(x.shape)
        assert x.ndim == 2
        return x

    out = _run_with_metadata(
        {"axes": "TYXC", "layout_kind": "3d_timeline_channels"},
        lambda: dispatch(default=default_func, args=(image,), kwargs={}),
    )

    assert out.shape == image.shape
    assert len(calls) == 6
    assert all(shape == (6, 7) for shape in calls)


def test_dispatch_zyx_runs_on_full_volume():
    calls = []
    image = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)

    def default_func(x):
        calls.append(x.shape)
        assert x.ndim == 3
        return x

    out = _run_with_metadata(
        {"axes": "ZYX", "layout_kind": "3d_image"},
        lambda: dispatch(default=default_func, args=(image,), kwargs={}),
    )

    assert out.shape == image.shape
    assert len(calls) == 1
    assert calls[0] == (4, 5, 6)


def test_worker_propagates_axes_metadata_to_dispatch(worker):
    upstream = FakeNode("get_layer")
    upstream.cached_results = {
        "out": (
            np.arange(2 * 10 * 11, dtype=np.float32).reshape(2, 10, 11),
            {"axes": "TYX", "layout_kind": "3d_timeline"},
        )
    }

    input_socket = FakeSocket("image")
    input_socket.connected_edges = [
        FakeEdge(start_socket=type("S", (), {"node": upstream, "name": "out"})())
    ]

    node = FakeNode(node_type="dummy", inputs=[input_socket])
    calls = []

    def per_time_slice(x):
        calls.append(x.shape)
        assert x.ndim == 2
        return x + 1

    def executable(image):
        return dispatch(default=per_time_slice, args=(image,), kwargs={})

    library_def = {
        "dummy": {
            "outputs": ["out"],
            "executable": executable,
        }
    }

    result = worker.execute_node_logic(node, library_def)
    out, meta = result["out"]

    assert out.shape == (2, 10, 11)
    assert len(calls) == 2
    assert all(shape == (10, 11) for shape in calls)
    assert meta["axes"] == "TYX"
    assert meta["layout_kind"] == "3d_timeline"


def test_dispatch_can_promote_large_numpy_to_dask_when_enabled():
    calls = {"default": 0, "dask": 0}
    image = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)

    def default_func(x):
        calls["default"] += 1
        return x

    def dask_func(x):
        calls["dask"] += 1
        assert isinstance(x, da.Array)
        return x

    out = dispatch(
        default=default_func,
        dask_func=dask_func,
        args=(image,),
        kwargs={},
        allow_dask_from_numpy=True,
        numpy_to_dask_min_bytes=0,
    )

    assert isinstance(out, da.Array)
    assert calls["default"] == 0
    assert calls["dask"] == 1
