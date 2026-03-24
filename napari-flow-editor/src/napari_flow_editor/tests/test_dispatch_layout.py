import numpy as np
import dask.array as da
import skimage.filters

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


def test_dispatch_uses_axis_labels_metadata_when_axes_missing():
    calls = []
    image = np.arange(3 * 8 * 9, dtype=np.float32).reshape(3, 8, 9)

    def default_func(x):
        calls.append(x.shape)
        assert x.ndim == 2
        return x + 1

    out = _run_with_metadata(
        {"axis_labels": ("t", "y", "x")},
        lambda: dispatch(default=default_func, args=(image,), kwargs={}),
    )

    assert out.shape == image.shape
    assert len(calls) == 3
    assert all(shape == (8, 9) for shape in calls)


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


def test_dispatch_promotes_numpy_to_dask_when_backend_available():
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
    )

    assert isinstance(out, da.Array)
    assert calls["default"] == 0
    assert calls["dask"] == 1


def test_dispatch_can_promote_numpy_to_dask_with_strategy():
    image = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)

    out = dispatch(
        default=lambda x: x + 1,
        args=(image,),
        kwargs={},
        dask_strategy="pointwise",
        dask_output_dtype=np.float32,
    )

    assert isinstance(out, da.Array)
    np.testing.assert_allclose(out.compute(), image + 1)


def test_dispatch_neighborhood_matches_skimage_gaussian():
    image = np.random.default_rng(0).random((8, 64, 64), dtype=np.float32)
    dask_image = da.from_array(image, chunks=(2, 32, 32))

    out = dispatch(
        default=skimage.filters.gaussian,
        args=(dask_image,),
        kwargs={"sigma": 1.2, "mode": "reflect", "preserve_range": True},
        dask_strategy="neighborhood",
        dask_halo_from_param="sigma",
        dask_boundary_from_param="mode",
        dask_output_dtype=np.float64,
    )

    expected = skimage.filters.gaussian(
        image, sigma=1.2, mode="reflect", preserve_range=True
    )
    np.testing.assert_allclose(out.compute(), expected, rtol=1e-6, atol=1e-6)


def test_dispatch_neighborhood_handles_small_axis_and_extra_modes():
    image = np.random.default_rng(1).random((2, 32, 32), dtype=np.float32)
    dask_image = da.from_array(image, chunks=(1, 16, 16))

    for mode in ("wrap", "constant"):
        out = dispatch(
            default=skimage.filters.gaussian,
            args=(dask_image,),
            kwargs={"sigma": 1.2, "mode": mode, "preserve_range": True},
            dask_strategy="neighborhood",
            dask_halo_from_param="sigma",
            dask_boundary_from_param="mode",
            dask_output_dtype=np.float64,
        )
        assert isinstance(out, da.Array)
        arr = out.compute()
        assert arr.shape == image.shape


def test_dispatch_per_level_fixed_world_scales_sigma_by_pyramid_factor():
    pyramid = [
        np.ones((64, 64), dtype=np.float32),
        np.ones((32, 32), dtype=np.float32),
        np.ones((16, 16), dtype=np.float32),
    ]
    seen = []

    def capture_sigma(x, sigma=0.0):
        seen.append((x.shape, float(sigma)))
        return x

    out = dispatch(
        default=capture_sigma,
        args=(pyramid,),
        kwargs={"sigma": 4.0},
        pyramid_strategy="per_level",
        pyramid_param_policy={"sigma": "fixed_world"},
    )

    assert isinstance(out, list)
    assert len(out) == 3
    assert [shape for shape, _ in seen] == [(64, 64), (32, 32), (16, 16)]
    assert [round(v, 6) for _, v in seen] == [4.0, 2.0, 1.0]


def test_dispatch_per_level_fixed_world_ignores_unchanged_axis_for_scalar_sigma():
    pyramid = [
        np.ones((10, 128, 128), dtype=np.float32),
        np.ones((10, 64, 64), dtype=np.float32),
        np.ones((10, 32, 32), dtype=np.float32),
    ]
    seen = []

    def capture_sigma(x, sigma=0.0):
        seen.append((x.shape, float(sigma)))
        return x

    out = dispatch(
        default=capture_sigma,
        args=(pyramid,),
        kwargs={"sigma": 1.0},
        pyramid_strategy="per_level",
        pyramid_param_policy={"sigma": "fixed_world"},
    )

    assert isinstance(out, list)
    assert len(out) == 3
    assert [shape for shape, _ in seen] == [(10, 128, 128), (10, 64, 64), (10, 32, 32)]
    assert [round(v, 6) for _, v in seen] == [1.0, 0.5, 0.25]


def test_dispatch_per_level_without_policy_keeps_sigma_constant():
    pyramid = [
        np.ones((64, 64), dtype=np.float32),
        np.ones((32, 32), dtype=np.float32),
        np.ones((16, 16), dtype=np.float32),
    ]
    seen = []

    def capture_sigma(x, sigma=0.0):
        seen.append(float(sigma))
        return x

    dispatch(
        default=capture_sigma,
        args=(pyramid,),
        kwargs={"sigma": 4.0},
        pyramid_strategy="per_level",
    )

    assert [round(v, 6) for v in seen] == [4.0, 4.0, 4.0]


def test_dispatch_accepts_multiscale_sequence_like_container():
    class FakeMultiScaleData:
        def __init__(self, levels):
            self._levels = list(levels)

        def __len__(self):
            return len(self._levels)

        def __getitem__(self, idx):
            return self._levels[idx]

    pyramid = FakeMultiScaleData(
        [
            np.ones((64, 64), dtype=np.float32),
            np.ones((32, 32), dtype=np.float32),
            np.ones((16, 16), dtype=np.float32),
        ]
    )
    seen = []

    def capture_shape(x):
        seen.append(tuple(x.shape))
        return x

    out = dispatch(
        default=capture_shape,
        args=(pyramid,),
        kwargs={},
        pyramid_strategy="per_level",
    )

    assert isinstance(out, list)
    assert [tuple(o.shape) for o in out] == [(64, 64), (32, 32), (16, 16)]
    assert seen == [(64, 64), (32, 32), (16, 16)]


def test_dispatch_accepts_multiscale_container_inside_envelope():
    class FakeMultiScaleData:
        def __init__(self, levels):
            self._levels = list(levels)

        def __len__(self):
            return len(self._levels)

        def __getitem__(self, idx):
            return self._levels[idx]

    ms = FakeMultiScaleData(
        [
            np.ones((64, 64), dtype=np.float32),
            np.ones((32, 32), dtype=np.float32),
            np.ones((16, 16), dtype=np.float32),
        ]
    )
    envelope = (ms, {"axes": "YX"})
    seen = []

    def capture_shape(x):
        seen.append(tuple(x.shape))
        return x

    out = dispatch(
        default=capture_shape,
        args=(envelope,),
        kwargs={},
        pyramid_strategy="per_level",
    )

    assert isinstance(out, list)
    assert [tuple(o.shape) for o in out] == [(64, 64), (32, 32), (16, 16)]
    assert seen == [(64, 64), (32, 32), (16, 16)]


def test_dispatch_accepts_shape_bearing_multiscale_inside_envelope():
    class MultiScaleData:
        __module__ = "napari.layers._multiscale_data"

        def __init__(self, levels):
            self._levels = list(levels)

        @property
        def shape(self):
            # Some multiscale wrappers expose a shape-like attribute.
            return self._levels[0].shape

        def __len__(self):
            return len(self._levels)

        def __getitem__(self, idx):
            return self._levels[idx]

    ms = MultiScaleData(
        [
            np.ones((64, 64), dtype=np.float32),
            np.ones((32, 32), dtype=np.float32),
            np.ones((16, 16), dtype=np.float32),
        ]
    )
    envelope = (ms, {"axes": "YX"})
    seen = []

    def capture_shape(x):
        seen.append(tuple(x.shape))
        return x

    out = dispatch(
        default=capture_shape,
        args=(envelope,),
        kwargs={},
        pyramid_strategy="per_level",
    )

    assert isinstance(out, list)
    assert [tuple(o.shape) for o in out] == [(64, 64), (32, 32), (16, 16)]
    assert seen == [(64, 64), (32, 32), (16, 16)]
