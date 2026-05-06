import numpy as np
import dask.array as da
from dask import delayed
import skimage.filters
import sys
import types

from napari_flow_editor.flow_nodes.decorator import (
    dispatch,
    pop_dispatch_context,
    push_dispatch_context,
)
from .conftest import FakeEdge, FakeNode, FakeSocket, unpack_execute_result


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

    result, _ = unpack_execute_result(worker.execute_node_logic(node, library_def))
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
        backend="dask",
    )

    assert isinstance(out, da.Array)
    assert calls["default"] == 0
    assert calls["dask"] == 1


def test_dispatch_auto_promotes_small_numpy_to_dask_by_default():
    calls = {"default": 0, "dask": 0}
    image = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)

    def default_func(x):
        calls["default"] += 1
        return x + 1

    def dask_func(x):
        calls["dask"] += 1
        return x + 5

    out = dispatch(
        default=default_func,
        dask_func=dask_func,
        args=(image,),
        kwargs={},
        backend="auto",
    )

    assert isinstance(out, da.Array)
    np.testing.assert_allclose(out.compute(), image + 5)
    assert calls["default"] == 0
    assert calls["dask"] == 1


def test_dispatch_auto_promotes_chunkable_numpy_to_dask_by_default():
    calls = {"default": 0}
    image = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)

    def default_func(x):
        calls["default"] += 1
        return x + 1

    out = dispatch(
        default=default_func,
        args=(image,),
        kwargs={},
        backend="auto",
        dask_strategy="pointwise",
        dask_output_dtype=np.float32,
        numpy_to_dask_chunks=(4, 4),
    )

    assert isinstance(out, da.Array)
    assert out.numblocks == (4, 4)
    assert calls["default"] == 0
    np.testing.assert_allclose(out.compute(), image + 1)
    assert calls["default"] == out.npartitions


def test_dispatch_auto_uses_dask_for_lazy_input():
    image = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    dask_image = da.from_array(image, chunks=(4, 4))

    out = dispatch(
        default=lambda x: x + 1,
        args=(dask_image,),
        kwargs={},
        backend="auto",
        dask_strategy="pointwise",
        dask_output_dtype=np.float32,
    )

    assert isinstance(out, da.Array)
    np.testing.assert_allclose(out.compute(), image + 1)


def test_dispatch_dask_pipeline_stays_lazy_until_compute():
    computed = []

    @delayed
    def make_source():
        computed.append("source")
        return np.arange(16, dtype=np.float32).reshape(4, 4)

    image = da.from_delayed(make_source(), shape=(4, 4), dtype=np.float32)

    first = dispatch(
        default=lambda x: x + 1,
        args=(image,),
        kwargs={},
        backend="auto",
        dask_strategy="pointwise",
        dask_output_dtype=np.float32,
    )
    second = dispatch(
        default=lambda x: x * 2,
        args=(first,),
        kwargs={},
        backend="auto",
        dask_strategy="pointwise",
        dask_output_dtype=np.float32,
    )

    assert isinstance(first, da.Array)
    assert isinstance(second, da.Array)
    assert computed == []

    np.testing.assert_allclose(second.compute(), (np.arange(16).reshape(4, 4) + 1) * 2)
    assert computed == ["source"]


def test_dispatch_spatial_auto_chunks_respect_tyx_layout():
    image = np.zeros((3, 256, 256), dtype=np.float32)

    out = _run_with_metadata(
        {"axes": "TYX", "layout_kind": "3d_timeline"},
        lambda: dispatch(
            default=lambda x: x + 1,
            args=(image,),
            kwargs={},
            backend="dask",
            dask_strategy="pointwise",
            dask_output_dtype=np.float32,
            numpy_to_dask_chunks="spatial_auto",
            dask_target_chunk_mb=0.0625,
            layout_policy="full_nd",
        ),
    )

    assert isinstance(out, da.Array)
    assert out.chunks[0] == (1, 1, 1)
    assert out.chunks[1] == (128, 128)
    assert out.chunks[2] == (128, 128)


def test_dispatch_auto_prefers_dask_cuda_for_numpy_when_available(monkeypatch):
    image = np.arange(64, dtype=np.float32).reshape(8, 8)

    fake_cp = types.SimpleNamespace(
        ndarray=np.ndarray,
        asarray=lambda x: np.asarray(x),
        asnumpy=lambda x: np.asarray(x),
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    out = dispatch(
        default=lambda x: x + 1,
        cuda_func=lambda x: x + 5,
        args=(image,),
        kwargs={},
        backend="auto",
        dask_strategy="pointwise",
        dask_output_dtype=np.float32,
        numpy_to_dask_chunks=(4, 4),
    )

    assert isinstance(out, da.Array)
    np.testing.assert_allclose(out.compute(), image + 5)


def test_dispatch_cuda_fallback_logs_message_when_gpu_unavailable(capsys, monkeypatch):
    image = np.arange(16, dtype=np.float32).reshape(4, 4)

    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    out = dispatch(
        default=lambda x: x + 1,
        cuda_func=lambda x: x + 5,
        args=(image,),
        kwargs={},
        backend="cuda",
    )

    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, image + 1)
    captured = capsys.readouterr().out
    assert "requested=cuda, selected=cpu" in captured


def test_dispatch_dask_cuda_runs_gpu_adapter_per_block(monkeypatch):
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    dask_image = da.from_array(image, chunks=(4, 4))

    fake_cp = types.SimpleNamespace(
        ndarray=np.ndarray,
        asarray=lambda x: np.asarray(x),
        asnumpy=lambda x: np.asarray(x),
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    out = dispatch(
        default=lambda x: x + 1,
        cuda_func=lambda x: x + 7,
        args=(dask_image,),
        kwargs={},
        backend="dask_cuda",
        dask_strategy="pointwise",
        dask_output_dtype=np.float32,
    )

    assert isinstance(out, da.Array)
    np.testing.assert_allclose(out.compute(), image + 7)


def test_dispatch_dask_cuda_neighborhood_keeps_gpu_blocks_until_compute(monkeypatch):
    class FakeCupyArray(np.ndarray):
        pass

    def fake_asarray(x):
        return np.asarray(x).view(FakeCupyArray)

    def fail_asnumpy(x):
        raise AssertionError("dask_cuda should not convert blocks back to NumPy")

    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    dask_image = da.from_array(image, chunks=(4, 4))
    fake_cp = types.SimpleNamespace(
        ndarray=FakeCupyArray,
        asarray=fake_asarray,
        asnumpy=fail_asnumpy,
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)

    out = dispatch(
        default=lambda x, sigma=0, mode="nearest": x + 1,
        cuda_func=lambda x, sigma=0, mode="nearest": x + 7,
        args=(dask_image,),
        kwargs={"sigma": 1, "mode": "nearest"},
        backend="dask_cuda",
        dask_strategy="neighborhood",
        dask_halo_from_param="sigma",
        dask_boundary_from_param="mode",
        dask_output_dtype=np.float32,
        cuda_output_dtype=np.float32,
    )

    assert isinstance(out, da.Array)
    assert isinstance(out._meta, FakeCupyArray)
    result = out.compute()
    np.testing.assert_allclose(np.asarray(result), image + 7)


def test_dispatch_auto_prefers_dask_cuda_for_lazy_when_available(monkeypatch):
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    dask_image = da.from_array(image, chunks=(4, 4))

    fake_cp = types.SimpleNamespace(
        ndarray=np.ndarray,
        asarray=lambda x: np.asarray(x),
        asnumpy=lambda x: np.asarray(x),
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    out = dispatch(
        default=lambda x: x + 1,
        cuda_func=lambda x: x + 7,
        args=(dask_image,),
        kwargs={},
        backend="auto",
        dask_strategy="pointwise",
        dask_output_dtype=np.float32,
    )

    assert isinstance(out, da.Array)
    np.testing.assert_allclose(out.compute(), image + 7)


def test_dispatch_cuda_function_mapping_uses_selected_node_params(monkeypatch):
    image = np.arange(16, dtype=np.float32).reshape(4, 4)

    fake_cp = types.SimpleNamespace(
        ndarray=np.ndarray,
        asarray=lambda x: np.asarray(x),
        asnumpy=lambda x: np.asarray(x),
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)

    def cpu_default(image, sigma=1.0, mode="nearest", preserve_range=True):
        return image + 1

    cuda_like = lambda image, sigma=1.0, mode="nearest": image + 9

    out = dispatch(
        default=cpu_default,
        cuda_function=cuda_like,
        cuda_arg_names=["image"],
        cuda_kwarg_names=["sigma", "mode"],
        args=(image,),
        kwargs={"sigma": 1.0, "mode": "nearest", "preserve_range": True},
        backend="cuda",
    )

    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, image + 9)


def test_dispatch_cuda_falls_back_when_user_kwargs_not_supported(capsys, monkeypatch):
    image = np.arange(16, dtype=np.float32).reshape(4, 4)

    fake_cp = types.SimpleNamespace(
        ndarray=np.ndarray,
        asarray=lambda x: np.asarray(x),
        asnumpy=lambda x: np.asarray(x),
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    # CUDA adapter does not accept "mode"
    def cpu_default(x, sigma=1.0, mode="nearest"):
        return x + 1

    out = dispatch(
        default=cpu_default,
        cuda_func=lambda x, sigma=1.0: x + 5,
        args=(image,),
        kwargs={"sigma": 1.0, "mode": "nearest"},
        backend="cuda",
    )

    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, image + 1)
    captured = capsys.readouterr().out
    assert "selected=cpu" in captured
    assert "CUDA cannot preserve current user parameters" in captured


def test_dispatch_dask_cuda_falls_back_for_unsupported_user_values(capsys, monkeypatch):
    image = np.arange(64, dtype=np.float32).reshape(8, 8)

    fake_cp = types.SimpleNamespace(
        ndarray=np.ndarray,
        asarray=lambda x: np.asarray(x),
        asnumpy=lambda x: np.asarray(x),
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)

    def cuda_like(x, sigma=1.0, mode="reflect"):
        if mode == "nearest":
            raise ValueError("nearest mode unsupported on this CUDA path")
        return x + 7

    def cpu_default(x, sigma=1.0, mode="nearest"):
        return x + 1

    out = dispatch(
        default=cpu_default,
        cuda_func=cuda_like,
        args=(image,),
        kwargs={"sigma": 1.0, "mode": "nearest"},
        backend="dask_cuda",
        dask_strategy="pointwise",
        dask_output_dtype=np.float32,
    )

    assert isinstance(out, da.Array)
    np.testing.assert_allclose(out.compute(), image + 1)
    captured = capsys.readouterr().out
    assert "selected=dask" in captured
    assert "CUDA cannot preserve current user parameters" in captured


def test_dispatch_can_promote_numpy_to_dask_with_strategy():
    image = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)

    out = dispatch(
        default=lambda x: x + 1,
        args=(image,),
        kwargs={},
        backend="dask",
        dask_strategy="pointwise",
        dask_output_dtype=np.float32,
    )

    assert isinstance(out, da.Array)
    np.testing.assert_allclose(out.compute(), image + 1)


def test_gaussian_dask_computes_partial_chunks_only():
    computed_chunks = []

    @delayed
    def mk_chunk(i, j):
        computed_chunks.append((i, j))
        return np.ones((64, 64), dtype=np.float32)

    # Build a lazy 8x8 tiled image (64 total chunks).
    grid = [
        [da.from_delayed(mk_chunk(i, j), shape=(64, 64), dtype=np.float32) for j in range(8)]
        for i in range(8)
    ]
    image = da.block(grid)

    out = dispatch(
        default=skimage.filters.gaussian,
        args=(image,),
        kwargs={"sigma": 1.0, "mode": "reflect", "preserve_range": True},
        dask_strategy="neighborhood",
        dask_halo_from_param="sigma",
        dask_boundary_from_param="mode",
        dask_output_dtype=np.float64,
    )

    # Graph build should remain lazy: no source chunk executed yet.
    assert computed_chunks == []
    assert isinstance(out, da.Array)

    # Compute a tiny ROI; only a subset of chunks should materialize.
    _ = out[:64, :64].mean().compute()
    assert 0 < len(computed_chunks) < 64


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


def test_dispatch_dask_options_can_override_map_overlap_depth_and_boundary():
    image = np.random.default_rng(3).random((4, 50, 50), dtype=np.float32)
    dask_image = da.from_array(image, chunks=(1, 25, 25))

    out = _run_with_metadata(
        {"axes": "TYX", "layout_kind": "3d_timeline"},
        lambda: dispatch(
            default=skimage.filters.gaussian,
            args=(dask_image,),
            kwargs={"sigma": 1.0, "mode": "nearest", "preserve_range": True},
            dask_options={
                "strategy": "neighborhood",
                "output_dtype": np.float64,
                "independent_axes_param": "sigma",
                "map_overlap": {
                    "depth": {"t": 0, "y": 2, "x": 2},
                    "boundary": "none",
                },
            },
        ),
    )

    expected = da.map_overlap(
        skimage.filters.gaussian,
        dask_image,
        depth={0: 0, 1: 2, 2: 2},
        boundary="none",
        dtype=np.float64,
        sigma=(0.0, 1.0, 1.0),
        mode="nearest",
        preserve_range=True,
    )

    assert isinstance(out, da.Array)
    assert out.numblocks == (4, 2, 2)
    np.testing.assert_allclose(out.compute(), expected.compute(), rtol=1e-6, atol=1e-6)


def test_dispatch_dask_options_support_callable_depth_for_shape_adaptation():
    image = np.random.default_rng(4).random((6, 96, 96), dtype=np.float32)
    dask_image = da.from_array(image, chunks=(1, 24, 24))

    def adaptive_depth(sample_arr, axes, kwargs_in):
        sigma = float(kwargs_in.get("sigma", 1.0))
        base = max(1, int(np.ceil(2.0 * sigma)))
        depth = {"t": 0}
        if isinstance(axes, str) and len(axes) == sample_arr.ndim:
            for axis_name in ("y", "x"):
                axis_token = axis_name.upper()
                if axis_token not in axes:
                    continue
                axis_i = axes.index(axis_token)
                chunk_size = int(sample_arr.chunks[axis_i][0])
                depth[axis_name] = min(base, max(1, chunk_size // 4))
        return depth

    out = _run_with_metadata(
        {"axes": "TYX", "layout_kind": "3d_timeline"},
        lambda: dispatch(
            default=skimage.filters.gaussian,
            args=(dask_image,),
            kwargs={"sigma": 2.0, "mode": "nearest", "preserve_range": True},
            dask_options={
                "strategy": "neighborhood",
                "output_dtype": np.float64,
                "independent_axes_param": "sigma",
                "map_overlap": {
                    "depth": adaptive_depth,
                    "boundary": "none",
                },
            },
        ),
    )

    assert isinstance(out, da.Array)
    assert out.shape == image.shape
    assert out.numblocks == (6, 4, 4)
    _ = out.compute()


def test_dispatch_independent_axes_param_keeps_tyx_semantics_with_smaller_graph():
    image = np.random.default_rng(2).random((8, 96, 96), dtype=np.float32)
    dask_image = da.from_array(image, chunks=(1, 32, 32))

    # Baseline behavior: explicit per-time slicing (time_policy='independent').
    out_split = _run_with_metadata(
        {"axes": "TYX", "layout_kind": "3d_timeline"},
        lambda: dispatch(
            default=skimage.filters.gaussian,
            args=(dask_image,),
            kwargs={"sigma": 1.8, "mode": "nearest", "preserve_range": True},
            dask_strategy="neighborhood",
            dask_halo_from_param="sigma",
            dask_boundary_from_param="mode",
            dask_output_dtype=np.float64,
        ),
    )

    # Optimized behavior: run full array once, but zero independent axes in sigma.
    out_vectorized = _run_with_metadata(
        {"axes": "TYX", "layout_kind": "3d_timeline"},
        lambda: dispatch(
            default=skimage.filters.gaussian,
            args=(dask_image,),
            kwargs={"sigma": 1.8, "mode": "nearest", "preserve_range": True},
            dask_strategy="neighborhood",
            dask_halo_from_param="sigma",
            dask_boundary_from_param="mode",
            independent_axes_param="sigma",
            dask_output_dtype=np.float64,
        ),
    )

    assert isinstance(out_split, da.Array)
    assert isinstance(out_vectorized, da.Array)
    np.testing.assert_allclose(
        out_vectorized.compute(),
        out_split.compute(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert len(out_vectorized.__dask_graph__()) < len(out_split.__dask_graph__())


def test_dispatch_tyx_metadata_with_trailing_channel_does_not_blur_time():
    image = np.zeros((3, 32, 32, 1), dtype=np.float32)
    image[1, 10:22, 10:22, 0] = 100.0
    dask_image = da.from_array(image, chunks=(1, 16, 16, 1))

    out = _run_with_metadata(
        {"axes": "TYX", "layout_kind": "3d_timeline"},
        lambda: dispatch(
            default=skimage.filters.gaussian,
            args=(dask_image,),
            kwargs={"sigma": 1.0, "mode": "nearest", "preserve_range": True},
            dask_strategy="neighborhood",
            dask_halo_from_param="sigma",
            dask_boundary_from_param="mode",
            independent_axes_param="sigma",
            dask_output_dtype=np.float64,
        ),
    )

    expected = skimage.filters.gaussian(
        image,
        sigma=(0.0, 1.0, 1.0, 0.0),
        mode="nearest",
        preserve_range=True,
    )

    assert isinstance(out, da.Array)
    assert out.shape == image.shape
    result = out.compute()
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    assert np.max(result[0]) == 0.0
    assert np.max(result[2]) == 0.0


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


def test_pyramid_per_level_only_computes_requested_level():
    computed_levels = []

    @delayed
    def mk_level(level_id, shape):
        computed_levels.append(level_id)
        return np.full(shape, fill_value=level_id, dtype=np.float32)

    shapes = [(8, 64, 64), (8, 32, 32), (8, 16, 16), (8, 8, 8)]
    pyramid = [
        da.from_delayed(mk_level(i, s), shape=s, dtype=np.float32)
        for i, s in enumerate(shapes)
    ]

    out = dispatch(
        default=skimage.filters.gaussian,
        args=(pyramid,),
        kwargs={"sigma": 1.0, "mode": "nearest", "preserve_range": True},
        dask_strategy="neighborhood",
        dask_halo_from_param="sigma",
        dask_boundary_from_param="mode",
        dask_output_dtype=np.float64,
        pyramid_strategy="per_level",
        pyramid_param_policy={"sigma": "fixed_world"},
    )

    assert isinstance(out, list)
    assert len(out) == 4
    assert computed_levels == []

    _ = out[3][0, :4, :4].mean().compute()
    assert computed_levels == [3]

    _ = out[1][0, :4, :4].mean().compute()
    assert computed_levels == [3, 1]


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
