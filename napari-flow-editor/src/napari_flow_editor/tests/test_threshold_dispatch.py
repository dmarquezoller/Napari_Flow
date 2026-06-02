import sys
import types

import dask.array as da
import numpy as np
import pytest
import skimage.filters

import napari_flow_editor.flow_nodes.decorator as dispatch_decorator
from napari_flow_editor.flow_nodes.decorator import (
    pop_dispatch_context,
    push_dispatch_context,
)
from napari_flow_editor.flow_nodes.filters import (
    threshold_isodata,
    threshold_local,
    threshold_li,
    threshold_mean,
    threshold_minimum,
    threshold_niblack,
    threshold_otsu,
    threshold_sauvola,
    threshold_triangle,
    threshold_yen,
)


THRESHOLD_CASES = [
    (
        threshold_local,
        lambda image: image
        > skimage.filters.threshold_local(
            image,
            block_size=15,
            method="mean",
            offset=0.01,
            mode="reflect",
        ),
        {"block_size": 15, "method": "mean", "offset": 0.01, "mode": "reflect"},
    ),
    (
        threshold_niblack,
        lambda image: image
        > skimage.filters.threshold_niblack(image, window_size=15, k=0.2),
        {"window_size": 15, "k": 0.2},
    ),
    (
        threshold_sauvola,
        lambda image: image
        > skimage.filters.threshold_sauvola(image, window_size=15, k=0.2),
        {"window_size": 15, "k": 0.2},
    ),
]

GLOBAL_THRESHOLD_CASES = [
    (
        threshold_otsu,
        lambda image: image > skimage.filters.threshold_otsu(image, nbins=64),
        {"nbins": 64},
    ),
    (
        threshold_li,
        lambda image: image > skimage.filters.threshold_li(image),
        {},
    ),
    (
        threshold_mean,
        lambda image: image > skimage.filters.threshold_mean(image),
        {},
    ),
    (
        threshold_minimum,
        lambda image: image > skimage.filters.threshold_minimum(image, nbins=64),
        {"nbins": 64},
    ),
    (
        threshold_triangle,
        lambda image: image > skimage.filters.threshold_triangle(image, nbins=64),
        {"nbins": 64},
    ),
    (
        threshold_yen,
        lambda image: image > skimage.filters.threshold_yen(image, nbins=64),
        {"nbins": 64},
    ),
    (
        threshold_isodata,
        lambda image: image > skimage.filters.threshold_isodata(image, nbins=64),
        {"nbins": 64},
    ),
]


def _image(shape=(64, 64)):
    rng = np.random.default_rng(123)
    base = rng.random(shape, dtype=np.float32)
    gradient = np.linspace(0.0, 0.4, shape[-1], dtype=np.float32)
    return np.clip(base * 0.7 + gradient, 0.0, 1.0).astype(np.float32)


def _bimodal_image(shape=(64, 64)):
    rng = np.random.default_rng(456)
    y, x = np.indices(shape)
    image = np.where(x < shape[-1] // 2, 0.22, 0.74).astype(np.float32)
    image += rng.normal(0.0, 0.025, shape).astype(np.float32)
    image += (y / max(1, shape[-2]) * 0.04).astype(np.float32)
    return np.clip(image, 0.0, 1.0)


def _to_numpy(value):
    if isinstance(value, da.Array):
        value = value.compute()
    try:
        import cupy as cp

        if isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
    except Exception:
        pass
    return np.asarray(value)


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    """Keep threshold dispatch tests deterministic on CPU-only CI/dev machines."""
    monkeypatch.setattr(dispatch_decorator, "_try_import_cupy", lambda: None)


@pytest.mark.parametrize("node_func, ref_func, kwargs", THRESHOLD_CASES)
def test_threshold_numpy_promotes_to_lazy_dask_and_matches_skimage(
    node_func, ref_func, kwargs
):
    image = _image()

    result = node_func(image, **kwargs)

    assert isinstance(result, da.Array)
    computed = _to_numpy(result)
    expected = ref_func(image)
    assert computed.dtype == np.bool_
    np.testing.assert_array_equal(computed, expected)


@pytest.mark.parametrize("node_func, ref_func, kwargs", THRESHOLD_CASES)
def test_threshold_dask_stays_lazy_and_matches_skimage(node_func, ref_func, kwargs):
    image = _image()
    lazy = da.from_array(image, chunks=(32, 32))

    result = node_func(lazy, **kwargs)

    assert isinstance(result, da.Array)
    assert result.chunks == lazy.chunks
    computed = _to_numpy(result)
    expected = ref_func(image)
    assert computed.dtype == np.bool_
    np.testing.assert_array_equal(computed, expected)


def test_threshold_tyx_keeps_time_independent():
    image = _image((3, 48, 52))
    image[1] = np.clip(image[1] * 0.5, 0.0, 1.0)
    lazy = da.from_array(image, chunks=(1, 24, 26))
    token = push_dispatch_context({"metadata": {"axes": "TYX"}})
    try:
        result = threshold_local(
            lazy,
            block_size=11,
            method="mean",
            offset=0.02,
            mode="reflect",
        )
    finally:
        pop_dispatch_context(token)

    assert isinstance(result, da.Array)
    assert result.chunks[0] == (1, 1, 1)
    expected = image > skimage.filters.threshold_local(
        image,
        block_size=(1, 11, 11),
        method="mean",
        offset=0.02,
        mode="reflect",
    )
    np.testing.assert_array_equal(_to_numpy(result), expected)


@pytest.mark.parametrize(
    "node_func, kwargs",
    [
        (threshold_local, {"block_size": 10}),
        (threshold_niblack, {"window_size": 10}),
        (threshold_sauvola, {"window_size": 10}),
    ],
)
def test_threshold_rejects_even_window_sizes(node_func, kwargs):
    with pytest.raises(ValueError, match="odd integer"):
        node_func(_image(), **kwargs)


@pytest.mark.parametrize("node_func, ref_func, kwargs", GLOBAL_THRESHOLD_CASES)
def test_global_threshold_numpy_matches_skimage(node_func, ref_func, kwargs):
    image = _bimodal_image()

    result = node_func(image, **kwargs)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result, ref_func(image))


@pytest.mark.parametrize("node_func, ref_func, kwargs", GLOBAL_THRESHOLD_CASES)
def test_global_threshold_dask_returns_lazy_mask_and_matches_skimage(
    node_func, ref_func, kwargs
):
    image = _bimodal_image()
    lazy = da.from_array(image, chunks=(16, 16))

    result = node_func(lazy, **kwargs)

    assert isinstance(result, da.Array)
    assert result.chunks == lazy.chunks
    computed = _to_numpy(result)
    assert computed.dtype == np.bool_
    np.testing.assert_array_equal(computed, ref_func(image))


def test_global_threshold_tyx_keeps_time_independent():
    image = np.stack(
        [
            _bimodal_image((40, 42)),
            np.clip(_bimodal_image((40, 42)) * 0.55, 0.0, 1.0),
            np.clip(_bimodal_image((40, 42)) * 1.35, 0.0, 1.0),
        ],
        axis=0,
    ).astype(np.float32)
    lazy = da.from_array(image, chunks=(1, 20, 21))
    token = push_dispatch_context({"metadata": {"axes": "TYX"}})
    try:
        result = threshold_mean(lazy)
    finally:
        pop_dispatch_context(token)

    assert isinstance(result, da.Array)
    assert result.chunks[0] == (1, 1, 1)
    expected = np.stack(
        [
            frame > skimage.filters.threshold_mean(frame)
            for frame in image
        ],
        axis=0,
    )
    np.testing.assert_array_equal(_to_numpy(result), expected)


def test_global_threshold_mean_uses_global_scalar_not_chunk_threshold(capsys):
    image = np.zeros((64, 64), dtype=np.float32)
    image[:, :32] = 0.1
    image[:, 32:] = 0.9
    lazy = da.from_array(image, chunks=(32, 32))

    result = threshold_mean(lazy)

    trace = capsys.readouterr().out
    assert "selected=cpu" in trace
    assert "selected=dask" not in trace
    assert "selected=dask_cuda" not in trace
    assert isinstance(result, da.Array)
    expected = image > skimage.filters.threshold_mean(image)
    np.testing.assert_array_equal(_to_numpy(result), expected)


def test_threshold_auto_selects_dask_cuda_when_available(monkeypatch, capsys):
    fake_cp = types.SimpleNamespace(
        ndarray=np.ndarray,
        asarray=lambda x: np.asarray(x),
        asnumpy=lambda x: np.asarray(x),
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)
        ),
    )
    fake_filters = types.ModuleType("cucim.skimage.filters")
    fake_filters.threshold_local = skimage.filters.threshold_local
    fake_filters.threshold_niblack = skimage.filters.threshold_niblack
    fake_filters.threshold_sauvola = skimage.filters.threshold_sauvola
    fake_skimage = types.ModuleType("cucim.skimage")
    fake_skimage.filters = fake_filters
    fake_cucim = types.ModuleType("cucim")
    fake_cucim.skimage = fake_skimage

    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    monkeypatch.setitem(sys.modules, "cucim", fake_cucim)
    monkeypatch.setitem(sys.modules, "cucim.skimage", fake_skimage)
    monkeypatch.setitem(sys.modules, "cucim.skimage.filters", fake_filters)
    monkeypatch.setattr(dispatch_decorator, "_try_import_cupy", lambda: fake_cp)

    image = _image()
    lazy = da.from_array(image, chunks=(32, 32))
    result = threshold_local(
        lazy,
        block_size=15,
        method="mean",
        offset=0.01,
        mode="reflect",
    )

    assert isinstance(result, da.Array)
    assert "selected=dask_cuda" in capsys.readouterr().out
    expected = image > skimage.filters.threshold_local(
        image,
        block_size=15,
        method="mean",
        offset=0.01,
        mode="reflect",
    )
    np.testing.assert_array_equal(_to_numpy(result), expected)


def test_global_threshold_missing_cucim_function_falls_back_to_cpu(monkeypatch, capsys):
    fake_cp = types.SimpleNamespace(
        ndarray=np.ndarray,
        asarray=lambda x: np.asarray(x),
        asnumpy=lambda x: np.asarray(x),
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)
        ),
    )
    fake_filters = types.ModuleType("cucim.skimage.filters")
    fake_skimage = types.ModuleType("cucim.skimage")
    fake_skimage.filters = fake_filters
    fake_cucim = types.ModuleType("cucim")
    fake_cucim.skimage = fake_skimage

    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    monkeypatch.setitem(sys.modules, "cucim", fake_cucim)
    monkeypatch.setitem(sys.modules, "cucim.skimage", fake_skimage)
    monkeypatch.setitem(sys.modules, "cucim.skimage.filters", fake_filters)
    monkeypatch.setattr(dispatch_decorator, "_try_import_cupy", lambda: fake_cp)

    image = _bimodal_image()
    result = threshold_otsu(image, nbins=64)

    assert "selected=cpu" in capsys.readouterr().out
    expected = image > skimage.filters.threshold_otsu(image, nbins=64)
    np.testing.assert_array_equal(result, expected)


def test_threshold_cucim_functions_match_skimage_when_available():
    cp = pytest.importorskip("cupy")
    cucim_filters = pytest.importorskip("cucim.skimage.filters")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device available")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    image = _image((32, 34))
    cases = [
        (
            cucim_filters.threshold_local,
            skimage.filters.threshold_local,
            {"block_size": 15, "method": "mean", "offset": 0.01, "mode": "reflect"},
        ),
        (
            cucim_filters.threshold_niblack,
            skimage.filters.threshold_niblack,
            {"window_size": 15, "k": 0.2},
        ),
        (
            cucim_filters.threshold_sauvola,
            skimage.filters.threshold_sauvola,
            {"window_size": 15, "k": 0.2},
        ),
    ]
    for cuda_func, ref_func, kwargs in cases:
        threshold_gpu = cuda_func(cp.asarray(image), **kwargs)
        threshold_ref = ref_func(image, **kwargs)
        np.testing.assert_allclose(
            cp.asnumpy(threshold_gpu),
            threshold_ref,
            rtol=1e-5,
            atol=1e-6,
        )


def test_global_threshold_cucim_functions_match_skimage_when_available():
    cp = pytest.importorskip("cupy")
    cucim_filters = pytest.importorskip("cucim.skimage.filters")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device available")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    image = _bimodal_image((32, 34))
    cases = [
        ("threshold_otsu", {"nbins": 64}),
        ("threshold_li", {}),
        ("threshold_minimum", {"nbins": 64}),
        ("threshold_triangle", {"nbins": 64}),
        ("threshold_yen", {"nbins": 64}),
        ("threshold_isodata", {"nbins": 64}),
    ]
    available = [
        (name, kwargs)
        for name, kwargs in cases
        if hasattr(cucim_filters, name)
    ]
    if not available:
        pytest.skip("No cuCIM global threshold functions available")

    for name, kwargs in available:
        cuda_func = getattr(cucim_filters, name)
        ref_func = getattr(skimage.filters, name)
        threshold_gpu = cuda_func(cp.asarray(image), **kwargs)
        if isinstance(threshold_gpu, cp.ndarray):
            threshold_gpu = cp.asnumpy(threshold_gpu)
        threshold_gpu = float(np.asarray(threshold_gpu))
        threshold_ref = float(ref_func(image, **kwargs))
        np.testing.assert_allclose(
            threshold_gpu,
            threshold_ref,
            rtol=1e-5,
            atol=1e-6,
        )
