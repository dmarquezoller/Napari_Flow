import numpy as np
import dask.array as da
import pytest
import skimage.exposure
import sys
import types

import napari_flow_editor.flow_nodes.decorator as dispatch_decorator
from napari_flow_editor.flow_nodes.decorator import (
    pop_dispatch_context,
    push_dispatch_context,
)
from napari_flow_editor.flow_nodes.exposure import (
    adjust_gamma,
    adjust_log,
    adjust_sigmoid,
    rescale_intensity,
)


EXPOSURE_CASES = [
    (
        adjust_gamma,
        skimage.exposure.adjust_gamma,
        {"gamma": 0.8, "gain": 1.2},
        {"gamma": 0.8, "gain": 1.2},
    ),
    (
        adjust_log,
        skimage.exposure.adjust_log,
        {"gain": 1.4, "inv": False},
        {"gain": 1.4, "inv": False},
    ),
    (
        adjust_log,
        skimage.exposure.adjust_log,
        {"gain": 1.1, "inv": True},
        {"gain": 1.1, "inv": True},
    ),
    (
        adjust_sigmoid,
        skimage.exposure.adjust_sigmoid,
        {"cut_off": 0.4, "gain": 6.0, "inv": False},
        {"cutoff": 0.4, "gain": 6.0, "inv": False},
    ),
    (
        adjust_sigmoid,
        skimage.exposure.adjust_sigmoid,
        {"cut_off": 0.6, "gain": 4.0, "inv": True},
        {"cutoff": 0.6, "gain": 4.0, "inv": True},
    ),
    (
        rescale_intensity,
        skimage.exposure.rescale_intensity,
        {"in_min": 0.1, "in_max": 0.9, "out_min": 0.0, "out_max": 1.0},
        {"in_range": (0.1, 0.9), "out_range": (0.0, 1.0)},
    ),
]


def _image(shape=(32, 40)):
    y = np.linspace(0.05, 0.95, shape[-2], dtype=np.float32)
    x = np.linspace(0.1, 0.9, shape[-1], dtype=np.float32)
    grid = (y[:, None] + x[None, :]) * 0.5
    if len(shape) == 2:
        return grid.astype(np.float32)
    return np.broadcast_to(grid, shape).copy().astype(np.float32)


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
    """Keep exposure dispatch tests deterministic on CUDA-capable machines."""
    monkeypatch.setattr(dispatch_decorator, "_try_import_cupy", lambda: None)


@pytest.mark.parametrize("node_func, ref_func, node_kwargs, ref_kwargs", EXPOSURE_CASES)
def test_exposure_numpy_promotes_to_lazy_dask_and_matches_skimage(
    node_func, ref_func, node_kwargs, ref_kwargs
):
    image = _image()

    result = node_func(image, **node_kwargs)

    assert isinstance(result, da.Array)
    computed = _to_numpy(result)
    expected = ref_func(image, **ref_kwargs).astype(np.float32, copy=False)
    assert computed.dtype == np.float32
    np.testing.assert_allclose(computed, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("node_func, ref_func, node_kwargs, ref_kwargs", EXPOSURE_CASES)
def test_exposure_dask_stays_lazy_and_matches_skimage(
    node_func, ref_func, node_kwargs, ref_kwargs
):
    image = _image()
    lazy = da.from_array(image, chunks=(16, 20))

    result = node_func(lazy, **node_kwargs)

    assert isinstance(result, da.Array)
    assert result.chunks == lazy.chunks
    computed = _to_numpy(result)
    expected = ref_func(image, **ref_kwargs).astype(np.float32, copy=False)
    assert computed.dtype == np.float32
    np.testing.assert_allclose(computed, expected, rtol=1e-6, atol=1e-6)


def test_exposure_pointwise_tyx_keeps_time_independent():
    image = _image((3, 24, 28))
    image[1] *= 0.5
    lazy = da.from_array(image, chunks=(1, 12, 14))
    token = push_dispatch_context({"metadata": {"axes": "TYX"}})
    try:
        result = adjust_gamma(lazy, gamma=1.5, gain=0.9)
    finally:
        pop_dispatch_context(token)

    assert isinstance(result, da.Array)
    assert result.chunks[0] == (1, 1, 1)
    expected = skimage.exposure.adjust_gamma(image, gamma=1.5, gain=0.9).astype(
        np.float32, copy=False
    )
    np.testing.assert_allclose(_to_numpy(result), expected, rtol=1e-6, atol=1e-6)


def test_exposure_pointwise_czyx_keeps_channel_independent():
    image = _image((2, 3, 20, 22))
    image[0] *= 0.75
    image[1] *= 1.05
    lazy = da.from_array(image, chunks=(1, 1, 10, 11))
    token = push_dispatch_context({"metadata": {"axes": "CZYX"}})
    try:
        result = adjust_log(lazy, gain=1.2, inv=False)
    finally:
        pop_dispatch_context(token)

    assert isinstance(result, da.Array)
    assert result.chunks[:2] == ((1, 1), (1, 1, 1))
    expected = skimage.exposure.adjust_log(image, gain=1.2, inv=False).astype(
        np.float32, copy=False
    )
    np.testing.assert_allclose(_to_numpy(result), expected, rtol=1e-6, atol=1e-6)


def test_exposure_auto_selects_dask_cuda_when_available(monkeypatch, capsys):
    def _asarray(x):
        return np.asarray(x)

    fake_cp = types.SimpleNamespace(
        ndarray=np.ndarray,
        asarray=_asarray,
        asnumpy=lambda x: np.asarray(x),
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)
        ),
    )
    fake_exposure = types.ModuleType("cucim.skimage.exposure")
    fake_exposure.adjust_gamma = skimage.exposure.adjust_gamma
    fake_exposure.adjust_log = skimage.exposure.adjust_log
    fake_exposure.rescale_intensity = skimage.exposure.rescale_intensity
    fake_exposure.adjust_sigmoid = skimage.exposure.adjust_sigmoid
    fake_skimage = types.ModuleType("cucim.skimage")
    fake_skimage.exposure = fake_exposure
    fake_cucim = types.ModuleType("cucim")
    fake_cucim.skimage = fake_skimage

    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    monkeypatch.setitem(sys.modules, "cucim", fake_cucim)
    monkeypatch.setitem(sys.modules, "cucim.skimage", fake_skimage)
    monkeypatch.setitem(sys.modules, "cucim.skimage.exposure", fake_exposure)
    monkeypatch.setattr(dispatch_decorator, "_try_import_cupy", lambda: fake_cp)

    image = _image()
    lazy = da.from_array(image, chunks=(16, 20))
    result = rescale_intensity(
        lazy,
        in_min=0.1,
        in_max=0.9,
        out_min=0.0,
        out_max=1.0,
    )

    assert isinstance(result, da.Array)
    assert "selected=dask_cuda" in capsys.readouterr().out
    expected = skimage.exposure.rescale_intensity(
        image,
        in_range=(0.1, 0.9),
        out_range=(0.0, 1.0),
    ).astype(np.float32, copy=False)
    np.testing.assert_allclose(_to_numpy(result), expected, rtol=1e-6, atol=1e-6)


def test_exposure_cucim_functions_match_skimage_when_available():
    cp = pytest.importorskip("cupy")
    cucim_exposure = pytest.importorskip("cucim.skimage.exposure")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device available")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    image = _image((16, 18))
    cases = [
        (
            cucim_exposure.adjust_gamma,
            skimage.exposure.adjust_gamma,
            {"gamma": 0.8, "gain": 1.2},
        ),
        (
            cucim_exposure.adjust_log,
            skimage.exposure.adjust_log,
            {"gain": 1.1, "inv": True},
        ),
        (
            cucim_exposure.rescale_intensity,
            skimage.exposure.rescale_intensity,
            {"in_range": (0.1, 0.9), "out_range": (0.0, 1.0)},
        ),
        (
            cucim_exposure.adjust_sigmoid,
            skimage.exposure.adjust_sigmoid,
            {"cutoff": 0.45, "gain": 5.0, "inv": False},
        ),
    ]
    for cuda_func, ref_func, kwargs in cases:
        out = cuda_func(cp.asarray(image), **kwargs)
        expected = ref_func(image, **kwargs)
        np.testing.assert_allclose(cp.asnumpy(out), expected, rtol=1e-6, atol=1e-6)
