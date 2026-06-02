import sys
import types

import dask.array as da
import numpy as np
import pytest
import skimage.morphology

import napari_flow_editor.flow_nodes.decorator as dispatch_decorator
from napari_flow_editor.flow_nodes.decorator import (
    pop_dispatch_context,
    push_dispatch_context,
)
from napari_flow_editor.flow_nodes.morphology import (
    binary_closing,
    binary_opening,
    black_tophat,
    dilation,
    erosion,
    white_tophat,
)


MORPHOLOGY_CASES = [
    (dilation, skimage.morphology.dilation, {"radius": 2}, "preserve", 2),
    (erosion, skimage.morphology.erosion, {"radius": 2}, "preserve", 2),
    (binary_opening, skimage.morphology.opening, {"radius": 2}, "bool", 4),
    (binary_closing, skimage.morphology.closing, {"radius": 2}, "bool", 4),
    (white_tophat, skimage.morphology.white_tophat, {"radius": 3}, "preserve", 6),
    (black_tophat, skimage.morphology.black_tophat, {"radius": 3}, "preserve", 6),
]


def _image(shape=(64, 68)):
    rng = np.random.default_rng(1234)
    image = rng.normal(0.45, 0.18, shape).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    yy, xx = np.indices(shape[-2:])
    image[..., (yy - 20) ** 2 + (xx - 22) ** 2 < 8 ** 2] = 0.95
    image[..., (yy - 42) ** 2 + (xx - 45) ** 2 < 6 ** 2] = 0.05
    return image.astype(np.float32, copy=False)


def _mask(shape=(64, 68)):
    image = _image(shape)
    return image > 0.58


def _footprint_for(image, radius):
    if image.ndim >= 3:
        return skimage.morphology.ball(radius)
    return skimage.morphology.disk(radius)


def _reference(ref_func, image, radius):
    return ref_func(image, footprint=_footprint_for(image, radius))


def _input_for(dtype_kind):
    if dtype_kind == "bool":
        return _mask()
    return _image()


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
    """Keep morphology dispatch tests deterministic on CPU-only CI/dev machines."""
    monkeypatch.setattr(dispatch_decorator, "_try_import_cupy", lambda: None)


@pytest.mark.parametrize(
    "node_func, ref_func, kwargs, dtype_kind, _depth", MORPHOLOGY_CASES
)
def test_morphology_numpy_promotes_to_lazy_dask_and_matches_skimage(
    node_func, ref_func, kwargs, dtype_kind, _depth
):
    image = _input_for(dtype_kind)

    result = node_func(image, **kwargs)

    assert isinstance(result, da.Array)
    computed = _to_numpy(result)
    expected = _reference(ref_func, image, kwargs["radius"])
    np.testing.assert_array_equal(computed, expected)
    if dtype_kind == "bool":
        assert computed.dtype == np.bool_
    else:
        assert computed.dtype == image.dtype


@pytest.mark.parametrize(
    "node_func, ref_func, kwargs, dtype_kind, _depth", MORPHOLOGY_CASES
)
def test_morphology_dask_stays_lazy_and_matches_skimage(
    node_func, ref_func, kwargs, dtype_kind, _depth
):
    image = _input_for(dtype_kind)
    lazy = da.from_array(image, chunks=(29, 31))

    result = node_func(lazy, **kwargs)

    assert isinstance(result, da.Array)
    computed = _to_numpy(result)
    expected = _reference(ref_func, image, kwargs["radius"])
    np.testing.assert_array_equal(computed, expected)


def test_morphology_tyx_keeps_time_independent():
    image = np.stack([_mask(), np.roll(_mask(), shift=9, axis=-1)], axis=0)
    lazy = da.from_array(image, chunks=(1, 30, 32))
    token = push_dispatch_context({"metadata": {"axes": "TYX"}})
    try:
        result = binary_closing(lazy, radius=2)
    finally:
        pop_dispatch_context(token)

    assert isinstance(result, da.Array)
    assert result.chunks[0] == (1, 1)
    expected = np.stack(
        [
            skimage.morphology.closing(frame, footprint=skimage.morphology.disk(2))
            for frame in image
        ],
        axis=0,
    )
    np.testing.assert_array_equal(_to_numpy(result), expected)


def test_morphology_zyx_uses_3d_footprint():
    image = np.zeros((7, 31, 33), dtype=bool)
    image[3, 15, 16] = True
    lazy = da.from_array(image, chunks=(4, 16, 17))
    token = push_dispatch_context({"metadata": {"axes": "ZYX"}})
    try:
        result = dilation(lazy, radius=2)
    finally:
        pop_dispatch_context(token)

    expected = skimage.morphology.dilation(
        image,
        footprint=skimage.morphology.ball(2),
    )
    np.testing.assert_array_equal(_to_numpy(result), expected)


def test_morphology_rejects_invalid_radius():
    with pytest.raises(ValueError, match="radius must be >= 1"):
        dilation(_image(), radius=0)


def test_morphology_auto_selects_dask_cuda_when_available(monkeypatch, capsys):
    fake_cp = types.SimpleNamespace(
        ndarray=np.ndarray,
        asarray=lambda x: np.asarray(x),
        asnumpy=lambda x: np.asarray(x),
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)
        ),
    )
    fake_morphology = types.ModuleType("cucim.skimage.morphology")
    for name in [
        "dilation",
        "erosion",
        "opening",
        "closing",
        "white_tophat",
        "black_tophat",
    ]:
        setattr(fake_morphology, name, getattr(skimage.morphology, name))
    fake_skimage = types.ModuleType("cucim.skimage")
    fake_skimage.morphology = fake_morphology
    fake_cucim = types.ModuleType("cucim")
    fake_cucim.skimage = fake_skimage

    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    monkeypatch.setitem(sys.modules, "cucim", fake_cucim)
    monkeypatch.setitem(sys.modules, "cucim.skimage", fake_skimage)
    monkeypatch.setitem(sys.modules, "cucim.skimage.morphology", fake_morphology)
    monkeypatch.setattr(dispatch_decorator, "_try_import_cupy", lambda: fake_cp)

    image = _image()
    lazy = da.from_array(image, chunks=(29, 31))
    result = white_tophat(lazy, radius=3)

    assert isinstance(result, da.Array)
    assert "selected=dask_cuda" in capsys.readouterr().out
    expected = skimage.morphology.white_tophat(
        image,
        footprint=skimage.morphology.disk(3),
    )
    np.testing.assert_array_equal(_to_numpy(result), expected)


def test_morphology_cucim_functions_match_skimage_when_available():
    cp = pytest.importorskip("cupy")
    cucim_morphology = pytest.importorskip("cucim.skimage.morphology")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device available")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    image = _image((24, 26))
    mask = image > 0.58
    cases = [
        (cucim_morphology.dilation, skimage.morphology.dilation, image, 2),
        (cucim_morphology.erosion, skimage.morphology.erosion, image, 2),
        (cucim_morphology.opening, skimage.morphology.opening, mask, 2),
        (cucim_morphology.closing, skimage.morphology.closing, mask, 2),
        (cucim_morphology.white_tophat, skimage.morphology.white_tophat, image, 3),
        (cucim_morphology.black_tophat, skimage.morphology.black_tophat, image, 3),
    ]
    for cuda_func, ref_func, arr, radius in cases:
        footprint = skimage.morphology.disk(radius)
        gpu_out = cuda_func(cp.asarray(arr), footprint=cp.asarray(footprint))
        expected = ref_func(arr, footprint=footprint)
        np.testing.assert_array_equal(cp.asnumpy(gpu_out), expected)
