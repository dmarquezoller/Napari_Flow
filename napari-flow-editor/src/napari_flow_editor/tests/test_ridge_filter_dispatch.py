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
    frangi,
    hessian,
    meijering,
    sato,
)


FULL_IMAGE_RIDGE_CASES = [
    (
        frangi,
        skimage.filters.frangi,
        {"sigma_range_low": 1.0, "sigma_range_high": 4.0, "sigma_step": 1.0},
        {"sigmas": (1.0, 2.0, 3.0), "alpha": 0.5, "beta": 0.5, "mode": "reflect"},
    ),
    (
        meijering,
        skimage.filters.meijering,
        {"sigmas_min": 1.0, "sigmas_max": 4.0, "sigmas_step": 1.0},
        {"sigmas": (1.0, 2.0, 3.0), "mode": "reflect", "black_ridges": True},
    ),
]


NEIGHBORHOOD_RIDGE_CASES = [
    (
        hessian,
        skimage.filters.hessian,
        {"sigmas_min": 1.0, "sigmas_max": 5.0},
        {"sigmas": (1.0, 3.0), "mode": "reflect", "black_ridges": True},
    ),
    (
        sato,
        skimage.filters.sato,
        {"sigmas_min": 1.0, "sigmas_max": 5.0},
        {"sigmas": (1.0, 3.0), "mode": "reflect", "black_ridges": True},
    ),
]


def _ridge_image(shape=(80, 84)):
    image = np.ones(shape, dtype=np.float32)
    yy, xx = np.indices(shape[-2:])
    image[..., np.abs(yy - shape[-2] // 2) <= 1] = 0.0
    image[..., np.abs(xx - shape[-1] // 2) <= 1] = 0.0
    image[..., np.abs((yy - xx) - 3) <= 1] = 0.15
    return image


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
    """Keep ridge dispatch tests deterministic on CPU-only CI/dev machines."""
    monkeypatch.setattr(dispatch_decorator, "_try_import_cupy", lambda: None)


@pytest.mark.parametrize(
    "node_func, ref_func, node_kwargs, ref_kwargs", NEIGHBORHOOD_RIDGE_CASES
)
def test_ridge_dask_matches_skimage_across_chunk_boundaries(
    node_func, ref_func, node_kwargs, ref_kwargs
):
    image = _ridge_image()
    lazy = da.from_array(image, chunks=(37, 41))

    result = node_func(lazy, **node_kwargs)

    assert isinstance(result, da.Array)
    computed = _to_numpy(result)
    expected = ref_func(image, **ref_kwargs).astype(np.float32, copy=False)
    assert computed.dtype == np.float32
    np.testing.assert_allclose(computed, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "node_func, ref_func, node_kwargs, ref_kwargs",
    FULL_IMAGE_RIDGE_CASES + NEIGHBORHOOD_RIDGE_CASES,
)
def test_ridge_numpy_matches_skimage(
    node_func, ref_func, node_kwargs, ref_kwargs
):
    image = _ridge_image()

    result = node_func(image, **node_kwargs)

    computed = _to_numpy(result)
    expected = ref_func(image, **ref_kwargs).astype(np.float32, copy=False)
    np.testing.assert_allclose(computed, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "node_func, ref_func, node_kwargs, ref_kwargs", FULL_IMAGE_RIDGE_CASES
)
def test_global_ridge_nodes_materialize_dask_input_intentionally(
    node_func, ref_func, node_kwargs, ref_kwargs
):
    image = _ridge_image()
    lazy = da.from_array(image, chunks=(37, 41))

    result = node_func(lazy, **node_kwargs)

    assert not isinstance(result, da.Array)
    computed = _to_numpy(result)
    expected = ref_func(image, **ref_kwargs).astype(np.float32, copy=False)
    np.testing.assert_allclose(computed, expected, rtol=2e-5, atol=2e-5)


def test_ridge_tyx_keeps_time_independent():
    image = np.stack(
        [
            _ridge_image((48, 52)),
            np.roll(_ridge_image((48, 52)), shift=7, axis=-1),
        ],
        axis=0,
    )
    lazy = da.from_array(image, chunks=(1, 24, 26))
    token = push_dispatch_context({"metadata": {"axes": "TYX"}})
    try:
        result = sato(lazy, sigmas_min=1.0, sigmas_max=4.0)
    finally:
        pop_dispatch_context(token)

    assert isinstance(result, da.Array)
    assert result.chunks[0] == (1, 1)
    expected = np.stack(
        [
            skimage.filters.sato(
                frame,
                sigmas=(1.0, 3.0),
                mode="reflect",
                black_ridges=True,
            )
            for frame in image
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    np.testing.assert_allclose(_to_numpy(result), expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "node_func, kwargs",
    [
        (frangi, {"sigma_range_low": 2.0, "sigma_range_high": 2.0}),
        (frangi, {"sigma_step": 0.0}),
        (hessian, {"sigmas_min": 4.0, "sigmas_max": 4.0}),
        (meijering, {"sigmas_step": 0.0}),
        (sato, {"sigmas_min": 4.0, "sigmas_max": 3.0}),
    ],
)
def test_ridge_rejects_invalid_sigma_ranges(node_func, kwargs):
    with pytest.raises(ValueError, match="sigmas"):
        node_func(_ridge_image(), **kwargs)


def test_ridge_auto_selects_dask_cuda_when_available(monkeypatch, capsys):
    fake_cp = types.SimpleNamespace(
        ndarray=np.ndarray,
        asarray=lambda x: np.asarray(x),
        asnumpy=lambda x: np.asarray(x),
        cuda=types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)
        ),
    )
    fake_filters = types.ModuleType("cucim.skimage.filters")
    fake_filters.hessian = skimage.filters.hessian
    fake_filters.sato = skimage.filters.sato
    fake_skimage = types.ModuleType("cucim.skimage")
    fake_skimage.filters = fake_filters
    fake_cucim = types.ModuleType("cucim")
    fake_cucim.skimage = fake_skimage

    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    monkeypatch.setitem(sys.modules, "cucim", fake_cucim)
    monkeypatch.setitem(sys.modules, "cucim.skimage", fake_skimage)
    monkeypatch.setitem(sys.modules, "cucim.skimage.filters", fake_filters)
    monkeypatch.setattr(dispatch_decorator, "_try_import_cupy", lambda: fake_cp)

    image = _ridge_image()
    lazy = da.from_array(image, chunks=(37, 41))
    result = sato(lazy, sigmas_min=1.0, sigmas_max=5.0)

    assert isinstance(result, da.Array)
    assert "selected=dask_cuda" in capsys.readouterr().out
    expected = skimage.filters.sato(
        image,
        sigmas=(1.0, 3.0),
        mode="reflect",
        black_ridges=True,
    ).astype(np.float32, copy=False)
    np.testing.assert_allclose(_to_numpy(result), expected, rtol=2e-5, atol=2e-5)


def test_ridge_cucim_functions_match_skimage_when_available():
    cp = pytest.importorskip("cupy")
    cucim_filters = pytest.importorskip("cucim.skimage.filters")
    missing = [
        name
        for name in ("hessian", "sato")
        if not hasattr(cucim_filters, name)
    ]
    if missing:
        pytest.skip(f"cuCIM missing ridge filters: {', '.join(missing)}")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device available")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    image = _ridge_image((32, 34))
    cases = [
        (
            cucim_filters.hessian,
            skimage.filters.hessian,
            {"sigmas": (1.0, 2.0), "black_ridges": True},
        ),
        (
            cucim_filters.sato,
            skimage.filters.sato,
            {"sigmas": (1.0, 2.0), "black_ridges": True},
        ),
    ]
    for cuda_func, ref_func, kwargs in cases:
        gpu_out = cuda_func(cp.asarray(image), **kwargs)
        expected = ref_func(image, **kwargs)
        np.testing.assert_allclose(
            cp.asnumpy(gpu_out),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )
