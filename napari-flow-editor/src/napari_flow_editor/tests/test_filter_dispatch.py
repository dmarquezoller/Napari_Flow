"""
Parametrized smoke tests for the Group 1-4 filter nodes updated with dispatch().

Strategy: one parametrized test covers all nodes in two passes:
  1. NumPy input  → output is float32, same shape
  2. Dask input   → result is lazy before compute(), float32, same shape after compute()

This is intentionally broad rather than per-node so that adding new dispatch-wrapped
filters only requires extending the NODES list, not writing new test functions.
"""
import numpy as np
import dask.array as da
import pytest

from napari_flow_editor.flow_nodes.filters import (
    butterworth,
    difference_of_gaussians,
    farid,
    farid_horizontal,
    farid_vertical,
    frangi,
    hessian,
    laplace,
    median_filter,
    meijering,
    prewitt_filter,
    roberts,
    sato,
    scharr,
    sobel_filter,
    unsharp_mask,
)

# ---------------------------------------------------------------------------
# Node registry: (function, extra_kwargs_overriding_defaults)
# All functions are called with a 2-D float32 image as first arg.
# ---------------------------------------------------------------------------

# Nodes that work with Dask input (neighborhood or pointwise strategy)
DASK_NODES = [
    (laplace,                {}),
    (laplace,                {"ksize": 5}),
    (median_filter,          {}),
    (median_filter,          {"radius": 3}),
    (sobel_filter,           {}),
    (prewitt_filter,         {}),
    (farid,                  {}),
    (farid_horizontal,       {}),
    (farid_vertical,         {}),
    (scharr,                 {}),
    (roberts,                {}),
    (unsharp_mask,           {}),
    (unsharp_mask,           {"radius": 3.0, "amount": 2.0}),
    (difference_of_gaussians, {"low_sigma": 1.0, "high_sigma": 2.0}),
    (hessian,                {}),
    (sato,                   {}),
]

# butterworth forces CPU (FFT), so only test with NumPy
NUMPY_ONLY_NODES = [
    (butterworth, {}),
    (butterworth, {"cutoff_frequency_ratio": 0.1, "order": 3}),
    (frangi, {}),
    (meijering, {}),
]

ALL_NUMPY_NODES = DASK_NODES + NUMPY_ONLY_NODES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(shape=(64, 64)):
    rng = np.random.default_rng(42)
    return rng.random(shape).astype(np.float32)


def _compute_result(result):
    """Compute a Dask result or return a NumPy result as-is."""
    if isinstance(result, da.Array):
        return result.compute()
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn, kwargs", ALL_NUMPY_NODES)
def test_filter_numpy_shape_and_dtype(fn, kwargs):
    """Every updated filter: correct shape and float32 output on NumPy input."""
    image = _make_image()
    result = _compute_result(fn(image, **kwargs))
    assert result.shape == image.shape, (
        f"{fn.__name__}: expected shape {image.shape}, got {result.shape}"
    )
    assert result.dtype == np.float32, (
        f"{fn.__name__}: expected float32, got {result.dtype}"
    )


@pytest.mark.parametrize("fn, kwargs", DASK_NODES)
def test_filter_dask_returns_lazy_then_correct(fn, kwargs):
    """Dispatch-wrapped filters: result is a lazy Dask array before compute()."""
    image = da.from_array(_make_image(), chunks=32)
    result = fn(image, **kwargs)
    assert isinstance(result, da.Array), (
        f"{fn.__name__}: expected a Dask array (lazy), got {type(result)}"
    )
    computed = result.compute()
    assert computed.shape == (64, 64), (
        f"{fn.__name__}: expected shape (64, 64), got {computed.shape}"
    )
    assert computed.dtype == np.float32, (
        f"{fn.__name__}: expected float32, got {computed.dtype}"
    )


@pytest.mark.parametrize("fn, kwargs", DASK_NODES)
def test_filter_numpy_promoted_to_dask(fn, kwargs):
    """NumPy input with a Dask strategy is promoted; result is still lazy."""
    image = _make_image()
    result = fn(image, **kwargs)
    # dispatch auto-promotes numpy to dask when a dask strategy is configured
    assert isinstance(result, da.Array), (
        f"{fn.__name__}: expected promotion to Dask array, got {type(result)}"
    )
