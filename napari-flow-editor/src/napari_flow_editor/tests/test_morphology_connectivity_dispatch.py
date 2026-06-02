import inspect

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
    label_objects,
    remove_small_holes,
    remove_small_objects,
    skeletonize,
)

_SKIMAGE_MAX_SIZE_API = "max_size" in inspect.signature(
    skimage.morphology.remove_small_objects
).parameters


def _to_numpy(value):
    if isinstance(value, da.Array):
        value = value.compute()
    return np.asarray(value)


def _with_axes(axes, func):
    token = push_dispatch_context({"metadata": {"axes": axes}})
    try:
        return func()
    finally:
        pop_dispatch_context(token)


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    monkeypatch.setattr(dispatch_decorator, "_try_import_cupy", lambda: None)


def test_label_objects_zyx_rechunks_to_full_volume():
    """ZYX input: full spatial volume rechunked to one block, labels globally unique across Z."""
    image = np.zeros((3, 24, 26), dtype=bool)
    image[:, 3:8, 4:9] = True       # object spanning all Z planes
    image[1, 15:20, 16:22] = True   # single-plane object
    lazy = da.from_array(image, chunks=(2, 12, 13))

    result = _with_axes("ZYX", lambda: label_objects(lazy, connectivity=1))

    assert isinstance(result, da.Array)
    assert result.chunks == ((3,), (24,), (26,))
    expected = skimage.morphology.label(image, connectivity=1).astype(np.int32)
    assert _to_numpy(result).dtype == np.int32
    np.testing.assert_array_equal(_to_numpy(result), expected)


def test_label_objects_tyx_keeps_time_independent():
    """TYX input: each T frame labeled independently; labels restart per frame."""
    image = np.zeros((3, 24, 26), dtype=bool)
    image[:, 3:8, 4:9] = True
    image[1, 15:20, 16:22] = True
    lazy = da.from_array(image, chunks=(2, 12, 13))

    result = _with_axes("TYX", lambda: label_objects(lazy, connectivity=1))

    assert isinstance(result, da.Array)
    assert result.chunks[0] == (1, 1, 1)
    expected = np.stack(
        [skimage.morphology.label(frame, connectivity=1) for frame in image],
        axis=0,
    ).astype(np.int32)
    assert _to_numpy(result).dtype == np.int32
    np.testing.assert_array_equal(_to_numpy(result), expected)


def test_remove_small_objects_zyx_uses_full_volume_connectivity():
    """ZYX input: object spanning 3 planes has 12 pixels total; survives min_size=6."""
    image = np.zeros((3, 18, 20), dtype=bool)
    image[:, 7:9, 8:10] = True  # 4 px per plane, 12 px across volume
    lazy = da.from_array(image, chunks=(1, 9, 10))

    result = _with_axes(
        "ZYX",
        lambda: remove_small_objects(lazy, min_size=6, connectivity=1),
    )

    assert isinstance(result, da.Array)
    assert result.chunks == ((3,), (18,), (20,))
    size_kwarg = {"max_size": 5} if _SKIMAGE_MAX_SIZE_API else {"min_size": 6}
    expected = skimage.morphology.remove_small_objects(image, connectivity=1, **size_kwarg)
    np.testing.assert_array_equal(_to_numpy(result), expected)
    np.testing.assert_array_equal(_to_numpy(result), image)  # object is preserved


def test_remove_small_holes_zyx_full_volume():
    image = np.ones((2, 24, 26), dtype=bool)
    image[:, 4:9, 4:9] = False     # 25-pixel hole per plane
    image[1, 13:21, 13:21] = False  # 64-pixel hole in plane 1 only
    lazy = da.from_array(image, chunks=(1, 12, 13))

    result = _with_axes(
        "ZYX",
        lambda: remove_small_holes(lazy, area_threshold=30, connectivity=1),
    )

    assert isinstance(result, da.Array)
    assert result.chunks == ((2,), (24,), (26,))
    area_kwarg = {"max_size": 29} if _SKIMAGE_MAX_SIZE_API else {"area_threshold": 30}
    expected = skimage.morphology.remove_small_holes(image, connectivity=1, **area_kwarg)
    np.testing.assert_array_equal(_to_numpy(result), expected)


def test_skeletonize_tyx_keeps_time_independent():
    image = np.zeros((2, 24, 26), dtype=bool)
    image[:, 5:18, 10:14] = True
    image[1, 10:14, 5:21] = True
    lazy = da.from_array(image, chunks=(1, 12, 13))

    result = _with_axes("TYX", lambda: skeletonize(lazy))

    assert isinstance(result, da.Array)
    assert result.chunks[0] == (1, 1)
    expected = np.stack(
        [skimage.morphology.skeletonize(frame) for frame in image],
        axis=0,
    )
    np.testing.assert_array_equal(_to_numpy(result), expected)


def test_connectivity_nodes_promote_numpy_to_lazy_dask():
    """NumPy input is promoted to a lazy Dask array; TYX gives one chunk per frame."""
    image = np.zeros((2, 16, 18), dtype=bool)
    image[:, 4:8, 5:9] = True

    result = _with_axes("TYX", lambda: label_objects(image, connectivity=1))

    assert isinstance(result, da.Array)
    assert result.chunks[0] == (1, 1)
    expected = np.stack(
        [skimage.morphology.label(frame, connectivity=1) for frame in image],
        axis=0,
    ).astype(np.int32)
    np.testing.assert_array_equal(_to_numpy(result), expected)
