import numpy as np
import dask.array as da
import skimage.feature

from napari_flow_editor.flow_nodes.decorator import (
    dispatch,
    pop_dispatch_context,
    push_dispatch_context,
)
from napari_flow_editor.flow_nodes.features import blob_dog


def _gaussian_blob(shape, center, sigma=5.0):
    grids = np.meshgrid(
        *[np.arange(n, dtype=np.float32) for n in shape],
        indexing="ij",
    )
    radius2 = 0.0
    for grid, coord in zip(grids, center):
        radius2 = radius2 + (grid - float(coord)) ** 2
    return np.exp(-0.5 * radius2 / float(sigma * sigma)).astype(np.float32)


def _with_axes(axes, fn):
    token = push_dispatch_context({"metadata": {"axes": axes}})
    try:
        return fn()
    finally:
        pop_dispatch_context(token)


def test_blob_dog_zyx_uses_z_halo_for_chunked_detection():
    image = _gaussian_blob((24, 96, 96), (12, 48, 48), sigma=5.0)
    dask_image = da.from_array(image, chunks=(8, 32, 32))

    coords, meta, layer_type = _with_axes(
        "ZYX",
        lambda: blob_dog(
            dask_image,
            min_sigma=3,
            max_sigma=10,
            threshold=0.05,
        ),
    )

    assert layer_type == "points"
    assert coords.shape == (1, 3)
    assert meta["size"].shape == (1,)
    assert meta["size"][0] > 0
    assert meta["face_color"] == [1.0, 1.0, 1.0, 0.0]
    assert meta["border_color"] == "white"
    assert meta["border_width_is_relative"] is False
    np.testing.assert_allclose(coords[0], [12, 48, 48], atol=1)


def test_blob_dog_tyx_treated_as_independent_axis():
    image = np.stack(
        [
            _gaussian_blob((96, 96), (48, 48), sigma=5.0),
            _gaussian_blob((96, 96), (48, 48), sigma=5.0),
            _gaussian_blob((96, 96), (48, 48), sigma=5.0),
        ],
        axis=0,
    )
    dask_image = da.from_array(image, chunks=(3, 32, 32))

    coords, meta, layer_type = _with_axes(
        "TYX",
        lambda: blob_dog(
            dask_image,
            min_sigma=3,
            max_sigma=10,
            threshold=0.05,
        ),
    )

    assert layer_type == "points"
    assert meta["size"].shape == (3,)
    assert np.all(meta["size"] > 0)
    rounded = {tuple(row) for row in np.round(coords).astype(int)}
    assert rounded == {(0, 48, 48), (1, 48, 48), (2, 48, 48)}


def test_chunked_delayed_points_forced_cpu_accepts_numpy_input():
    image = _gaussian_blob((64, 64), (32, 32), sigma=5.0)

    result = _with_axes(
        "YX",
        lambda: dispatch(
            default=skimage.feature.blob_dog,
            args=(image,),
            kwargs={"min_sigma": 3, "max_sigma": 10, "threshold": 0.05},
            output_type="points",
            backend="cpu",
            dask_options={
                "strategy": "chunked_delayed",
                "halo_from_param": "max_sigma",
                "halo_factor": 2.0,
                "output_coord_cols": slice(None, -1),
            },
        ),
    )

    coords, meta, layer_type = result
    assert layer_type == "points"
    assert meta == {}
    assert coords.shape == (1, 2)
    np.testing.assert_allclose(coords[0], [32, 32], atol=1)
