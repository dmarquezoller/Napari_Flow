import numpy as np

from napari_flow_editor.io import ome_zarr_writer as writer


class _FakeGroup:
    def __init__(self):
        self.attrs = {}


def test_infer_axes_prefers_metadata_for_ctzyx():
    arr = np.zeros((2, 160, 8, 64, 64), dtype=np.uint16)

    assert writer.infer_axes(arr, metadata={"axes": "CTZYX"}) == "ctzyx"


def test_auto_chunks_are_spatial_and_independent_axis_aware():
    shape = (2, 160, 851, 2048, 2048)

    assert writer.auto_chunks(shape, "ctzyx") == (1, 1, 32, 512, 512)


def test_save_ome_zarr_writes_ngff_metadata_without_forcing_test_io(monkeypatch):
    fake_group = _FakeGroup()
    to_zarr_calls = []

    def fake_open_group(path, mode):
        return fake_group

    def fake_to_zarr(array, url, component, overwrite, **kwargs):
        to_zarr_calls.append(
            {
                "shape": tuple(array.shape),
                "chunksize": tuple(array.chunksize),
                "url": url,
                "component": component,
                "overwrite": overwrite,
                "kwargs": kwargs,
            }
        )

    monkeypatch.setattr(writer, "_open_group", fake_open_group)
    monkeypatch.setattr(writer.da, "to_zarr", fake_to_zarr)

    arr = np.zeros((2, 3, 4), dtype=np.uint16)
    request = writer.OmeZarrSaveRequest(
        data=arr,
        path="/tmp/out.zarr",
        name="demo",
        metadata={"axes": "ZYX", "scale": (2.0, 0.5, 0.5)},
    )

    result = writer.save_ome_zarr(request)

    assert result.path == "/tmp/out.zarr"
    assert result.axes == "zyx"
    assert result.chunks == (2, 3, 4)
    assert result.levels == 1
    assert result.requested_levels == 1
    assert result.source_levels == 1
    assert to_zarr_calls == [
        {
            "shape": (2, 3, 4),
            "chunksize": (2, 3, 4),
            "url": "/tmp/out.zarr",
            "component": "0",
            "overwrite": True,
            "kwargs": {"zarr_format": 2},
        }
    ]
    multiscales = fake_group.attrs["multiscales"]
    assert multiscales[0]["version"] == "0.4"
    assert [axis["name"] for axis in multiscales[0]["axes"]] == ["z", "y", "x"]
    assert multiscales[0]["datasets"][0]["path"] == "0"
    assert multiscales[0]["datasets"][0]["coordinateTransformations"][0] == {
        "type": "scale",
        "scale": [2.0, 0.5, 0.5],
    }


def test_save_ome_zarr_reorders_ctzyx_to_tczyx(monkeypatch):
    fake_group = _FakeGroup()
    calls = []

    monkeypatch.setattr(writer, "_open_group", lambda path, mode: fake_group)

    def fake_to_zarr(array, url, component, overwrite, **kwargs):
        calls.append(
            {
                "shape": tuple(array.shape),
                "chunksize": tuple(array.chunksize),
                "component": component,
            }
        )

    monkeypatch.setattr(writer.da, "to_zarr", fake_to_zarr)

    arr = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)
    request = writer.OmeZarrSaveRequest(
        data=arr,
        path="/tmp/ctzyx.zarr",
        metadata={
            "axes": "CTZYX",
            "scale": (1.0, 2.0, 3.0, 4.0, 5.0),
            "contrast_limits": (10, 100),
            "colormap": "orange",
        },
    )

    result = writer.save_ome_zarr(request)

    assert result.axes == "tczyx"
    assert result.shape == (3, 2, 4, 5, 6)
    assert calls[0]["shape"] == (3, 2, 4, 5, 6)
    axes = fake_group.attrs["multiscales"][0]["axes"]
    assert [axis["name"] for axis in axes] == ["t", "c", "z", "y", "x"]
    scale = fake_group.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert scale == [2.0, 1.0, 3.0, 4.0, 5.0]
    assert fake_group.attrs["omero"]["channels"][0]["window"]["start"] == 10.0
    assert fake_group.attrs["omero"]["channels"][0]["window"]["end"] == 100.0
    assert fake_group.attrs["omero"]["channels"][0]["color"] == "FFA500"
    assert fake_group.attrs["omero"]["rdefs"] == {"model": "color"}
    assert result.omero_channels == 2


def test_omero_metadata_supports_per_channel_display_settings():
    omero = writer._omero_display_metadata(
        "demo",
        "tczyx",
        (1, 2, 4, 8, 8),
        {
            "contrast_limits": [[1, 10], [2, 20]],
            "colormap": ["red", "green"],
            "visible": False,
        },
    )

    assert omero["channels"][0]["window"]["start"] == 1.0
    assert omero["channels"][0]["window"]["end"] == 10.0
    assert omero["channels"][0]["color"] == "FF0000"
    assert omero["channels"][0]["active"] is False
    assert omero["channels"][1]["window"]["start"] == 2.0
    assert omero["channels"][1]["window"]["end"] == 20.0
    assert omero["channels"][1]["color"] == "00FF00"


def test_save_ome_zarr_generates_requested_yx_pyramid_levels(monkeypatch):
    fake_group = _FakeGroup()
    calls = []

    monkeypatch.setattr(writer, "_open_group", lambda path, mode: fake_group)

    def fake_to_zarr(array, url, component, overwrite, **kwargs):
        calls.append({"shape": tuple(array.shape), "component": component})

    monkeypatch.setattr(writer.da, "to_zarr", fake_to_zarr)

    arr = np.zeros((1, 1, 4, 8, 8), dtype=np.float32)
    request = writer.OmeZarrSaveRequest(
        data=arr,
        path="/tmp/pyramid.zarr",
        metadata={"axes": "TCZYX"},
        pyramid_levels=3,
    )

    result = writer.save_ome_zarr(request)

    assert result.levels == 3
    assert result.requested_levels == 3
    assert result.source_levels == 1
    assert calls == [
        {"shape": (1, 1, 4, 8, 8), "component": "0"},
        {"shape": (1, 1, 4, 4, 4), "component": "1"},
        {"shape": (1, 1, 4, 2, 2), "component": "2"},
    ]
    datasets = fake_group.attrs["multiscales"][0]["datasets"]
    assert datasets[1]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 1.0, 2.0, 2.0]
    assert datasets[2]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 1.0, 4.0, 4.0]


def test_save_ome_zarr_generates_requested_zyx_pyramid_levels(monkeypatch):
    fake_group = _FakeGroup()
    calls = []

    monkeypatch.setattr(writer, "_open_group", lambda path, mode: fake_group)

    def fake_to_zarr(array, url, component, overwrite, **kwargs):
        calls.append({"shape": tuple(array.shape), "component": component})

    monkeypatch.setattr(writer.da, "to_zarr", fake_to_zarr)

    arr = np.zeros((1, 1, 8, 8, 8), dtype=np.float32)
    request = writer.OmeZarrSaveRequest(
        data=arr,
        path="/tmp/pyramid_zyx.zarr",
        metadata={"axes": "TCZYX"},
        pyramid_levels=3,
        pyramid_downsample="zyx",
    )

    result = writer.save_ome_zarr(request)

    assert result.levels == 3
    assert calls == [
        {"shape": (1, 1, 8, 8, 8), "component": "0"},
        {"shape": (1, 1, 4, 4, 4), "component": "1"},
        {"shape": (1, 1, 2, 2, 2), "component": "2"},
    ]
    datasets = fake_group.attrs["multiscales"][0]["datasets"]
    assert datasets[1]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 2.0, 2.0, 2.0]
    assert datasets[2]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 4.0, 4.0, 4.0]


def test_save_ome_zarr_truncates_existing_multiscale_to_requested_levels(monkeypatch):
    fake_group = _FakeGroup()
    calls = []

    monkeypatch.setattr(writer, "_open_group", lambda path, mode: fake_group)

    def fake_to_zarr(array, url, component, overwrite, **kwargs):
        calls.append({"shape": tuple(array.shape), "component": component})

    monkeypatch.setattr(writer.da, "to_zarr", fake_to_zarr)

    levels = [
        np.zeros((1, 1, 4, 16, 16), dtype=np.float32),
        np.zeros((1, 1, 4, 8, 8), dtype=np.float32),
        np.zeros((1, 1, 4, 4, 4), dtype=np.float32),
        np.zeros((1, 1, 4, 2, 2), dtype=np.float32),
        np.zeros((1, 1, 4, 1, 1), dtype=np.float32),
    ]
    request = writer.OmeZarrSaveRequest(
        data=levels,
        path="/tmp/existing_multiscale.zarr",
        metadata={"axes": "TCZYX"},
        pyramid_levels=2,
    )

    result = writer.save_ome_zarr(request)

    assert result.levels == 2
    assert result.requested_levels == 2
    assert result.source_levels == 5
    assert calls == [
        {"shape": (1, 1, 4, 16, 16), "component": "0"},
        {"shape": (1, 1, 4, 8, 8), "component": "1"},
    ]
    assert fake_group.attrs["napari_flow"]["requested_levels"] == 2
    assert fake_group.attrs["napari_flow"]["written_levels"] == 2
    assert fake_group.attrs["napari_flow"]["source_levels"] == 5
