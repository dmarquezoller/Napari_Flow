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
