import numpy as np
from .conftest import FakeNode, FakeSocket, FakeEdge


def test_execute_node_logic_wraps_result(worker):
    def dummy_func():
        return np.ones((2, 2))

    node = FakeNode(node_type="dummy", params={})
    library_def = {
        "dummy": {
            "outputs": ["out"],
            "executable": dummy_func,
        }
    }

    result = worker.execute_node_logic(node, library_def)
    data, meta = result["out"]

    assert data.shape == (2, 2)
    assert isinstance(meta, dict)


def test_execute_node_logic_unwraps_input_envelope(worker):
    # Upstream node with cached envelope
    upstream = FakeNode("upstream")
    upstream.cached_results = {
        "out": (np.zeros((3, 3)), {"axes": "ZYX"})
    }

    # Downstream node expects input
    input_socket = FakeSocket("image")
    input_socket.connected_edges = [FakeEdge(start_socket=type("S", (), {"node": upstream, "name": "out"})())]

    node = FakeNode(node_type="dummy", inputs=[input_socket])

    def dummy_func(image):
        assert image.shape == (3, 3)
        return image + 1

    library_def = {
        "dummy": {
            "outputs": ["out"],
            "executable": dummy_func,
        }
    }

    result = worker.execute_node_logic(node, library_def)
    _, meta = result["out"]
    assert meta["axes"] == "ZYX"


def test_execute_node_logic_preserves_inherited_contrast_limits(worker):
    upstream = FakeNode("upstream")
    upstream.cached_results = {
        "out": (
            np.zeros((8, 8), dtype=np.float32),
            {"contrast_limits": [0.0, 1.0], "name": "Source"},
        )
    }

    input_socket = FakeSocket("image")
    input_socket.connected_edges = [
        FakeEdge(start_socket=type("S", (), {"node": upstream, "name": "out"})())
    ]
    node = FakeNode(node_type="dummy", inputs=[input_socket])

    def dummy_func(image):
        # Processed output has a very different intensity range.
        return image + 100.0

    library_def = {
        "dummy": {
            "outputs": ["out"],
            "executable": dummy_func,
        }
    }

    result = worker.execute_node_logic(node, library_def)
    data, meta = result["out"]

    assert data.shape == (8, 8)
    assert meta["contrast_limits"] == [0.0, 1.0]


def test_execute_node_logic_computes_contrast_limits_when_missing(worker):
    upstream = FakeNode("upstream")
    upstream.cached_results = {
        "out": (
            np.zeros((8, 8), dtype=np.float32),
            {"name": "Source"},
        )
    }

    input_socket = FakeSocket("image")
    input_socket.connected_edges = [
        FakeEdge(start_socket=type("S", (), {"node": upstream, "name": "out"})())
    ]
    node = FakeNode(node_type="dummy", inputs=[input_socket])

    def dummy_func(image):
        # Processed output has a very different intensity range.
        return image + 100.0

    library_def = {
        "dummy": {
            "outputs": ["out"],
            "executable": dummy_func,
        }
    }

    result = worker.execute_node_logic(node, library_def)
    data, meta = result["out"]

    assert data.shape == (8, 8)
    assert "contrast_limits" in meta
    lo, hi = meta["contrast_limits"]
    assert lo >= 99.0
    assert hi <= 101.0
    assert hi > lo


def test_execute_node_logic_keeps_explicit_contrast_limits(worker):
    node = FakeNode(node_type="dummy", params={})

    def dummy_func():
        return np.ones((4, 4), dtype=np.float32), {"contrast_limits": [5.0, 6.0]}

    library_def = {
        "dummy": {
            "outputs": ["out"],
            "executable": dummy_func,
        }
    }

    result = worker.execute_node_logic(node, library_def)
    _, meta = result["out"]
    assert meta["contrast_limits"] == [5.0, 6.0]


def test_execute_node_logic_uses_node_scoped_output_name_for_processed_nodes(worker):
    upstream = FakeNode("upstream")
    upstream.cached_results = {
        "out": (
            np.zeros((8, 8), dtype=np.float32),
            {"name": "nuclei"},
        )
    }

    input_socket = FakeSocket("image")
    input_socket.connected_edges = [
        FakeEdge(start_socket=type("S", (), {"node": upstream, "name": "out"})())
    ]
    node = FakeNode(node_type="dummy", title="Gaussian Blur", inputs=[input_socket])

    def dummy_func(image):
        return image + 1

    library_def = {
        "dummy": {
            "outputs": ["out"],
            "category": "Filters",
            "executable": dummy_func,
        }
    }

    result = worker.execute_node_logic(node, library_def)
    _, meta = result["out"]
    assert meta["name"] == "Gaussian Blur Output"


def test_execute_node_logic_preserves_name_for_input_category_nodes(worker):
    upstream = FakeNode("upstream")
    upstream.cached_results = {
        "out": (
            np.zeros((8, 8), dtype=np.float32),
            {"name": "nuclei"},
        )
    }

    input_socket = FakeSocket("image")
    input_socket.connected_edges = [
        FakeEdge(start_socket=type("S", (), {"node": upstream, "name": "out"})())
    ]
    node = FakeNode(node_type="dummy", title="Select Layer", inputs=[input_socket])

    def dummy_func(image):
        return image

    library_def = {
        "dummy": {
            "outputs": ["out"],
            "category": "Inputs",
            "executable": dummy_func,
        }
    }

    result = worker.execute_node_logic(node, library_def)
    _, meta = result["out"]
    assert meta["name"] == "nuclei"


def test_execute_node_logic_video_render_injects_current_params(worker):
    captured = {}

    def _fake_request(node_uid, config):
        captured["node_uid"] = node_uid
        captured["config"] = config
        return {"output_path": "/tmp/demo.mp4", "frames": 5}

    worker._request_interaction = _fake_request

    node = FakeNode(
        node_type="video_node",
        title="Make Video",
        params={
            "instructions": [
                {
                    "op": "rotate",
                    "space": "3d",
                    "rot_axis": "z",
                    "direction": "clockwise",
                    "start": 0.0,
                    "end": 10.0,
                    "step": 1.0,
                    "axis": None,
                }
            ],
            "fps": 12,
            "format": ".gif",
            "folder": "/tmp",
            "filename": "demo",
        },
        inputs=[],
        outputs=[],
    )

    def _video_sink(
        instructions=None,
        fps=20,
        format=".mp4",
        folder="",
        filename="video",
        interaction=None,
    ):
        assert isinstance(interaction, dict)
        assert interaction["frames"] == 5
        return None

    library_def = {
        "video_node": {
            "outputs": [],
            "interactive": {"interaction_type": "video_render"},
            "executable": _video_sink,
        }
    }

    result = worker.execute_node_logic(node, library_def)
    assert result == {}
    assert captured["config"]["interaction_type"] == "video_render"
    assert isinstance(captured["config"]["video_params"]["instructions"], list)
    assert captured["config"]["video_params"]["fps"] == 12
    assert captured["config"]["video_params"]["instructions"][0]["op"] == "rotate"
