from napari_flow_editor.flow_nodes.video import make_video


def test_make_video_node_registration_metadata():
    meta = make_video._node_meta
    assert meta["label"] == "Make Video"
    assert meta["category"] == "Video"
    assert meta["outputs"] == []
    assert meta["interactive"]["interaction_type"] == "video_render"
    assert "instructions" in meta["params_config"]


def test_make_video_node_accepts_interaction_payload():
    result = make_video(
        instructions=[
            {
                "op": "sweep",
                "axis": "idx:0",
                "start": 0,
                "end": 2,
                "step": 1,
                "space": None,
                "rot_axis": None,
                "direction": None,
            }
        ],
        fps=10,
        format=".gif",
        folder="/tmp",
        filename="demo",
        interaction={"output_path": "/tmp/demo.gif", "frames": 3},
    )
    assert result is None
