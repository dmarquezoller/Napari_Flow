from .decorator import register_node


@register_node(
    label="Make Video",
    category="Video",
    outputs=[],
    interactive={
        "interaction_type": "video_render",
        "prompt": "Click Run to render the video from the current viewer state.",
    },
    params_config={
        "instructions": {
            "type": "table",
            "value": [
                {
                    "op": "rotate",
                    "view": "3d",
                    "space": "3d",
                    "rot_axis": "z",
                    "direction": "clockwise",
                    "axis": None,
                    "start": 0.0,
                    "end": 360.0,
                    "step": 1.0,
                }
            ],
            "columns": [
                {
                    "name": "op",
                    "label": "Op",
                    "type": "enum",
                    "options": ["rotate", "sweep"],
                    "width": 90,
                },
                {
                    "name": "view",
                    "label": "View",
                    "type": "enum",
                    "options": ["keep", "2d", "3d"],
                    "width": 75,
                },
                {
                    "name": "space",
                    "label": "Space",
                    "type": "enum",
                    "options": ["2d", "3d"],
                    "show_if": {"op": ["rotate"]},
                    "width": 80,
                },
                {
                    "name": "rot_axis",
                    "label": "Rot",
                    "type": "enum",
                    "options": ["x", "y", "z"],
                    "show_if": {"op": ["rotate"], "space": ["3d"]},
                    "width": 95,
                },
                {
                    "name": "direction",
                    "label": "Dir",
                    "type": "enum",
                    "options": ["clockwise", "counterclockwise"],
                    "show_if": {"op": ["rotate"]},
                    "width": 105,
                },
                {
                    "name": "axis",
                    "label": "Axis",
                    "type": "enum",
                    "options": ["idx:0"],
                    "show_if": {"op": ["sweep"]},
                    "width": 120,
                },
                {
                    "name": "start",
                    "label": "Start",
                    "type": "float",
                    "value": 0.0,
                    "step": 1.0,
                    "width": 85,
                },
                {
                    "name": "end",
                    "label": "End",
                    "type": "float",
                    "value": 360.0,
                    "step": 1.0,
                    "width": 85,
                },
                {
                    "name": "step",
                    "label": "Step",
                    "type": "float",
                    "value": 1.0,
                    "min": -3600.0,
                    "max": 3600.0,
                    "step": 0.5,
                    "width": 85,
                },
            ],
        },
        "fps": {
            "type": "int",
            "value": 20,
            "min": 1,
            "max": 120,
            "label": "FPS",
        },
        "format": {
            "type": "enum",
            "options": [".mp4", ".gif"],
            "label": "Format",
        },
        "folder": {
            "type": "path",
            "mode": "directory",
            "label": "Save Folder",
        },
        "filename": {
            "type": "text",
            "value": "napari_video",
            "label": "Filename",
        },
    },
    description=(
        "Renders a viewer animation to MP4/GIF.\n"
        "Use instruction rows to chain rotate/sweep segments.\n"
        "For sweep, use axis idx options (idx:N)."
    ),
)
def make_video(
    instructions=None,
    fps: int = 20,
    format: str = ".mp4",
    folder: str = "",
    filename: str = "napari_video",
    interaction=None,
):
    """
    Video rendering is executed on the main thread by the UI interaction
    handler. This node itself is a sink and only reports completion.
    """
    if isinstance(interaction, dict):
        out_path = interaction.get("output_path")
        frame_count = interaction.get("frames")
        if out_path:
            if frame_count is not None:
                print(f"🎬 Video saved: {out_path} ({frame_count} frames)")
            else:
                print(f"🎬 Video saved: {out_path}")
    return None
