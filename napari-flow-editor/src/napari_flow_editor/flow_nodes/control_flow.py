from .decorator import register_node

# --- BEGIN NODE (Control-Flow) ---
@register_node(
    label="Begin",
    category="Control Flow",
    outputs=[],
    logic={
        "in": False,
        "out": True,
        "allow_multi_in": False,
        "allow_multi_out": False,
    },
    params_config={},
)
def begin():
    """Entry point for the exec thread."""
    return None


@register_node(
    label="Begin Batch",
    category="Control Flow",
    outputs=[],
    logic={
        "in": False,
        "out": True,
        "allow_multi_in": False,
        "allow_multi_out": False,
    },
    params_config={
        "csv_path": {
            "type": "path",
            "mode": "file",
            "filter": "CSV Files (*.csv);;All Files (*)",
        }
    },
)
def begin_batch(csv_path: str = ""):
    """
    Entry point for batch execution.

    The execution engine reads one CSV row at a time and runs the connected
    exec thread for each row.
    """
    return None


# --- LOOP NODE (Control-Flow) ---
@register_node(
    label="Loop",
    category="Control Flow",
    outputs=[],
    logic={
        "in": True,
        "out": True,
        # Loop uses single feedback links by design (for now).
        "allow_multi_in": False,
        "allow_multi_out": False,
    },
    params_config={
        "mode": {"options": ["N times", "Until confirm"]},
        "iterations": {"min": 1, "max": 100, "step": 1},
    }
)
def loop_control(mode: str = "N times", iterations: int = 3):
    """
    Control-flow Loop node for visual programming.
    - mode: Loop mode (N times, Until confirm)
    - iterations: Number of times to loop (if mode is N times)
    This node does not process data, but controls execution flow via logic sockets.
    """
    # Placeholder: actual loop logic is handled by the execution engine.
    return None
