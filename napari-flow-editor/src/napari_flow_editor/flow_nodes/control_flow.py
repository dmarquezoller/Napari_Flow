from .decorator import register_node

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
