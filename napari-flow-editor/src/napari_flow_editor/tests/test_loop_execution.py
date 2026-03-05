from types import SimpleNamespace

from napari_flow_editor.execution_engine import ExecutionWorker


class _Socket:
    def __init__(self, node, name):
        self.node = node
        self.name = name
        self.connected_edges = []


class _Edge:
    def __init__(self, start_socket, end_socket):
        self.start_socket = start_socket
        self.end_socket = end_socket
        start_socket.connected_edges.append(self)
        end_socket.connected_edges.append(self)


class _Node:
    def __init__(self, uid, node_type="dummy", mode="N times", iterations=1):
        self.uid = uid
        self.title = uid
        self.node_type = node_type
        self.parameters = {"mode": mode, "iterations": iterations}
        self.inputs = []
        self.outputs = []
        self.logic_inputs = []
        self.logic_outputs = []
        self.last_signature = None
        self.cached_results = {}


def _make_worker():
    scene = SimpleNamespace(items=lambda: [])
    viewer = SimpleNamespace(layers={})
    return ExecutionWorker(scene, viewer)


def _with_exec_input(node):
    sock = _Socket(node, "logic_in")
    node.logic_inputs = [sock]
    return sock


def _with_exec_outputs(node, names):
    node.logic_outputs = [_Socket(node, n) for n in names]
    return node.logic_outputs


def _connect_exec(src_node, out_name, dst_node):
    src = next(s for s in src_node.logic_outputs if s.name == out_name)
    dst = dst_node.logic_inputs[0]
    _Edge(src, dst)


def test_next_logic_node_uses_named_output():
    worker = _make_worker()
    loop = _Node("loop", node_type="loop_control")
    body = _Node("body")
    done = _Node("done")
    _with_exec_outputs(loop, ["loop_body", "completed"])
    _with_exec_input(body)
    _with_exec_input(done)
    _connect_exec(loop, "loop_body", body)
    _connect_exec(loop, "completed", done)

    assert worker._next_logic_node(loop, "loop_body") is body
    assert worker._next_logic_node(loop, "completed") is done


def test_next_logic_node_defaults_to_first_output():
    worker = _make_worker()
    n = _Node("n")
    a = _Node("a")
    b = _Node("b")
    _with_exec_outputs(n, ["first", "second"])
    _with_exec_input(a)
    _with_exec_input(b)
    _connect_exec(n, "first", a)
    _connect_exec(n, "second", b)

    assert worker._next_logic_node(n) is a


def test_execute_loop_node_n_times_runs_body_and_returns_completed():
    worker = _make_worker()
    loop = _Node("loop", node_type="loop_control", mode="N times", iterations=3)
    body = _Node("body")
    done = _Node("done")
    _with_exec_outputs(loop, ["loop_body", "completed"])
    _with_exec_input(body)
    _with_exec_input(done)
    _connect_exec(loop, "loop_body", body)
    _connect_exec(loop, "completed", done)

    calls = []

    def _fake_exec_path(start_node, _library, force_recompute=False):
        calls.append((start_node.uid, force_recompute))

    worker._execute_exec_path = _fake_exec_path
    nxt = worker._execute_loop_node(loop, library_def={})

    assert calls == [("body", True), ("body", True), ("body", True)]
    assert nxt is done


def test_execute_loop_node_without_body_logs_warning_and_continues_completed():
    worker = _make_worker()
    loop = _Node("loop", node_type="loop_control", mode="N times", iterations=3)
    done = _Node("done")
    _with_exec_outputs(loop, ["loop_body", "completed"])
    _with_exec_input(done)
    _connect_exec(loop, "completed", done)

    logs = []
    worker.log_signal.connect(logs.append)
    nxt = worker._execute_loop_node(loop, library_def={})

    assert nxt is done
    assert any("has no Loop Body connection" in msg for msg in logs)


def test_exec_transition_signal_emitted_for_linear_exec_path():
    worker = _make_worker()
    a = _Node("a")
    b = _Node("b")
    _with_exec_input(a)
    _with_exec_outputs(a, ["exec_out"])
    _with_exec_input(b)
    _with_exec_outputs(b, ["exec_out"])
    _connect_exec(a, "exec_out", b)

    transitions = []
    calls = []
    worker.exec_transition_signal.connect(
        lambda from_uid, out_name: transitions.append((from_uid, out_name))
    )

    def _fake_exec_single(node, _library, force_recompute=False):
        calls.append((node.uid, force_recompute))

    worker._execute_single_node = _fake_exec_single
    worker._execute_exec_path(a, library_def={})

    assert transitions == [("a", "exec_out")]
    assert calls == [("a", False), ("b", False)]


def test_exec_transition_signal_emitted_for_loop_body_and_completed():
    worker = _make_worker()
    loop = _Node("loop", node_type="loop_control", mode="N times", iterations=1)
    body = _Node("body")
    done = _Node("done")
    _with_exec_outputs(loop, ["loop_body", "completed"])
    _with_exec_input(body)
    _with_exec_outputs(body, ["exec_out"])
    _with_exec_input(done)
    _connect_exec(loop, "loop_body", body)
    _connect_exec(loop, "completed", done)

    transitions = []
    worker.exec_transition_signal.connect(
        lambda from_uid, out_name: transitions.append((from_uid, out_name))
    )
    worker._execute_single_node = lambda *args, **kwargs: None

    nxt = worker._execute_loop_node(loop, library_def={})

    assert nxt is done
    assert ("loop", "loop_body") in transitions
    assert ("loop", "completed") in transitions


def test_execute_loop_node_until_confirm_stops_immediately():
    worker = _make_worker()
    loop = _Node("loop", node_type="loop_control", mode="Until confirm", iterations=1)
    body = _Node("body")
    done = _Node("done")
    _with_exec_outputs(loop, ["loop_body", "completed"])
    _with_exec_input(body)
    _with_exec_input(done)
    _connect_exec(loop, "loop_body", body)
    _connect_exec(loop, "completed", done)

    signals = []
    worker.loop_control_state_signal.connect(lambda active, uid: signals.append((active, uid)))

    calls = []

    def _fake_exec_path(start_node, _library, force_recompute=False):
        calls.append((start_node.uid, force_recompute))
        worker.request_loop_stop()

    worker._execute_exec_path = _fake_exec_path
    nxt = worker._execute_loop_node(loop, library_def={})

    assert calls == [("body", True)]
    assert signals == [(True, "loop"), (False, "loop")]
    assert nxt is done


def test_execute_exec_path_runs_loop_then_completed_branch():
    worker = _make_worker()
    a = _Node("a")
    loop = _Node("loop", node_type="loop_control", mode="N times", iterations=2)
    body = _Node("body")
    done = _Node("done")

    _with_exec_input(a)
    _with_exec_outputs(a, ["exec_out"])
    _with_exec_input(loop)
    _with_exec_outputs(loop, ["loop_body", "completed"])
    _with_exec_input(body)
    _with_exec_outputs(body, ["exec_out"])  # unconnected -> end body path
    _with_exec_input(done)
    _with_exec_outputs(done, ["exec_out"])  # unconnected -> end main path

    _connect_exec(a, "exec_out", loop)
    _connect_exec(loop, "loop_body", body)
    _connect_exec(loop, "completed", done)

    calls = []

    def _fake_exec_single(node, _library, force_recompute=False):
        calls.append((node.uid, force_recompute))

    worker._execute_single_node = _fake_exec_single
    worker._execute_exec_path(a, library_def={})

    assert calls == [
        ("a", False),
        ("body", True),
        ("body", True),
        ("done", False),
    ]


def test_execute_node_logic_refreshes_dirty_data_source():
    worker = _make_worker()
    src = _Node("src")
    dst = _Node("dst", node_type="consumer")
    dst.parameters = {}

    src_out = _Socket(src, "data_out")
    dst_in = _Socket(dst, "image_input")
    src.outputs = [src_out]
    dst.inputs = [dst_in]
    _Edge(src_out, dst_in)

    # Stale cache exists, but source is marked dirty.
    src.cached_results = {"data_out": ("old", {"name": "old"})}
    src.last_signature = None
    src.status = "gray"

    calls = []

    def _fake_exec_single(node, _library, force_recompute=False):
        calls.append((node.uid, force_recompute))
        node.cached_results["data_out"] = ("new", {"name": "new"})
        node.last_signature = "fresh"
        node.status = "green"

    worker._execute_single_node = _fake_exec_single

    library_def = {
        "consumer": {
            "executable": lambda image_input: image_input,
            "outputs": [],
        }
    }
    worker.execute_node_logic(dst, library_def)

    assert calls == [("src", True)]
