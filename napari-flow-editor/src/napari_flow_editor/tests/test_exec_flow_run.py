import sys
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
    def __init__(self, uid, node_type="dummy"):
        self.uid = uid
        self.title = uid
        self.node_type = node_type
        self.inputs = []
        self.outputs = []
        self.logic_inputs = []
        self.logic_outputs = []
        self.parameters = {}
        self.cached_results = {}
        self.last_signature = None
        self.status = "gray"


def _with_exec_input(node):
    node.logic_inputs = [_Socket(node, "logic_in")]
    return node.logic_inputs[0]


def _with_exec_outputs(node, names):
    node.logic_outputs = [_Socket(node, n) for n in names]
    return node.logic_outputs


def _connect_exec(src_node, out_name, dst_node):
    src = next(s for s in src_node.logic_outputs if s.name == out_name)
    dst = dst_node.logic_inputs[0]
    _Edge(src, dst)


def _make_worker(nodes):
    scene = SimpleNamespace(items=lambda: nodes)
    viewer = SimpleNamespace(layers={})
    return ExecutionWorker(scene, viewer)


def _install_fake_plugin_module(monkeypatch, node_library=None):
    fake_module = SimpleNamespace(Node=_Node, NODE_LIBRARY=node_library or {})
    monkeypatch.setitem(sys.modules, "napari_flow_editor.napari_plugin_v2", fake_module)


def test_run_errors_when_no_begin_node(monkeypatch):
    _install_fake_plugin_module(monkeypatch)
    worker = _make_worker([_Node("n1", node_type="gaussian_blur")])
    errors = []
    worker.error_signal.connect(errors.append)

    worker.run()

    assert errors
    assert "No Begin node found" in errors[0]


def test_run_errors_on_multiple_begin_nodes(monkeypatch):
    _install_fake_plugin_module(monkeypatch)
    b1 = _Node("b1", node_type="begin")
    b2 = _Node("b2", node_type="begin")
    _with_exec_outputs(b1, ["exec_out"])
    _with_exec_outputs(b2, ["exec_out"])

    worker = _make_worker([b1, b2])
    errors = []
    worker.error_signal.connect(errors.append)

    worker.run()

    assert errors
    assert "Multiple Begin nodes found" in errors[0]


def test_run_logs_when_begin_has_no_exec_connection(monkeypatch):
    _install_fake_plugin_module(monkeypatch)
    begin = _Node("begin", node_type="begin")
    _with_exec_outputs(begin, ["exec_out"])

    worker = _make_worker([begin])
    logs = []
    errors = []
    worker.log_signal.connect(logs.append)
    worker.error_signal.connect(errors.append)

    worker.run()

    assert not errors
    assert any("Begin node is not connected" in msg for msg in logs)


def test_run_starts_exec_path_from_begin_output(monkeypatch):
    _install_fake_plugin_module(monkeypatch)
    begin = _Node("begin", node_type="begin")
    step = _Node("step", node_type="gaussian_blur")
    _with_exec_outputs(begin, ["exec_out"])
    _with_exec_input(step)
    _with_exec_outputs(step, ["exec_out"])
    _connect_exec(begin, "exec_out", step)

    worker = _make_worker([begin, step])
    called = []
    errors = []
    worker.error_signal.connect(errors.append)

    def _fake_exec_path(start_node, _library):
        called.append(start_node.uid)

    worker._execute_exec_path = _fake_exec_path
    worker.run()

    assert not errors
    assert called == ["step"]


def _run_step_once(monkeypatch, nodes, step_state=None):
    _install_fake_plugin_module(monkeypatch)
    worker = ExecutionWorker(
        SimpleNamespace(items=lambda: nodes),
        SimpleNamespace(layers={}),
        run_mode="step",
        step_state=step_state,
    )

    executed = []
    errors = []
    states = []

    worker.error_signal.connect(errors.append)
    worker.step_state_signal.connect(states.append)
    worker._execute_single_node = lambda node, _lib, force_recompute=False: executed.append(
        (node.uid, force_recompute)
    )

    worker.run()
    return executed, states[-1] if states else None, errors


def test_run_next_executes_one_node_per_click(monkeypatch):
    begin = _Node("begin", node_type="begin")
    a = _Node("a")
    b = _Node("b")
    _with_exec_outputs(begin, ["exec_out"])
    _with_exec_input(a)
    _with_exec_outputs(a, ["exec_out"])
    _with_exec_input(b)
    _with_exec_outputs(b, ["exec_out"])
    _connect_exec(begin, "exec_out", a)
    _connect_exec(a, "exec_out", b)

    executed_1, state_1, errors_1 = _run_step_once(monkeypatch, [begin, a, b])
    assert not errors_1
    assert executed_1 == [("a", False)]
    assert isinstance(state_1, dict)
    assert state_1["current_uid"] == "b"

    executed_2, state_2, errors_2 = _run_step_once(
        monkeypatch, [begin, a, b], step_state=state_1
    )
    assert not errors_2
    assert executed_2 == [("b", False)]
    assert state_2 is None


def test_run_next_steps_through_loop_iterations(monkeypatch):
    begin = _Node("begin", node_type="begin")
    loop = _Node("loop", node_type="loop_control")
    loop.parameters = {"mode": "N times", "iterations": 2}
    body = _Node("body")
    done = _Node("done")

    _with_exec_outputs(begin, ["exec_out"])
    _with_exec_input(loop)
    _with_exec_outputs(loop, ["loop_body", "completed"])
    _with_exec_input(body)
    _with_exec_outputs(body, ["exec_out"])  # unconnected => returns to loop frame
    _with_exec_input(done)
    _with_exec_outputs(done, ["exec_out"])

    _connect_exec(begin, "exec_out", loop)
    _connect_exec(loop, "loop_body", body)
    _connect_exec(loop, "completed", done)

    executed_1, state_1, errors_1 = _run_step_once(
        monkeypatch, [begin, loop, body, done]
    )
    assert not errors_1
    assert executed_1 == [("body", True)]
    assert isinstance(state_1, dict)
    assert state_1["current_uid"] == "loop"

    executed_2, state_2, errors_2 = _run_step_once(
        monkeypatch, [begin, loop, body, done], step_state=state_1
    )
    assert not errors_2
    assert executed_2 == [("body", True)]
    assert isinstance(state_2, dict)
    assert state_2["current_uid"] == "loop"

    executed_3, state_3, errors_3 = _run_step_once(
        monkeypatch, [begin, loop, body, done], step_state=state_2
    )
    assert not errors_3
    assert executed_3 == [("done", False)]
    assert state_3 is None


def test_run_next_rejects_begin_batch(monkeypatch):
    _install_fake_plugin_module(monkeypatch)
    begin_batch = _Node("bb", node_type="begin_batch")
    step = _Node("step")
    _with_exec_outputs(begin_batch, ["exec_out"])
    _with_exec_input(step)
    _with_exec_outputs(step, ["exec_out"])
    _connect_exec(begin_batch, "exec_out", step)

    worker = ExecutionWorker(
        SimpleNamespace(items=lambda: [begin_batch, step]),
        SimpleNamespace(layers={}),
        run_mode="step",
    )
    errors = []
    worker.error_signal.connect(errors.append)

    worker.run()

    assert errors
    assert "Run Next does not support Begin Batch" in errors[0]
