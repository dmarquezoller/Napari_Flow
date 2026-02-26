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


def _connect_data(src_node, dst_node):
    src = src_node.outputs[0]
    dst = dst_node.inputs[0]
    _Edge(src, dst)


def _connect_logic(src_node, src_attr, dst_node, dst_attr):
    src = getattr(src_node, src_attr)[0]
    dst = getattr(dst_node, dst_attr)[0]
    _Edge(src, dst)


def _make_worker():
    scene = SimpleNamespace(items=lambda: [])
    viewer = SimpleNamespace(layers={})
    return ExecutionWorker(scene, viewer)


def test_detect_loop_group_finds_body_nodes_in_order():
    worker = _make_worker()

    loop = _Node("loop", node_type="loop_control", mode="N times", iterations=3)
    a = _Node("a")
    b = _Node("b")
    c = _Node("c")

    for n in (loop, a, b, c):
        n.inputs = [_Socket(n, "in")]
        n.outputs = [_Socket(n, "out")]
        n.logic_inputs = [_Socket(n, "logic_in")]
        n.logic_outputs = [_Socket(n, "logic_out")]

    _connect_data(a, b)
    _connect_data(b, c)
    _connect_logic(loop, "logic_outputs", a, "logic_inputs")
    _connect_logic(c, "logic_outputs", loop, "logic_inputs")

    groups = worker.detect_loop_groups([a, b, c, loop])
    assert len(groups) == 1
    group = groups[0]

    assert group["mode"] == "N times"
    assert group["iterations"] == 3
    assert [n.uid for n in group["body_order"]] == ["a", "b", "c"]


def test_execute_loop_group_n_times_runs_expected_iterations():
    worker = _make_worker()
    node1 = _Node("n1")
    node2 = _Node("n2")
    calls = []

    def _fake_exec(node, _library):
        calls.append(node.uid)

    worker._execute_single_node = _fake_exec
    group = {
        "loop_node": _Node("loop", node_type="loop_control", mode="N times", iterations=3),
        "mode": "N times",
        "iterations": 3,
        "body_order": [node1, node2],
    }

    worker._execute_loop_group(group, library_def={})
    assert calls == ["n1", "n2", "n1", "n2", "n1", "n2"]


def test_execute_loop_group_until_confirm_stops_on_request():
    worker = _make_worker()
    node1 = _Node("n1")
    node2 = _Node("n2")
    loop_node = _Node("loop", node_type="loop_control", mode="Until confirm", iterations=1)
    calls = []
    signals = []

    worker.loop_control_state_signal.connect(
        lambda active, uid: signals.append((active, uid))
    )

    def _fake_exec(node, _library):
        calls.append(node.uid)
        # Stop immediately after first node executes.
        if len(calls) == 1:
            worker.request_loop_stop()

    worker._execute_single_node = _fake_exec
    group = {
        "loop_node": loop_node,
        "mode": "Until confirm",
        "iterations": 1,
        "body_order": [node1, node2],
    }

    worker._execute_loop_group(group, library_def={})
    assert calls == ["n1"]
    assert signals == [(True, "loop"), (False, "loop")]
