import csv
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
        self.dynamic_param_bindings = {}
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


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_run_batch_executes_each_row_with_dynamic_parameter(monkeypatch, tmp_path):
    csv_path = tmp_path / "batch.csv"
    _write_csv(
        csv_path,
        [
            {"sigma": "1.5"},
            {"sigma": "2.0"},
            {"sigma": "3.25"},
        ],
    )

    begin = _Node("begin_batch", node_type="begin_batch")
    begin.parameters = {"csv_path": str(csv_path)}
    _with_exec_outputs(begin, ["exec_out"])

    step = _Node("step", node_type="gaussian_blur")
    _with_exec_input(step)
    _with_exec_outputs(step, ["exec_out"])
    step.parameters = {"sigma": 0.0}
    step.dynamic_param_bindings = {"sigma": "sigma"}

    _connect_exec(begin, "exec_out", step)

    observed = []

    def _fake_blur(sigma=0.0):
        observed.append(float(sigma))
        return float(sigma)

    node_library = {
        "gaussian_blur": {
            "outputs": ["out"],
            "executable": _fake_blur,
            "parameters": {"sigma": {"type": "float", "default": 1.0}},
        }
    }
    _install_fake_plugin_module(monkeypatch, node_library=node_library)

    worker = _make_worker([begin, step])
    errors = []
    worker.error_signal.connect(errors.append)
    worker.run()

    assert not errors
    assert observed == [1.5, 2.0, 3.25]


def test_run_batch_errors_for_missing_dynamic_column(monkeypatch, tmp_path):
    csv_path = tmp_path / "batch.csv"
    _write_csv(csv_path, [{"sigma": "1"}])

    begin = _Node("begin_batch", node_type="begin_batch")
    begin.parameters = {"csv_path": str(csv_path)}
    _with_exec_outputs(begin, ["exec_out"])

    step = _Node("step", node_type="gaussian_blur")
    _with_exec_input(step)
    _with_exec_outputs(step, ["exec_out"])
    step.parameters = {"sigma": 0.0}
    step.dynamic_param_bindings = {"sigma": "missing_col"}

    _connect_exec(begin, "exec_out", step)

    node_library = {
        "gaussian_blur": {
            "outputs": ["out"],
            "executable": lambda sigma=0.0: sigma,
            "parameters": {"sigma": {"type": "float", "default": 1.0}},
        }
    }
    _install_fake_plugin_module(monkeypatch, node_library=node_library)

    worker = _make_worker([begin, step])
    errors = []
    worker.error_signal.connect(errors.append)
    worker.run()

    assert errors
    assert "missing_col" in errors[0]
