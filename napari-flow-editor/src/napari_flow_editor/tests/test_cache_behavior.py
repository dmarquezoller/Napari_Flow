import numpy as np

from .conftest import FakeEdge, FakeNode, FakeSocket


def test_execute_single_node_skips_when_signature_unchanged(worker):
    node = FakeNode("dummy")
    calls = []
    logs = []

    worker.log_signal.connect(logs.append)
    worker.calculate_signature = lambda _node: "sig-1"

    def _fake_logic(_node, _library):
        calls.append("run")
        return {"out": (np.ones((1, 1)), {})}

    worker.execute_node_logic = _fake_logic

    worker._execute_single_node(node, library_def={})
    worker._execute_single_node(node, library_def={})

    assert calls == ["run"]
    assert node.last_signature == "sig-1"
    assert any("Skipping:" in msg for msg in logs)


def test_execute_single_node_force_recompute_bypasses_cache(worker):
    node = FakeNode("dummy")
    calls = []
    worker.calculate_signature = lambda _node: "sig-1"

    def _fake_logic(_node, _library):
        calls.append("run")
        return {"out": (np.ones((1, 1)), {})}

    worker.execute_node_logic = _fake_logic

    worker._execute_single_node(node, library_def={})
    worker._execute_single_node(node, library_def={}, force_recompute=True)

    assert calls == ["run", "run"]
    assert node.last_signature == "sig-1"


def test_calculate_signature_changes_when_upstream_signature_changes(worker):
    upstream = FakeNode("source")
    downstream_input = FakeSocket("image")
    downstream = FakeNode("consumer", inputs=[downstream_input], params={"sigma": 1.0})

    start_socket = type("S", (), {"node": upstream, "name": "out"})()
    downstream_input.connected_edges = [FakeEdge(start_socket=start_socket)]

    upstream.last_signature = "upstream-a"
    sig_a = worker.calculate_signature(downstream)

    upstream.last_signature = "upstream-b"
    sig_b = worker.calculate_signature(downstream)

    assert sig_a != sig_b
