import numpy as np
import pytest


class FakeLayer:
    def __init__(self, data, metadata=None):
        self.data = data
        self.metadata = metadata or {}


class FakeViewer:
    def __init__(self, layers=None):
        self.layers = layers or {}


class FakeSocket:
    def __init__(self, name, is_logic=False, socket_type=None):
        self.name = name
        self.connected_edges = []
        self.is_logic = is_logic
        if socket_type is not None:
            self.socket_type = socket_type
        else:
            self.socket_type = ("logic_in" if is_logic else "input")  # default; tests can override


class FakeEdge:
    def __init__(self, start_socket):
        self.start_socket = start_socket


class FakeNode:
    def __init__(
        self,
        node_type,
        title="Node",
        params=None,
        inputs=None,
        outputs=None,
        logic_inputs=None,
        logic_outputs=None,
    ):
        self.node_type = node_type
        self.title = title
        self.uid = "fake-uid"
        self.params = params or {}
        self.parameters = self.params.copy()
        self.inputs = inputs or []
        self.outputs = outputs or []
        # Most nodes in the new model are exec-enabled by default.
        self.logic_inputs = (
            logic_inputs
            if logic_inputs is not None
            else [FakeSocket("logic_in", is_logic=True, socket_type="logic_in")]
        )
        self.logic_outputs = (
            logic_outputs
            if logic_outputs is not None
            else [FakeSocket("exec_out", is_logic=True, socket_type="logic_out")]
        )
        self.cached_results = {}
        self.last_signature = None
        self.status = "green"

    def all_sockets(self):
        return self.inputs + self.outputs + self.logic_inputs + self.logic_outputs


def unpack_execute_result(result):
    """
    Compatibility helper for ExecutionWorker.execute_node_logic return shape.

    Newer engine versions return:
      (node_outputs: dict, node_trace: dict)
    while older versions returned:
      node_outputs: dict
    """
    if isinstance(result, dict):
        return result, {}
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], dict)
        and isinstance(result[1], dict)
    ):
        return result[0], result[1]
    raise AssertionError(
        f"Unexpected execute_node_logic result type: {type(result)!r}"
    )


@pytest.fixture
def viewer():
    data = np.arange(12).reshape(3, 4)
    layer = FakeLayer(data, metadata={"name": "LayerA"})
    return FakeViewer(layers={"LayerA": layer})


@pytest.fixture
def worker(viewer):
    from napari_flow_editor.execution_engine import ExecutionWorker
    return ExecutionWorker(scene=None, viewer=viewer)
