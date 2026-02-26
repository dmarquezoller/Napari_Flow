from napari_flow_editor.flow_nodes.decorator import register_node


def test_register_node_logic_defaults_are_applied():
    @register_node(label="Test", category="Filters")
    def _node(image):
        return image

    logic = _node._node_meta["logic"]
    assert logic["in"] is True
    assert logic["out"] is True
    assert logic["allow_multi_in"] is True
    assert logic["allow_multi_out"] is True


def test_register_node_logic_overrides_are_applied():
    @register_node(
        label="Test",
        category="Control Flow",
        logic={"in": True, "out": False, "allow_multi_in": False, "allow_multi_out": True},
    )
    def _node():
        return None

    logic = _node._node_meta["logic"]
    assert logic["in"] is True
    assert logic["out"] is False
    assert logic["allow_multi_in"] is False
    assert logic["allow_multi_out"] is True
