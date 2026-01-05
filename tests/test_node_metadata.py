from pyguide.node import GuideNode


def test_node_metadata():
    node = GuideNode(
        depth=0,
        split_type="interaction",
        interaction_group=[1, 2]
    )
    assert node.split_type == "interaction"
    assert node.interaction_group == [1, 2]

def test_node_metadata_default():
    node = GuideNode(depth=0)
    # By default, it should be main if not leaf, or None?
    # Spec says split_type ("main" or "interaction")
    # For a leaf, maybe None.
    assert hasattr(node, "split_type")
    assert hasattr(node, "interaction_group")
