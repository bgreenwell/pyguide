from pyguide.node import GuideNode


def test_node_init():
    node = GuideNode(depth=0)
    assert node.depth == 0
    assert not node.is_leaf
    assert node.prediction is None
    assert node.split_feature is None
    assert node.split_threshold is None
    assert node.left is None
    assert node.right is None


def test_leaf_node():
    node = GuideNode(depth=2, is_leaf=True, prediction=1)
    assert node.depth == 2
    assert node.is_leaf
    assert node.prediction == 1
