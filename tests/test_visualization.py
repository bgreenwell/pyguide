import numpy as np

from pyguide import GuideTreeClassifier, plot_tree
from pyguide.visualization import MockTree, build_mock_tree


def test_mock_tree_initialization():
    children_left = np.array([1, -1, -1])
    children_right = np.array([2, -1, -1])
    feature = np.array([0, -2, -2])
    threshold = np.array([0.5, -2, -2])
    value = np.array([[[5, 5]], [[5, 0]], [[0, 5]]])
    impurity = np.array([0.5, 0.0, 0.0])
    n_node_samples = np.array([10, 5, 5])
    
    tree = MockTree(children_left, children_right, feature, threshold, value, impurity, n_node_samples)
    assert tree.node_count == 3
    assert tree.max_depth == 1
    assert tree.n_classes[0] == 2

def test_build_mock_tree_structure():
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y = np.array([0, 1, 0, 1])
    clf = GuideTreeClassifier(max_depth=1)
    clf.fit(X, y)
    
    mock_tree = build_mock_tree(clf._root, n_classes=2, is_classifier=True)
    assert mock_tree.node_count >= 1
    assert hasattr(mock_tree, "children_left")
    assert hasattr(mock_tree, "children_right")

def test_plot_tree_smoke():
    # Smoke test to ensure it doesn't crash
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    clf = GuideTreeClassifier().fit(X, y)
    
    import matplotlib.pyplot as plt
    plt.figure()
    plot_tree(clf)
    plt.close()
