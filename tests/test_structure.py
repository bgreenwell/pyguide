import numpy as np

from pyguide import GuideTreeClassifier, GuideTreeRegressor


def test_structure_classifier():
    # Linearly separable dataset
    X = np.array([[0], [0], [1], [1]]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])
    
    # Use high significance threshold to ensure it splits even if p-values are high
    clf = GuideTreeClassifier(max_depth=None, min_samples_split=2, significance_threshold=1.0)
    clf.fit(X, y)
    
    # Verify attributes
    assert hasattr(clf, "n_leaves_")
    assert hasattr(clf, "max_depth_")
    assert hasattr(clf, "get_n_leaves")
    assert hasattr(clf, "get_depth")
    
    n_leaves = clf.get_n_leaves()
    depth = clf.get_depth()
    
    assert n_leaves == clf.n_leaves_
    assert depth == clf.max_depth_
    
    assert n_leaves == 2
    assert depth == 1
    
    # Manual check of the tree structure
    def count_leaves(node):
        if node.is_leaf:
            return 1
        return count_leaves(node.left) + count_leaves(node.right)
    
    def calc_depth(node):
        if node.is_leaf:
            return 0
        return 1 + max(calc_depth(node.left), calc_depth(node.right))
        
    assert n_leaves == count_leaves(clf._root)
    assert depth == calc_depth(clf._root)

def test_structure_regressor():
    X = np.array([[0], [1], [2], [3]]).reshape(-1, 1)
    y = np.array([0, 0, 10, 10])
    
    # Force split
    reg = GuideTreeRegressor(max_depth=1, significance_threshold=1.0)
    reg.fit(X, y)
    
    assert reg.n_leaves_ == 2
    assert reg.max_depth_ == 1
