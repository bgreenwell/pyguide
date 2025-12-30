import numpy as np

from pyguide import GuideTreeClassifier


def test_max_depth():
    # Force a deep tree and see if it stops
    X = np.arange(10).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    clf = GuideTreeClassifier(max_depth=1)
    clf.fit(X, y)

    # Root node should have children, but children should be leaves
    assert clf._root.left.is_leaf
    assert clf._root.right.is_leaf


def test_min_samples_split():
    X = np.arange(4).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])

    # min_samples_split=10, but only 4 samples. Should not split.
    clf = GuideTreeClassifier(min_samples_split=10)
    clf.fit(X, y)

    assert clf._root.is_leaf


def test_pure_node_no_split():
    # Already pure
    X = np.arange(10).reshape(-1, 1)
    y = np.array([1] * 10)

    clf = GuideTreeClassifier()
    clf.fit(X, y)

    assert clf._root.is_leaf
    assert clf._root.prediction == 1
