import numpy as np

from pyguide import GuideTreeClassifier, GuideTreeRegressor


def test_apply_classifier():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 1, 1])
    
    clf = GuideTreeClassifier(max_depth=1, significance_threshold=1.0)
    clf.fit(X, y)
    
    indices = clf.apply(X)
    assert len(indices) == 4
    # With max_depth=1, we expect 2 unique leaf indices
    assert len(np.unique(indices)) == 2
    # Samples with X[0]=0 should be in same leaf
    assert indices[0] == indices[1]
    # Samples with X[0]=1 should be in same leaf
    assert indices[2] == indices[3]
    # Different leaves
    assert indices[0] != indices[2]

def test_apply_regressor():
    X = np.array([[0], [1], [2], [3]]).reshape(-1, 1)
    y = np.array([0, 0, 10, 10])
    
    reg = GuideTreeRegressor(max_depth=1, significance_threshold=1.0)
    reg.fit(X, y)
    
    indices = reg.apply(X)
    assert len(indices) == 4
    assert len(np.unique(indices)) == 2
    assert indices[0] == indices[1]
    assert indices[2] == indices[3]
    assert indices[0] != indices[2]

def test_decision_path():
    from scipy.sparse import csr_matrix
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    
    clf = GuideTreeClassifier(max_depth=1, significance_threshold=1.0)
    clf.fit(X, y)
    
    indicator = clf.decision_path(X)
    assert isinstance(indicator, csr_matrix)
    assert indicator.shape == (2, 3) # 2 samples, 3 nodes (root + 2 leaves)
    
    # Both samples pass through root (id=0)
    assert indicator[0, 0] == 1
    assert indicator[1, 0] == 1
    
    # Path for sample 0: root -> leaf 1 (or 2)
    path0 = indicator.getrow(0).indices
    assert 0 in path0
    assert len(path0) == 2

def test_decision_path_regressor():
    from scipy.sparse import csr_matrix
    X = np.array([[0], [1], [2], [3]]).reshape(-1, 1)
    y = np.array([0, 0, 10, 10])
    
    reg = GuideTreeRegressor(max_depth=1, significance_threshold=1.0)
    reg.fit(X, y)
    
    indicator = reg.decision_path(X)
    assert isinstance(indicator, csr_matrix)
    assert indicator.shape == (4, 3)
    
    # All samples pass through root
    assert np.all(indicator[:, 0].toarray() == 1)
