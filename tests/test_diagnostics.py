import numpy as np
import pytest
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
