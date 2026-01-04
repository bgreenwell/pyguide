import numpy as np
from pyguide import GuideTreeClassifier, GuideTreeRegressor

def test_classifier_split_metadata():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 10)
    y = np.array([0, 1, 1, 0] * 10)  # XOR
    
    # Use a threshold that will be exceeded by main effect p-values (which are ~1.0 for XOR)
    clf = GuideTreeClassifier(interaction_depth=1, significance_threshold=0.05)
    clf.fit(X, y)
    
    assert clf._root.split_type == "interaction"
    assert sorted(list(clf._root.interaction_group)) == [0, 1]

def test_regressor_split_metadata():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 10)
    y = np.array([0.0, 1.0, 1.0, 0.0] * 10)
    
    reg = GuideTreeRegressor(interaction_depth=1, significance_threshold=0.05)
    reg.fit(X, y)
    
    assert reg._root.split_type == "interaction"
    assert sorted(list(reg._root.interaction_group)) == [0, 1]

def test_classifier_main_metadata():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 10)
    y = np.array([0, 0, 1, 1] * 10) # Pure main effect on X0
    
    clf = GuideTreeClassifier(interaction_depth=1, significance_threshold=0.05)
    clf.fit(X, y)
    
    assert clf._root.split_type == "main"
    assert clf._root.interaction_group is None
