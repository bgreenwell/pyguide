import numpy as np
from pyguide import GuideTreeClassifier, GuideTreeRegressor

def test_classifier_p_values_stored():
    X = np.random.rand(50, 5)
    y = (X[:, 0] > 0.5).astype(int)
    
    clf = GuideTreeClassifier(max_depth=1)
    clf.fit(X, y)
    
    # Root should have p-values of length 5
    assert clf._root.curvature_p_values is not None
    assert len(clf._root.curvature_p_values) == 5
    # Values should be between 0 and 1
    assert np.all(clf._root.curvature_p_values >= 0)
    assert np.all(clf._root.curvature_p_values <= 1)

def test_regressor_p_values_stored():
    X = np.random.rand(50, 5)
    y = X[:, 0] + np.random.normal(0, 0.1, 50)
    
    reg = GuideTreeRegressor(max_depth=1)
    reg.fit(X, y)
    
    assert reg._root.curvature_p_values is not None
    assert len(reg._root.curvature_p_values) == 5
