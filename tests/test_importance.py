import numpy as np
import pytest
from pyguide import GuideTreeClassifier, GuideTreeRegressor
from sklearn.datasets import make_classification, make_regression

def test_importance_classifier():
    # X0 is predictive, X1 is noise
    X = np.zeros((100, 2))
    X[:50, 0] = 0
    X[50:, 0] = 1
    X[:, 1] = np.random.randn(100) # noise
    y = X[:, 0].astype(int)
    
    clf = GuideTreeClassifier(max_depth=1, significance_threshold=1.0)
    clf.fit(X, y)
    
    importance = clf.feature_importances_
    assert len(importance) == 2
    assert np.isclose(np.sum(importance), 1.0)
    assert importance[0] > importance[1]
    assert importance[1] == 0.0 # X1 was never used

def test_importance_regressor():
    # X0 is predictive, X1 is noise
    X = np.random.rand(100, 2)
    y = 10 * X[:, 0] + np.random.normal(0, 0.1, 100)
    
    reg = GuideTreeRegressor(max_depth=1, significance_threshold=1.0)
    reg.fit(X, y)
    
    importance = reg.feature_importances_
    assert len(importance) == 2
    assert np.isclose(np.sum(importance), 1.0)
    assert importance[0] > importance[1]

def test_select_from_model():
    from sklearn.feature_selection import SelectFromModel
    # Use very clear predictive signals and noise
    X = np.random.randn(100, 10)
    # y only depends on first 2 features
    y = 5 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 0.1, 100)
    
    # Use max_depth=1 to ensure it only picks one (or two) features
    selector = SelectFromModel(GuideTreeRegressor(max_depth=2, significance_threshold=0.05), threshold="mean")
    selector.fit(X, y)
    X_new = selector.transform(X)
    
    print(f"Features selected: {X_new.shape[1]}")
    assert X_new.shape[1] < 10
    assert X_new.shape[1] > 0
    
    # Ensure it picked from the informative ones (0 or 1)
    support = selector.get_support()
    assert support[0] or support[1]