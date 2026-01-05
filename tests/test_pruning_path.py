import numpy as np
from sklearn.datasets import load_iris, make_regression

from pyguide import GuideTreeClassifier, GuideTreeRegressor


def test_pruning_path_classifier():
    X, y = load_iris(return_X_y=True)
    clf = GuideTreeClassifier(max_depth=None, min_samples_split=2)
    clf.fit(X, y)
    
    path = clf.cost_complexity_pruning_path(X, y)
    
    assert "ccp_alphas" in path
    assert "impurities" in path
    
    alphas = path["ccp_alphas"]
    impurities = path["impurities"]
    
    assert len(alphas) == len(impurities)
    # Alphas should be non-negative and increasing
    assert len(alphas) > 1
    assert np.all(alphas >= 0)
    assert np.all(np.diff(alphas) >= 0)
    # Impurities should be increasing as we prune more (larger alpha)
    assert np.all(np.diff(impurities) >= 0)

def test_pruning_path_regressor():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    reg = GuideTreeRegressor(max_depth=None, min_samples_split=2)
    reg.fit(X, y)
    
    path = reg.cost_complexity_pruning_path(X, y)
    
    assert "ccp_alphas" in path
    assert "impurities" in path
    
    alphas = path["ccp_alphas"]
    impurities = path["impurities"]
    
    assert len(alphas) == len(impurities)
    assert np.all(alphas >= 0)
    assert np.all(np.diff(alphas) >= 0)
    assert np.all(np.diff(impurities) >= 0)
