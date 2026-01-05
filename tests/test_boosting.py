import numpy as np
import pytest
from sklearn.datasets import make_regression
from pyguide import GuideGradientBoostingRegressor

def test_gbm_regressor_basic():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    
    gbm = GuideGradientBoostingRegressor(n_estimators=10, max_depth=2, random_state=42)
    gbm.fit(X, y)
    
    assert len(gbm.estimators_) == 10
    assert gbm.init_ is not None
    
    preds = gbm.predict(X)
    assert preds.shape == y.shape
    
    # Check that error is reasonably low (better than mean)
    mse = np.mean((y - preds)**2)
    baseline_mse = np.mean((y - np.mean(y))**2)
    assert mse < baseline_mse * 0.5

def test_gbm_subsample():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    
    # Test with subsampling
    gbm = GuideGradientBoostingRegressor(n_estimators=5, max_depth=2, subsample=0.5, random_state=42)
    gbm.fit(X, y)
    
    assert len(gbm.estimators_) == 5
    preds = gbm.predict(X)
    mse = np.mean((y - preds)**2)
    assert mse < np.mean((y - np.mean(y))**2)
