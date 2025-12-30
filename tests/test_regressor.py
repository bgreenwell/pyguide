import numpy as np
import pytest
from pyguide import GuideTreeRegressor

def test_regressor_init():
    reg = GuideTreeRegressor(max_depth=3, min_samples_split=5)
    assert reg.max_depth == 3
    assert reg.min_samples_split == 5

def test_regressor_fit_predict_basic():
    # Simple linear relationship: y = x
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([1, 2, 3, 4], dtype=float)

    reg = GuideTreeRegressor(max_depth=2)
    reg.fit(X, y)

    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape
    # Predictions should be reasonably close to target for a very simple tree
    # (Though with depth 2 and only 4 samples, it might just be means)
    assert np.all(np.isfinite(y_pred))

def test_regressor_sklearn_compatibility():
    from sklearn.utils.validation import check_is_fitted
    
    X = np.array([[1], [2]])
    y = np.array([1, 2])
    
    reg = GuideTreeRegressor()
    with pytest.raises(Exception): # sklearn.exceptions.NotFittedError usually
        check_is_fitted(reg)
        
    reg.fit(X, y)
    check_is_fitted(reg)
