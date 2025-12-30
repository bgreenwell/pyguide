import numpy as np
import pytest
from pyguide.regressor import GuideTreeRegressor

def test_residual_to_class_logic():
    # Residuals: r = y - mean(y)
    y = np.array([10, 20, 30, 40, 50], dtype=float)
    mean_y = np.mean(y) # 30
    
    # Expected z: 
    # 10 - 30 = -20 (z=0)
    # 20 - 30 = -10 (z=0)
    # 30 - 30 = 0   (z=0)
    # 40 - 30 = 10  (z=1)
    # 50 - 30 = 20  (z=1)
    
    reg = GuideTreeRegressor()
    # We'll need to expose or test the logic that will be in _fit_node
    # For now, let's just test the logic itself if we implement it as a helper
    
    residuals = y - mean_y
    z = (residuals > 0).astype(int)
    
    expected_z = np.array([0, 0, 0, 1, 1])
    assert np.array_equal(z, expected_z)

def test_regressor_variable_selection_integration():
    # Simple case where one variable is clearly predictive
    # X0 is predictive, X1 is noise
    # If we split on X0, we reduce residuals
    X = np.array([
        [0, 10],
        [0, 20],
        [1, 10],
        [1, 20]
    ], dtype=float)
    y = np.array([10, 10, 50, 50], dtype=float)
    
    # y_mean = 30
    # residuals = [-20, -20, 20, 20]
    # z = [0, 0, 1, 1]
    # z is perfectly correlated with X0
    
    reg = GuideTreeRegressor(max_depth=1, significance_threshold=1.0)
    reg.fit(X, y)
    
    # After implementation, reg.tree_ should not be a leaf with constant mean
    # it should have a split_feature.
    # (Currently it's a stub, so this test will fail)
    assert reg.tree_.is_leaf is False
    assert reg.tree_.split_feature == 0
