import numpy as np
import pytest
from pyguide.selection import select_split_variable

def test_select_split_variable_simple():
    # X has 3 features. Feature 1 (index 0) is perfectly correlated with y.
    # Feature 2 and 3 are random noise.
    np.random.seed(42)
    n_samples = 100
    x0 = np.array([0]*50 + [1]*50)
    x1 = np.random.rand(n_samples)
    x2 = np.random.rand(n_samples)
    X = np.column_stack([x0, x1, x2])
    y = x0.copy()
    
    best_idx, best_p = select_split_variable(X, y)
    assert best_idx == 0
    assert best_p < 0.05

def test_select_split_variable_all_noise():
    # All features are random noise.
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, 100)
    
    best_idx, best_p = select_split_variable(X, y)
    # Even with noise, one will have the "best" p-value, but it should be relatively high
    assert 0 <= best_idx < 3
    assert best_p >= 0 # p-value is always non-negative
