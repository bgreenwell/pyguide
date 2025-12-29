import numpy as np
import pytest
from pyguide.interactions import calc_interaction_p_value

def test_interaction_xor_pattern():
    # XOR pattern:
    # x1  x2  y
    # 0   0   0
    # 0   1   1
    # 1   0   1
    # 1   1   0
    # Main effects are 0 (uncorrelated individually).
    # Interaction is strong.
    
    x1 = np.array([0, 0, 1, 1] * 10)
    x2 = np.array([0, 1, 0, 1] * 10)
    y  = np.array([0, 1, 1, 0] * 10)
    
    # Check interaction between x1 and x2
    p = calc_interaction_p_value(x1, x2, y, is_cat1=True, is_cat2=True)
    assert p < 0.05

def test_no_interaction():
    # Random data
    np.random.seed(42)
    x1 = np.random.rand(100)
    x2 = np.random.rand(100)
    y = np.random.randint(0, 2, 100)
    
    p = calc_interaction_p_value(x1, x2, y, is_cat1=False, is_cat2=False)
    assert p > 0.05
