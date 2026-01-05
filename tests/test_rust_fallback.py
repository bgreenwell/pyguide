import numpy as np
import pytest
from unittest.mock import patch
import pyguide.stats as stats
import pyguide.splitting as splitting

def test_stats_python_fallback():
    # Force HAS_RUST to False for this test
    with patch("pyguide.stats.HAS_RUST", False):
        x = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0])
        z = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        # Test binning
        binned = stats._bin_continuous(x, n_bins=4)
        assert len(binned) == len(x)
        
        # Test contingency
        contingency = stats._fast_contingency(binned, z)
        assert contingency.shape == (4, 2)
        
        # Test chi2
        stat, p = stats._chi2_test(contingency)
        assert p < 0.1

def test_splitting_python_fallback():
    # Force HAS_RUST to False for this test
    with patch("pyguide.splitting.HAS_RUST", False):
        x = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        # Test numerical split
        threshold, missing_go_left, gain = splitting._find_best_threshold_numerical(x, y, criterion="gini")
        assert threshold is not None
        assert 4.0 < threshold < 10.0
        assert gain > 0
        
        # Test SSE split
        y_reg = x * 2.0
        threshold, missing_go_left, gain = splitting._find_best_threshold_numerical(x, y_reg, criterion="mse")
        assert threshold is not None
        assert gain > 0
