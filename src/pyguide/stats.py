import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

def _bin_continuous(x, n_bins=None):
    """
    Bin a continuous variable into groups based on quartiles (GUIDE style).
    """
    if n_bins is None:
        n_bins = 4 if len(x) >= 40 else 3
    
    try:
        # Use pandas qcut for quartile-based binning
        # duplicates='drop' handles cases with many identical values
        return pd.qcut(x, q=n_bins, labels=False, duplicates='drop')
    except (ValueError, IndexError):
        # Fallback to simple uniform binning if qcut fails
        # (e.g., if all values are identical or nearly so)
        return np.zeros_like(x, dtype=int)

def calc_curvature_p_value(x, z, is_categorical=False):
    """
    Calculate the Chi-square p-value for the association between x and z.
    
    Parameters
    ----------
    x : array-like
        The predictor variable.
    z : array-like
        The target variable (categorical labels).
    is_categorical : bool
        Whether x is categorical. If False, x will be binned.
        
    Returns
    -------
    p : float
        The p-value from the Chi-square test. Returns 1.0 on failure.
    """
    # 1. Prepare binned/categorical x
    if not is_categorical:
        x_processed = _bin_continuous(x)
    else:
        x_processed = x
        
    # 2. Create contingency table
    # We use pandas crosstab for convenience, but could use numpy if needed for performance
    try:
        contingency = pd.crosstab(x_processed, z)
        
        # 3. Chi-square test
        # If the contingency table is too small (e.g., only one row/col), chi2 fails
        if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return 1.0
            
        chi2, p, dof, expected = chi2_contingency(contingency)
        return p
    except Exception:
        return 1.0
