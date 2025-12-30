import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact


def _bin_continuous(x, n_bins=None):
    """
    Bin a continuous variable into groups based on quartiles (GUIDE style).
    """
    unique_values = np.unique(x)

    if n_bins is None:
        if len(x) >= 40:
            n_bins = 4
        elif len(x) >= 3:
            n_bins = 3
        else:
            # Too few samples to bin effectively, return ranks to preserve order
            return np.argsort(np.argsort(x))

    # If the feature has very few unique values, don't bin it using quartiles
    # as it might collapse into a single bin. Instead, map to unique ranks.
    if len(unique_values) <= n_bins:
        # Map values to their rank (0, 1, 2...)
        mapping = {val: i for i, val in enumerate(unique_values)}
        return np.array([mapping[v] for v in x])

    try:
        # Use pandas qcut for quartile-based binning
        # duplicates='drop' handles cases with many identical values
        return pd.qcut(x, q=n_bins, labels=False, duplicates="drop")
    except (ValueError, IndexError):
        # Fallback to simple uniform binning if qcut fails
        return np.zeros_like(x, dtype=int)


def calc_curvature_p_value(x, z, is_categorical=False):
    """
    Calculate the Chi-square p-value for the association between x and z.
    Missing values in x are treated as a separate category.
    """
    # 1. Prepare binned/categorical x
    if not is_categorical:
        # Separate NaNs from continuous values for binning
        nan_mask = np.isnan(x)
        x_non_nan = x[~nan_mask]

        if len(x_non_nan) > 0:
            x_binned = _bin_continuous(x_non_nan)
            # Reconstruct x_processed with NaNs as a separate bin (e.g., -1)
            x_processed = np.full(len(x), -1, dtype=int)
            x_processed[~nan_mask] = x_binned
        else:
            x_processed = np.full(len(x), -1, dtype=int)
    else:
        # Categorical x: ensure NaNs are represented (e.g., as 'MISSING' string)
        if x.dtype.kind == "O" or x.dtype.kind == "U" or x.dtype.kind == "S":
            # Use pandas to handle various missing representations
            x_processed = pd.Series(x).fillna("MISSING")
        else:
            nan_mask = np.isnan(x)
            x_processed = x.copy()
            # Use a value that doesn't exist in x
            if len(x) > 0:
                missing_val = np.nanmin(x) - 1 if not np.all(np.isnan(x)) else -1
                x_processed[nan_mask] = missing_val
            else:
                x_processed = x

    # 2. Create contingency table
    try:
        contingency = pd.crosstab(x_processed, z)

        # 3. Chi-square test
        # If the contingency table is too small (e.g., only one row/col), chi2 fails
        if (
            contingency.size == 0
            or contingency.shape[0] < 2
            or contingency.shape[1] < 2
        ):
            return 1.0

        # Use Fisher's exact test for 2x2 tables (more robust for small samples)
        if contingency.shape == (2, 2):
            _, p = fisher_exact(contingency)
            return p

        chi2, p, dof, expected = chi2_contingency(contingency)
        return p
    except Exception:
        return 1.0
