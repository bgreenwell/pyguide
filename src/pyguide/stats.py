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


def _fast_contingency(x, z):
    """
    Fast contingency table creation using numpy.
    x and z should be integer arrays.
    Returns a 2D numpy array.
    """
    # 1. Map categories to [0, n_cats-1] and [0, n_classes-1]
    # np.unique with return_inverse is still relatively slow,
    # but we only call it once per variable selection.
    ux, x_idx = np.unique(x, return_inverse=True)
    uz, z_idx = np.unique(z, return_inverse=True)

    n_x = len(ux)
    n_z = len(uz)

    if n_x < 2 or n_z < 2:
        return None

    # Use bincount for O(N) contingency table
    # index = x_idx * n_z + z_idx
    count = np.bincount(x_idx * n_z + z_idx, minlength=n_x * n_z)
    return count.reshape(n_x, n_z)


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
        # Categorical x: ensure NaNs are represented
        if x.dtype.kind in ["O", "U", "S"]:
            # Fill NaNs manually to avoid pandas Series creation
            x_processed = x.copy()
            mask = pd.isna(x)
            if np.any(mask):
                x_processed[mask] = "MISSING"
        else:
            nan_mask = np.isnan(x)
            x_processed = x.copy()
            if len(x) > 0:
                # Use a value that doesn't exist in x
                missing_val = np.nanmin(x) - 1 if not np.all(np.isnan(x)) else -1
                x_processed[nan_mask] = missing_val

    # 2. Create contingency table
    contingency = _fast_contingency(x_processed, z)

    if contingency is None:
        return 1.0

    # 3. Chi-square test
    try:
        # Use Fisher's exact test for 2x2 tables
        if contingency.shape == (2, 2):
            _, p = fisher_exact(contingency)
            return p

        chi2, p, dof, expected = chi2_contingency(contingency)
        return p
    except Exception:
        return 1.0
