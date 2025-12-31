import numpy as np
from scipy.stats import chi2_contingency

from .stats import _bin_continuous, _fast_contingency


def calc_interaction_p_value(X_subset, z, categorical_mask=None):
    """
    Calculate interaction p-value between features in X_subset on target z.

    Parameters
    ----------
    X_subset : array-like of shape (n_samples, n_vars)
        The features to test for interaction.
    z : array-like of shape (n_samples,)
        The target values (class labels or residual signs).
    categorical_mask : array-like of shape (n_vars,), optional
        Boolean mask indicating which features in X_subset are categorical.

    GUIDE Strategy:
    - Bin numerical variables into 2 groups (median split).
    - Combine all binned/categorical variables into unique groups.
    - Perform Chi-square test on groups vs z.
    """
    n_samples, n_vars = X_subset.shape
    if categorical_mask is None:
        categorical_mask = np.zeros(n_vars, dtype=bool)

    binned_vars = []
    for i in range(n_vars):
        x = X_subset[:, i]
        is_cat = categorical_mask[i]
        if not is_cat:
            # GUIDE interaction test typically uses 2 bins (median)
            binned_vars.append(_bin_continuous(x, n_bins=2))
        else:
            binned_vars.append(x)

    # 3. Create combined groups
    # To combine binned variables into unique groups, we use return_inverse on unique rows.
    combined = np.column_stack(binned_vars)
    _, combined_idx = np.unique(combined, axis=0, return_inverse=True)

    try:
        contingency = _fast_contingency(combined_idx, z)

        if (
            contingency is None
            or contingency.shape[0] < 2
            or contingency.shape[1] < 2
        ):
            return 1.0

        chi2, p, dof, expected = chi2_contingency(contingency)
        return p
    except Exception:
        return 1.0
