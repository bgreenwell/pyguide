import numpy as np
from scipy.stats import chi2_contingency

from .stats import _bin_continuous, _fast_contingency


def calc_interaction_p_value(x1, x2, z, is_cat1=False, is_cat2=False):
    """
    Calculate interaction p-value between x1 and x2 on target z.

    GUIDE Strategy:
    - If both numerical: Split into 4 quadrants based on medians.
    - If categorical: Use categories.
    - If mixed: Bin numerical part.

    Then perform Chi-square test on the groups formed by (x1, x2) vs z.
    """
    # 1. Discretize x1
    if not is_cat1:
        # Use median split (2 bins) for interaction test usually
        # But _bin_continuous uses quartiles (4 bins) or 3.
        # GUIDE interaction test typically uses 2 bins (median) to form 4 quadrants total.
        x1_binned = _bin_continuous(x1, n_bins=2)
    else:
        x1_binned = x1

    # 2. Discretize x2
    if not is_cat2:
        x2_binned = _bin_continuous(x2, n_bins=2)
    else:
        x2_binned = x2

    # 3. Create combined groups
    # To combine x1_binned and x2_binned into unique pairs, we can use integer labels.
    # We use return_inverse to get unique codes for each pair.
    # Stack and find unique rows
    combined = np.column_stack([x1_binned, x2_binned])
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
