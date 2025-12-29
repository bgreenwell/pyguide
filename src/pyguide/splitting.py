import numpy as np


def _gini(y):
    """Calculate Gini impurity of a vector of class labels."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs**2)


def _find_best_threshold_numerical(x, y):
    """
    Find the best split threshold for a numerical feature x to separate y.
    Minimizes Gini impurity.
    """
    best_threshold = None
    best_gain = -1.0
    current_impurity = _gini(y)
    n_samples = len(y)

    # Sort x and rearrange y accordingly
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]

    # Iterate through possible split points
    # We only care about changes in value
    unique_values, unique_indices = np.unique(x_sorted, return_index=True)

    if len(unique_values) < 2:
        return None, 0.0

    # We test thresholds between unique values
    # unique_indices points to the first occurrence of each value in sorted array
    # We skip the first one because we can't split before the first value

    for i in range(1, len(unique_values)):
        split_idx = unique_indices[i]

        # y_left = y_sorted[:split_idx]
        # y_right = y_sorted[split_idx:]

        # Optimization: maintain counts instead of recomputing Gini from scratch
        # But for MVP, simple recomputation is safer and cleaner
        y_left = y_sorted[:split_idx]
        y_right = y_sorted[split_idx:]

        gini_left = _gini(y_left)
        gini_right = _gini(y_right)

        n_left = len(y_left)
        n_right = len(y_right)

        weighted_impurity = (n_left / n_samples) * gini_left + (
            n_right / n_samples
        ) * gini_right
        gain = current_impurity - weighted_impurity

        if gain > best_gain:
            best_gain = gain
            # Threshold is midpoint between current value and previous value
            midpoint = (x_sorted[split_idx] + x_sorted[split_idx - 1]) / 2.0
            best_threshold = midpoint

    return best_threshold, best_gain


def _find_best_split_categorical(x, y):
    """
    Find best categorical split (one-vs-rest).
    Returns the category to go left.
    """
    best_category = None
    best_gain = -1.0
    current_impurity = _gini(y)
    n_samples = len(y)

    unique_categories = np.unique(x)

    if len(unique_categories) < 2:
        return None, 0.0

    for cat in unique_categories:
        # Split: x == cat (Left) vs x != cat (Right)
        mask = x == cat
        y_left = y[mask]
        y_right = y[~mask]

        if len(y_left) == 0 or len(y_right) == 0:
            continue

        gini_left = _gini(y_left)
        gini_right = _gini(y_right)

        n_left = len(y_left)
        n_right = len(y_right)

        weighted_impurity = (n_left / n_samples) * gini_left + (
            n_right / n_samples
        ) * gini_right
        gain = current_impurity - weighted_impurity

        if gain > best_gain:
            best_gain = gain
            best_category = cat

    return best_category, best_gain


def find_best_split(x, y, is_categorical=False):
    """
    Find the best split for feature x.
    """
    if is_categorical:
        return _find_best_split_categorical(x, y)
    else:
        return _find_best_threshold_numerical(x, y)
