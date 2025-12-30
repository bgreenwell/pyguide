import numpy as np


def _gini(y):
    """Calculate Gini impurity of a vector of class labels."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs**2)


def _sse(y):
    """Calculate Sum of Squared Errors of a vector y."""
    if len(y) == 0:
        return 0.0
    return np.sum((y - np.mean(y)) ** 2)


def _find_best_threshold_numerical(x, y, criterion="gini"):
    """
    Find the best split threshold for a numerical feature x to separate y.
    """
    best_threshold = None
    best_gain = -1.0

    if criterion == "gini":
        current_impurity = _gini(y)
        calc_impurity = _gini
    else:
        current_impurity = _sse(y)
        calc_impurity = _sse

    n_samples = len(y)

    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]

    unique_values, unique_indices = np.unique(x_sorted, return_index=True)

    if len(unique_values) < 2:
        return None, 0.0

    for i in range(1, len(unique_values)):
        split_idx = unique_indices[i]

        y_left = y_sorted[:split_idx]
        y_right = y_sorted[split_idx:]

        imp_left = calc_impurity(y_left)
        imp_right = calc_impurity(y_right)

        if criterion == "gini":
            n_left = len(y_left)
            n_right = len(y_right)
            weighted_impurity = (n_left / n_samples) * imp_left + (
                n_right / n_samples
            ) * imp_right
            gain = current_impurity - weighted_impurity
        else:
            # For SSE, gain is total reduction in SSE: SSE_total - (SSE_left + SSE_right)
            gain = current_impurity - (imp_left + imp_right)

        if gain > best_gain:
            best_gain = gain
            midpoint = (x_sorted[split_idx] + x_sorted[split_idx - 1]) / 2.0
            best_threshold = midpoint

    return best_threshold, best_gain


def _find_best_split_categorical(x, y, criterion="gini"):
    """
    Find best categorical split (one-vs-rest).
    Returns the category to go left.
    """
    best_category = None
    best_gain = -1.0

    if criterion == "gini":
        current_impurity = _gini(y)
        calc_impurity = _gini
    else:
        current_impurity = _sse(y)
        calc_impurity = _sse

    n_samples = len(y)

    unique_categories = np.unique(x)

    if len(unique_categories) < 2:
        return None, 0.0

    for cat in unique_categories:
        mask = x == cat
        y_left = y[mask]
        y_right = y[~mask]

        if len(y_left) == 0 or len(y_right) == 0:
            continue

        imp_left = calc_impurity(y_left)
        imp_right = calc_impurity(y_right)

        if criterion == "gini":
            n_left = len(y_left)
            n_right = len(y_right)
            weighted_impurity = (n_left / n_samples) * imp_left + (
                n_right / n_samples
            ) * imp_right
            gain = current_impurity - weighted_impurity
        else:
            gain = current_impurity - (imp_left + imp_right)

        if gain > best_gain:
            best_gain = gain
            best_category = cat

    return best_category, best_gain


def find_best_split(x, y, is_categorical=False, criterion="gini"):
    """
    Find the best split for feature x.
    """
    if is_categorical:
        return _find_best_split_categorical(x, y, criterion=criterion)
    else:
        return _find_best_threshold_numerical(x, y, criterion=criterion)
