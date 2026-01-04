import numpy as np
import pandas as pd


def _gini(y):
    """Calculate Gini impurity of a vector of class labels."""
    n = len(y)
    if n == 0:
        return 0.0
    # Use bincount for efficiency if y are integers
    if np.issubdtype(y.dtype, np.integer):
        counts = np.bincount(y)
    else:
        _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return 1.0 - np.sum(probs**2)


def _gini_from_counts(counts, n):
    """Fast Gini from counts."""
    if n == 0:
        return 0.0
    probs = counts / n
    return 1.0 - np.sum(probs**2)


def _sse(y):
    """Calculate Sum of Squared Errors of a vector y."""
    n = len(y)
    if n == 0:
        return 0.0
    return np.sum(y**2) - (np.sum(y) ** 2) / n


def _sse_from_stats(sum_y, sum_y2, n):
    """Fast SSE from sums."""
    if n == 0:
        return 0.0
    return sum_y2 - (sum_y**2) / n


def _find_best_threshold_numerical(x, y, criterion="gini"):
    """
    Find the best split threshold for a numerical feature x to separate y.
    Optimized O(N log N) implementation using cumulative stats.
    """
    nan_mask = pd.isna(x)
    x_non_nan = x[~nan_mask]
    y_non_nan = y[~nan_mask]
    y_nan = y[nan_mask]
    n_total = len(y)
    n_nan = len(y_nan)

    if len(x_non_nan) == 0:
        return None, True, 0.0

    # Sort data
    idx = np.argsort(x_non_nan)
    x_sorted = x_non_nan[idx]
    y_sorted = y_non_nan[idx]

    # Find unique values and their split indices
    unique_vals, split_indices = np.unique(x_sorted, return_index=True)
    if len(unique_vals) < 2:
        return None, True, 0.0
    # remove the first index (0) as we split AFTER values
    split_indices = split_indices[1:]

    best_threshold = None
    best_gain = -1.0
    best_missing_go_left = True

    if criterion == "gini":
        current_impurity = _gini(y)
        # Precompute class counts
        n_classes = np.max(y) + 1 if len(y) > 0 else 0
        total_counts = np.bincount(y, minlength=n_classes)
        nan_counts = np.bincount(y_nan, minlength=n_classes) if n_nan > 0 else np.zeros(n_classes)
        
        # Cumulative counts from the left
        # We need counts for each split point
        # y_sorted is (N_non_nan,)
        y_one_hot = np.zeros((len(y_sorted), n_classes))
        y_one_hot[np.arange(len(y_sorted)), y_sorted] = 1
        cum_counts = np.cumsum(y_one_hot, axis=0)
        
        for split_idx in split_indices:
            left_counts_non_nan = cum_counts[split_idx - 1]
            right_counts_non_nan = total_counts - nan_counts - left_counts_non_nan
            n_left_non_nan = split_idx
            n_right_non_nan = n_total - n_nan - split_idx

            # Option 1: Missing go left
            n_left = n_left_non_nan + n_nan
            n_right = n_right_non_nan
            if n_left > 0 and n_right > 0:
                imp_left = _gini_from_counts(left_counts_non_nan + nan_counts, n_left)
                imp_right = _gini_from_counts(right_counts_non_nan, n_right)
                gain = current_impurity - (n_left/n_total * imp_left + n_right/n_total * imp_right)
                if gain > best_gain:
                    best_gain = gain
                    best_missing_go_left = True
                    best_threshold = (x_sorted[split_idx] + x_sorted[split_idx-1]) / 2.0

            # Option 2: Missing go right
            n_left = n_left_non_nan
            n_right = n_right_non_nan + n_nan
            if n_left > 0 and n_right > 0:
                imp_left = _gini_from_counts(left_counts_non_nan, n_left)
                imp_right = _gini_from_counts(right_counts_non_nan + nan_counts, n_right)
                gain = current_impurity - (n_left/n_total * imp_left + n_right/n_total * imp_right)
                if gain > best_gain:
                    best_gain = gain
                    best_missing_go_left = False
                    best_threshold = (x_sorted[split_idx] + x_sorted[split_idx-1]) / 2.0
    else:
        # SSE Optimization
        current_impurity = _sse(y)
        sum_y_total = np.sum(y)
        sum_y2_total = np.sum(y**2)
        sum_y_nan = np.sum(y_nan)
        sum_y2_nan = np.sum(y_nan**2)
        
        cum_sum_y = np.cumsum(y_sorted)
        cum_sum_y2 = np.cumsum(y_sorted**2)
        
        for split_idx in split_indices:
            sum_y_l_nn = cum_sum_y[split_idx - 1]
            sum_y2_l_nn = cum_sum_y2[split_idx - 1]
            sum_y_r_nn = sum_y_total - sum_y_nan - sum_y_l_nn
            sum_y2_r_nn = sum_y2_total - sum_y2_nan - sum_y2_l_nn
            
            n_l_nn = split_idx
            n_r_nn = n_total - n_nan - split_idx

            # Option 1: Missing go left
            n_l = n_l_nn + n_nan
            n_r = n_r_nn
            if n_l > 0 and n_r > 0:
                imp_l = _sse_from_stats(sum_y_l_nn + sum_y_nan, sum_y2_l_nn + sum_y2_nan, n_l)
                imp_r = _sse_from_stats(sum_y_r_nn, sum_y2_r_nn, n_r)
                gain = current_impurity - (imp_l + imp_r)
                if gain > best_gain:
                    best_gain = gain
                    best_missing_go_left = True
                    best_threshold = (x_sorted[split_idx] + x_sorted[split_idx-1]) / 2.0

            # Option 2: Missing go right
            n_l = n_l_nn
            n_r = n_r_nn + n_nan
            if n_l > 0 and n_r > 0:
                imp_l = _sse_from_stats(sum_y_l_nn, sum_y2_l_nn, n_l)
                imp_r = _sse_from_stats(sum_y_r_nn + sum_y_nan, sum_y2_r_nn + sum_y2_nan, n_r)
                gain = current_impurity - (imp_l + imp_r)
                if gain > best_gain:
                    best_gain = gain
                    best_missing_go_left = False
                    best_threshold = (x_sorted[split_idx] + x_sorted[split_idx-1]) / 2.0

    return best_threshold, best_missing_go_left, best_gain


def _calculate_gain(
    y_left, y_right, current_impurity, n_samples, calc_impurity, criterion
):
    if len(y_left) == 0 or len(y_right) == 0:
        return -1.0

    imp_left = calc_impurity(y_left)
    imp_right = calc_impurity(y_right)

    if criterion == "gini":
        n_left = len(y_left)
        n_right = len(y_right)
        weighted_impurity = (n_left / n_samples) * imp_left + (
            n_right / n_samples
        ) * imp_right
        return current_impurity - weighted_impurity
    else:
        return current_impurity - (imp_left + imp_right)


def _find_best_split_categorical(x, y, criterion="gini"):
    """
    Find best categorical split.
    For regression and binary classification, we use the 'ordered' heuristic:
    sort categories by their mean target value.
    Returns (set_of_categories_to_go_left, missing_go_left, gain).
    """
    best_categories = None
    best_gain = -1.0
    best_missing_go_left = True

    # Handle missing values in categorical (None or NaN)
    if x.dtype.kind == "O" or x.dtype.kind == "U" or x.dtype.kind == "S":
        # Need to be careful with pd.isna on object arrays containing various types
        nan_mask = pd.isna(x)
    else:
        nan_mask = pd.isna(x)

    y_nan = y[nan_mask]
    x_non_nan = x[~nan_mask]
    y_non_nan = y[~nan_mask]

    if len(x_non_nan) == 0:
        return None, True, 0.0

    if criterion == "gini":
        current_impurity = _gini(y)
        calc_impurity = _gini
    else:
        current_impurity = _sse(y)
        calc_impurity = _sse

    n_samples = len(y)
    unique_categories = np.unique(x_non_nan)

    if len(unique_categories) < 2:
        return None, True, 0.0

    # 1. Calculate mean target for each category
    cat_means = []
    for cat in unique_categories:
        cat_means.append(np.mean(y_non_nan[x_non_nan == cat]))

    # 2. Sort categories by mean
    sorted_idx = np.argsort(cat_means)
    sorted_cats = unique_categories[sorted_idx]

    # 3. Evaluate splits along the sorted order (K-1 possible splits)
    # This is optimal for MSE and Gini with binary targets.
    for i in range(1, len(sorted_cats)):
        left_cats = set(sorted_cats[:i])

        mask = np.array([val in left_cats for val in x_non_nan])
        y_left_non_nan = y_non_nan[mask]
        y_right_non_nan = y_non_nan[~mask]

        # Option 1: Missing go left
        y_left = (
            np.concatenate([y_left_non_nan, y_nan])
            if len(y_nan) > 0
            else y_left_non_nan
        )
        y_right = y_right_non_nan
        gain_left = _calculate_gain(
            y_left, y_right, current_impurity, n_samples, calc_impurity, criterion
        )

        # Option 2: Missing go right
        y_left = y_left_non_nan
        y_right = (
            np.concatenate([y_right_non_nan, y_nan])
            if len(y_nan) > 0
            else y_right_non_nan
        )
        gain_right = _calculate_gain(
            y_left, y_right, current_impurity, n_samples, calc_impurity, criterion
        )

        if gain_left > best_gain or gain_right > best_gain:
            if gain_left >= gain_right:
                best_gain = gain_left
                best_missing_go_left = True
            else:
                best_gain = gain_right
                best_missing_go_left = False
            best_categories = left_cats

    return best_categories, best_missing_go_left, best_gain


def find_best_split(x, y, is_categorical=False, criterion="gini"):
    """
    Find the best split for feature x.
    Returns (threshold/category, missing_go_left, gain).
    """
    if is_categorical:
        return _find_best_split_categorical(x, y, criterion=criterion)
    else:
        return _find_best_threshold_numerical(x, y, criterion=criterion)
