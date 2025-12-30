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
    Also decides where missing values (NaNs) should go.
    """
    best_threshold = None
    best_gain = -1.0
    best_missing_go_left = True

    nan_mask = np.isnan(x)
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

    idx = np.argsort(x_non_nan)
    x_sorted = x_non_nan[idx]
    y_sorted = y_non_nan[idx]

    unique_values, unique_indices = np.unique(x_sorted, return_index=True)

    if len(unique_values) < 2:
        # If all non-nan values are same, we can still decide where NaN goes
        # but that's not a split on x.
        return None, True, 0.0

    for i in range(1, len(unique_values)):
        split_idx = unique_indices[i]

        y_left_non_nan = y_sorted[:split_idx]
        y_right_non_nan = y_sorted[split_idx:]

        # Option 1: Missing go left
        y_left = np.concatenate([y_left_non_nan, y_nan]) if len(y_nan) > 0 else y_left_non_nan
        y_right = y_right_non_nan
        
        gain_left = _calculate_gain(y_left, y_right, current_impurity, n_samples, calc_impurity, criterion)
        
        # Option 2: Missing go right
        y_left = y_left_non_nan
        y_right = np.concatenate([y_right_non_nan, y_nan]) if len(y_nan) > 0 else y_right_non_nan
        
        gain_right = _calculate_gain(y_left, y_right, current_impurity, n_samples, calc_impurity, criterion)

        if gain_left > best_gain or gain_right > best_gain:
            if gain_left >= gain_right:
                best_gain = gain_left
                best_missing_go_left = True
            else:
                best_gain = gain_right
                best_missing_go_left = False
                
            midpoint = (x_sorted[split_idx] + x_sorted[split_idx - 1]) / 2.0
            best_threshold = midpoint

    return best_threshold, best_missing_go_left, best_gain


def _calculate_gain(y_left, y_right, current_impurity, n_samples, calc_impurity, criterion):
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
    Find best categorical split (one-vs-rest).
    Returns (category_to_go_left, missing_go_left, gain).
    """
    best_category = None
    best_gain = -1.0
    best_missing_go_left = True

    # Handle missing values in categorical (None or NaN)
    # x might be object array
    if x.dtype.kind == 'O' or x.dtype.kind == 'U' or x.dtype.kind == 'S':
        nan_mask = pd.isna(x) if hasattr(x, '__len__') else np.array([False]*len(x))
        # Handle the case where x is a numpy array of objects which can have None or np.nan
        if x.dtype.kind == 'O':
            nan_mask = np.array([(v is None or (isinstance(v, float) and np.isnan(v))) for v in x])
    else:
        nan_mask = np.isnan(x)

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

    for cat in unique_categories:
        mask = x_non_nan == cat
        y_left_non_nan = y_non_nan[mask]
        y_right_non_nan = y_non_nan[~mask]

        # Option 1: Missing go left
        y_left = np.concatenate([y_left_non_nan, y_nan]) if len(y_nan) > 0 else y_left_non_nan
        y_right = y_right_non_nan
        gain_left = _calculate_gain(y_left, y_right, current_impurity, n_samples, calc_impurity, criterion)

        # Option 2: Missing go right
        y_left = y_left_non_nan
        y_right = np.concatenate([y_right_non_nan, y_nan]) if len(y_nan) > 0 else y_right_non_nan
        gain_right = _calculate_gain(y_left, y_right, current_impurity, n_samples, calc_impurity, criterion)

        if gain_left > best_gain or gain_right > best_gain:
            if gain_left >= gain_right:
                best_gain = gain_left
                best_missing_go_left = True
            else:
                best_gain = gain_right
                best_missing_go_left = False
            best_category = cat

    return best_category, best_missing_go_left, best_gain


def find_best_split(x, y, is_categorical=False, criterion="gini"):
    """
    Find the best split for feature x.
    Returns (threshold/category, missing_go_left, gain).
    """
    if is_categorical:
        return _find_best_split_categorical(x, y, criterion=criterion)
    else:
        return _find_best_threshold_numerical(x, y, criterion=criterion)
