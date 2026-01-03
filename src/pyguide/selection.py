import numpy as np

from .stats import calc_curvature_p_value


def select_split_variable(X, y, categorical_features=None, feature_indices=None):
    """
    Select the best variable to split on based on curvature tests (Chi-square).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The training input samples.
    y : array-like of shape (n_samples,)
        The target values (class labels).
    categorical_features : array-like of shape (n_features,), optional
        Boolean mask indicating which features are categorical.
    feature_indices : array-like, optional
        Indices of features to consider. If None, consider all features.

    Returns
    -------
    best_feature_idx : int
        The index of the selected feature.
    best_p : float
        The p-value of the selected feature.
    """
    n_features = X.shape[1]
    p_values = np.ones(n_features)

    if categorical_features is None:
        categorical_features = np.zeros(n_features, dtype=bool)

    loop_indices = feature_indices if feature_indices is not None else range(n_features)

    for i in loop_indices:
        is_cat = categorical_features[i]
        p_values[i] = calc_curvature_p_value(X[:, i], y, is_categorical=is_cat)

    best_feature_idx = np.argmin(p_values)
    best_p = p_values[best_feature_idx]

    return best_feature_idx, best_p, p_values
