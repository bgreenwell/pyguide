import numpy as np

from .stats import calc_curvature_test


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
    chi2_stats : ndarray
        The Chi-square statistics for all features.
    """
    n_features = X.shape[1]
    chi2_stats = np.zeros(n_features)
    # We track best p-value separately because we need it for thresholding
    best_p = 1.0
    best_feature_idx = 0

    if categorical_features is None:
        categorical_features = np.zeros(n_features, dtype=bool)

    loop_indices = feature_indices if feature_indices is not None else range(n_features)

    for i in loop_indices:
        is_cat = categorical_features[i]
        stat, p = calc_curvature_test(X[:, i], y, is_categorical=is_cat)
        chi2_stats[i] = stat
        
        if p < best_p:
            best_p = p
            best_feature_idx = i
        elif p == best_p:
            # Tie-breaking logic (optional, currently first wins or random?)
            # Here first wins.
            pass

    return best_feature_idx, best_p, chi2_stats
