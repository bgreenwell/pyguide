import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .interactions import calc_interaction_p_value
from .node import GuideNode
from .selection import select_split_variable
from .splitting import find_best_split


class GuideTreeRegressor(RegressorMixin, BaseEstimator):
    """
    GUIDE (Generalized, Unbiased, Interaction Detection and Estimation) Tree Regressor.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        significance_threshold=0.05,
        interaction_depth=1,
        categorical_features=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.significance_threshold = significance_threshold
        self.interaction_depth = interaction_depth
        self.categorical_features = categorical_features

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.categorical = self.categorical_features is not None
        return tags

    def _get_categorical_mask(self, X, n_features):
        """Identify categorical features."""
        if self.categorical_features is None:
            if isinstance(X, pd.DataFrame):
                return X.dtypes.isin(["object", "category"]).values
            if hasattr(X, "dtype") and X.dtype.kind in ["O", "U", "S"]:
                return np.ones(n_features, dtype=bool)
            return np.zeros(n_features, dtype=bool)

        mask = np.zeros(n_features, dtype=bool)
        mask[self.categorical_features] = True
        return mask

    def fit(self, X, y):
        """
        Build a GUIDE tree regressor from the training set (X, y).
        """
        X_orig = X

        if self.categorical_features is not None or isinstance(X, pd.DataFrame):
            dtype = None
        elif hasattr(X, "dtype") and X.dtype.kind in ["U", "S"]:
            dtype = None
        else:
            dtype = "numeric"

        X, y = check_X_y(X, y, dtype=dtype, force_all_finite="allow-nan")

        self.n_features_in_ = X.shape[1]
        self._categorical_mask = self._get_categorical_mask(X_orig, self.n_features_in_)

        # Build the tree
        self.tree_ = self._fit_node(X, y, depth=0)

        self.is_fitted_ = True
        return self

    def _calculate_lookahead_gain(self, X, y, split_feat, next_feat):
        """
        Calculate total gain of splitting on split_feat, then splitting children on next_feat.
        Using SSE criterion for regression.
        """
        is_cat = self._categorical_mask[split_feat]
        threshold, missing_go_left, gain1 = find_best_split(
            X[:, split_feat], y, is_categorical=is_cat, criterion="mse"
        )

        if threshold is None:
            return 0.0

        if is_cat:
            left_mask = X[:, split_feat] == threshold
        else:
            left_mask = X[:, split_feat] <= threshold
            
        # Handle NaNs
        nan_mask = np.isnan(X[:, split_feat]) if not is_cat else pd.isna(X[:, split_feat])
        if is_cat and X.dtype.kind == 'O':
             nan_mask = np.array([(v is None or (isinstance(v, float) and np.isnan(v))) for v in X[:, split_feat]])

        if missing_go_left:
            left_mask = left_mask | nan_mask
        else:
            left_mask = left_mask & ~nan_mask

        y_left = y[left_mask]
        y_right = y[~left_mask]
        X_left = X[left_mask]
        X_right = X[~left_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            return gain1

        # Gain from second level (next_feat)
        is_cat_next = self._categorical_mask[next_feat]
        _, _, gain2_left = find_best_split(
            X_left[:, next_feat], y_left, is_categorical=is_cat_next, criterion="mse"
        )
        _, _, gain2_right = find_best_split(
            X_right[:, next_feat], y_right, is_categorical=is_cat_next, criterion="mse"
        )

        total_gain = gain1 + gain2_left + gain2_right
        return total_gain

    def _fit_node(self, X, y, depth):
        """Recursive function to grow the tree for regression."""
        n_samples = len(y)
        prediction = np.mean(y)

        # 1. Check stopping criteria
        # Use a small tolerance for constant y
        if (
            (n_samples > 0 and np.all(np.abs(y - prediction) < 1e-9))
            or n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return GuideNode(depth=depth, is_leaf=True, prediction=prediction)

        # 2. Variable Selection (GUIDE step 1)
        # Residual-based target for regression
        z = (y > prediction).astype(int)

        best_idx, p = select_split_variable(
            X, z, categorical_features=self._categorical_mask
        )

        # Interaction Detection (Fallback)
        interaction_split_override = False
        if p > self.significance_threshold and self.interaction_depth > 0:
            best_int_p = 1.0
            best_int_pair = None
            n_features = X.shape[1]

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    p_int = calc_interaction_p_value(
                        X[:, i],
                        X[:, j],
                        z,
                        is_cat1=self._categorical_mask[i],
                        is_cat2=self._categorical_mask[j],
                    )
                    if p_int < best_int_p:
                        best_int_p = p_int
                        best_int_pair = (i, j)

            if best_int_p < self.significance_threshold:
                i, j = best_int_pair
                gain_i_then_j = self._calculate_lookahead_gain(X, y, i, j)
                gain_j_then_i = self._calculate_lookahead_gain(X, y, j, i)

                if gain_i_then_j >= gain_j_then_i:
                    best_idx = i
                else:
                    best_idx = j
                interaction_split_override = True

        # Check significance threshold
        if not interaction_split_override and p > self.significance_threshold:
            return GuideNode(depth=depth, is_leaf=True, prediction=prediction)

        # 3. Split Point Optimization (GUIDE step 2)
        is_cat = self._categorical_mask[best_idx]
        threshold, missing_go_left, gain = find_best_split(
            X[:, best_idx], y, is_categorical=is_cat, criterion="mse"
        )

        # 4. If no valid split found, return leaf
        if threshold is None or (gain <= 0 and not interaction_split_override):
            return GuideNode(depth=depth, is_leaf=True, prediction=prediction)

        # 5. Create node and recurse
        node = GuideNode(
            depth=depth,
            split_feature=best_idx,
            split_threshold=threshold,
            missing_go_left=missing_go_left,
        )

        if is_cat:
            left_mask = X[:, best_idx] == threshold
        else:
            left_mask = X[:, best_idx] <= threshold
            
        # Handle NaNs
        nan_mask = np.isnan(X[:, best_idx]) if not is_cat else pd.isna(X[:, best_idx])
        if is_cat and X.dtype.kind == 'O':
             nan_mask = np.array([(v is None or (isinstance(v, float) and np.isnan(v))) for v in X[:, best_idx]])

        if missing_go_left:
            left_mask = left_mask | nan_mask
        else:
            left_mask = left_mask & ~nan_mask

        node.left = self._fit_node(X[left_mask], y[left_mask], depth + 1)
        node.right = self._fit_node(X[~left_mask], y[~left_mask], depth + 1)

        return node

    def predict(self, X):
        """
        Predict regression target for X.
        """
        check_is_fitted(self)

        dtype = (
            None
            if (
                self.categorical_features is not None
                or (
                    hasattr(self, "_categorical_mask")
                    and np.any(self._categorical_mask)
                )
            )
            else "numeric"
        )

        X = check_array(X, dtype=dtype, force_all_finite="allow-nan")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.__class__.__name__} is expecting {self.n_features_in_} features as input."
            )

        return np.array([self._predict_single(x, self.tree_) for x in X])

    def _predict_single(self, x, node):
        """Predict for a single sample by traversing the tree."""
        if node.is_leaf:
            return node.prediction

        is_cat = self._categorical_mask[node.split_feature]

        if is_cat:
            go_left = x[node.split_feature] == node.split_threshold
        else:
            go_left = x[node.split_feature] <= node.split_threshold

        if go_left:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
