import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .node import GuideNode


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

    def _get_categorical_mask(self, X, n_features):
        """Identify categorical features."""
        if self.categorical_features is None:
            if isinstance(X, pd.DataFrame):
                return X.dtypes.isin(["object", "category"]).values
            return np.zeros(n_features, dtype=bool)

        mask = np.zeros(n_features, dtype=bool)
        mask[self.categorical_features] = True
        return mask

    def fit(self, X, y):
        """
        Build a GUIDE tree regressor from the training set (X, y).
        """
        X_orig = X
        X, y = check_X_y(X, y)

        self.n_features_in_ = X.shape[1]
        self._categorical_mask = self._get_categorical_mask(X_orig, self.n_features_in_)

        # Initial stub: Constant prediction (mean)
        self.tree_ = GuideNode(depth=0, is_leaf=True, prediction=np.mean(y))

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict regression target for X.
        """
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.__class__.__name__} is expecting {self.n_features_in_} features as input."
            )

        # For now, just return the constant prediction
        return np.full(X.shape[0], self.tree_.prediction)
