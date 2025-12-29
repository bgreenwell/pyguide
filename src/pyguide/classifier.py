import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets

from .node import GuideNode
from .selection import select_split_variable
from .splitting import find_best_split

class GuideTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    GUIDE (Generalized, Unbiased, Interaction Detection and Estimation) Tree Classifier.
    """
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        significance_threshold=0.05,
        interaction_depth=1,
        categorical_features=None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.significance_threshold = significance_threshold
        self.interaction_depth = interaction_depth
        self.categorical_features = categorical_features

    def _get_categorical_mask(self, X):
        """Identify categorical features."""
        n_features = X.shape[1]
        if self.categorical_features is None:
            # Simple heuristic: if it's a DataFrame, check dtypes
            if isinstance(X, pd.DataFrame):
                return X.dtypes.isin(['object', 'category']).values
            # If it's a numpy array, we can't easily tell, so assume all numerical
            return np.zeros(n_features, dtype=bool)
        
        mask = np.zeros(n_features, dtype=bool)
        mask[self.categorical_features] = True
        return mask

    def fit(self, X, y):
        """
        Build a GUIDE tree classifier from the training set (X, y).
        """
        # 1. Scikit-learn validation
        # We store the original format for categorical detection
        X_orig = X
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        self._categorical_mask = self._get_categorical_mask(X_orig)
        
        # 2. Build the tree
        self.tree_ = self._fit_node(X, y, depth=0)
        
        self.is_fitted_ = True
        return self

    def _fit_node(self, X, y, depth):
        """Recursive function to grow the tree."""
        n_samples = len(y)
        unique_y = np.unique(y)
        
        # 1. Majority class and probabilities
        counts = np.bincount(y, minlength=self.n_classes_)
        if np.sum(counts) > 0:
            probabilities = counts / np.sum(counts)
        else:
            probabilities = np.ones(self.n_classes_) / self.n_classes_
            
        prediction = self.classes_[np.argmax(counts)]
        
        # 2. Check stopping criteria
        if (len(unique_y) == 1 or 
            n_samples < self.min_samples_split or 
            (self.max_depth is not None and depth >= self.max_depth)):
            return GuideNode(depth=depth, is_leaf=True, prediction=prediction, probabilities=probabilities)
            
        # 3. Variable Selection (GUIDE step 1)
        best_idx, p = select_split_variable(X, y, categorical_features=self._categorical_mask)
        
        # 4. Split Point Optimization (GUIDE step 2)
        is_cat = self._categorical_mask[best_idx]
        threshold, gain = find_best_split(X[:, best_idx], y, is_categorical=is_cat)
        
        # 5. If no valid split found, return leaf
        if threshold is None or gain <= 0:
            return GuideNode(depth=depth, is_leaf=True, prediction=prediction, probabilities=probabilities)
            
        # 6. Create node and recurse
        node = GuideNode(depth=depth, split_feature=best_idx, split_threshold=threshold, probabilities=probabilities)
        
        if is_cat:
            left_mask = (X[:, best_idx] == threshold)
        else:
            left_mask = (X[:, best_idx] <= threshold)
            
        node.left = self._fit_node(X[left_mask], y[left_mask], depth + 1)
        node.right = self._fit_node(X[~left_mask], y[~left_mask], depth + 1)
        
        return node

    def predict(self, X):
        """
        Predict class for X.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        return np.array([self._predict_single(x, self.tree_) for x in X])

    def _predict_single(self, x, node):
        """Predict for a single sample by traversing the tree."""
        if node.is_leaf:
            return node.prediction
            
        is_cat = self._categorical_mask[node.split_feature]
        
        if is_cat:
            go_left = (x[node.split_feature] == node.split_threshold)
        else:
            go_left = (x[node.split_feature] <= node.split_threshold)
            
        if go_left:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        return np.array([self._predict_proba_single(x, self.tree_) for x in X])

    def _predict_proba_single(self, x, node):
        """Predict probabilities for a single sample."""
        if node.is_leaf:
            return node.probabilities
            
        is_cat = self._categorical_mask[node.split_feature]
        
        if is_cat:
            go_left = (x[node.split_feature] == node.split_threshold)
        else:
            go_left = (x[node.split_feature] <= node.split_threshold)
            
        if go_left:
            return self._predict_proba_single(x, node.left)
        else:
            return self._predict_proba_single(x, node.right)
