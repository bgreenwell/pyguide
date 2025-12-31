import itertools
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .interactions import calc_interaction_p_value
from .node import GuideNode
from .selection import select_split_variable
from .splitting import (
    find_best_split,
    _gini,
)
from .visualization import build_mock_tree


class GuideTreeClassifier(ClassifierMixin, BaseEstimator):
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
        categorical_features=None,
        ccp_alpha=0.0,
        interaction_features=None,
        max_interaction_candidates=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.significance_threshold = significance_threshold
        self.interaction_depth = interaction_depth
        self.categorical_features = categorical_features
        self.ccp_alpha = ccp_alpha
        self.interaction_features = interaction_features
        self.max_interaction_candidates = max_interaction_candidates

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        # If categorical_features is not None, we expect categorical data
        # However, check_estimator passes objects without warning,
        # so we must be careful.
        tags.input_tags.categorical = self.categorical_features is not None
        return tags

    def _get_categorical_mask(self, X, n_features):
        """Identify categorical features."""
        if self.categorical_features is None:
            # Simple heuristic: if it's a DataFrame, check dtypes
            if isinstance(X, pd.DataFrame):
                return X.dtypes.isin(["object", "category"]).values
            # If it's a numpy array or something else, check for object/string types
            if hasattr(X, "dtype") and X.dtype.kind in ["O", "U", "S"]:
                return np.ones(n_features, dtype=bool)
            # Assume all numerical
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

        # If user explicitly provided categorical features or X is obviously categorical
        # we allow non-numeric dtypes.
        if self.categorical_features is not None or isinstance(X, pd.DataFrame):
            dtype = None
        elif hasattr(X, "dtype") and X.dtype.kind in ["U", "S"]:
            dtype = None
        else:
            dtype = "numeric"

        X, y = check_X_y(X, y, dtype=dtype, ensure_all_finite="allow-nan")
        check_classification_targets(y)

        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        self._categorical_mask = self._get_categorical_mask(X_orig, self.n_features_in_)

        # 2. Build the tree
        self._root = self._fit_node(X, y, depth=0)

        # 3. Post-pruning
        if self.ccp_alpha > 0.0:
            self._prune_tree(self._root, len(y))

        # 4. Assign node IDs (pre-order traversal)
        self.n_nodes_ = self._assign_node_ids(self._root, 0)

        self.is_fitted_ = True
        return self

    def _assign_node_ids(self, node, next_id):
        """Recursively assign IDs to nodes."""
        node.node_id = next_id
        next_id += 1
        if not node.is_leaf:
            next_id = self._assign_node_ids(node.left, next_id)
            next_id = self._assign_node_ids(node.right, next_id)
        return next_id

    def apply(self, X):
        """
        Return the index of the leaf that each sample is predicted as.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=None, ensure_all_finite="allow-nan")
        return np.array([self._apply_single(x, self._root) for x in X])

    def _apply_single(self, x, node):
        if node.is_leaf:
            return node.node_id

        is_cat = self._categorical_mask[node.split_feature]
        val = x[node.split_feature]
        is_nan = False
        if is_cat:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                is_nan = True
        else:
            if np.isnan(val):
                is_nan = True

        if is_nan:
            go_left = node.missing_go_left
        else:
            if is_cat:
                go_left = val in node.split_threshold
            else:
                go_left = val <= node.split_threshold

        if go_left:
            return self._apply_single(x, node.left)
        else:
            return self._apply_single(x, node.right)

    def decision_path(self, X):
        """
        Return the decision path in the tree.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=None, ensure_all_finite="allow-nan")

        from scipy.sparse import csr_matrix

        n_samples = X.shape[0]
        indptr = [0]
        indices = []

        for x in X:
            path = []
            self._decision_path_single(x, self._root, path)
            indices.extend(path)
            indptr.append(len(indices))

        data = np.ones(len(indices), dtype=int)
        return csr_matrix((data, indices, indptr), shape=(n_samples, self.n_nodes_))

    def _decision_path_single(self, x, node, path):
        path.append(node.node_id)
        if node.is_leaf:
            return

        is_cat = self._categorical_mask[node.split_feature]
        val = x[node.split_feature]
        is_nan = False
        if is_cat:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                is_nan = True
        else:
            if np.isnan(val):
                is_nan = True

        if is_nan:
            go_left = node.missing_go_left
        else:
            if is_cat:
                go_left = val in node.split_threshold
            else:
                go_left = val <= node.split_threshold

        if go_left:
            self._decision_path_single(x, node.left, path)
        else:
            self._decision_path_single(x, node.right, path)

    def _prune_tree(self, node, n_total):
        """
        Recursively prune the tree using Minimal Cost-Complexity Pruning.
        """
        if node.is_leaf:
            return node.impurity * (node.n_samples / n_total), 1

        # Recursive call
        left_impurity, left_leaves = self._prune_tree(node.left, n_total)
        right_impurity, right_leaves = self._prune_tree(node.right, n_total)

        subtree_impurity = left_impurity + right_impurity
        subtree_leaves = left_leaves + right_leaves

        # Cost of current node as a leaf
        node_impurity_scaled = node.impurity * (node.n_samples / n_total)

        # Pruning condition: R(t) - R(T_t) <= alpha * (|T_t| - 1)
        if node_impurity_scaled - subtree_impurity <= self.ccp_alpha * (
            subtree_leaves - 1
        ):
            # Prune!
            node.is_leaf = True
            node.left = None
            node.right = None
            return node_impurity_scaled, 1
        else:
            return subtree_impurity, subtree_leaves

    def cost_complexity_pruning_path(self, X, y, sample_weight=None):
        """
        Compute the pruning path during Minimal Cost-Complexity Pruning.
        Currently a stub for scikit-learn compatibility.
        """
        # TODO: Implement actual pruning path calculation
        return {"ccp_alphas": np.array([0.0]), "impurities": np.array([0.0])}

    @property
    def tree_(self):
        """Returns a scikit-learn compatible MockTree."""
        check_is_fitted(self)
        return build_mock_tree(
            self._root, n_classes=self.n_classes_, is_classifier=True
        )

    def get_depth(self):
        """
        Return the depth of the decision tree.
        The depth of a tree is the maximum distance between the root
        and any leaf.
        """
        check_is_fitted(self)
        return self._get_depth(self._root)

    def _get_depth(self, node):
        if node.is_leaf:
            return 0
        return 1 + max(self._get_depth(node.left), self._get_depth(node.right))

    def get_n_leaves(self):
        """
        Return the number of leaves of the decision tree.
        """
        check_is_fitted(self)
        return self._get_n_leaves(self._root)

    def _get_n_leaves(self, node):
        if node.is_leaf:
            return 1
        return self._get_n_leaves(node.left) + self._get_n_leaves(node.right)

    @property
    def n_leaves_(self):
        return self.get_n_leaves()

    @property
    def max_depth_(self):
        return self.get_depth()

    @property
    def feature_importances_(self):
        """
        Return the feature importances.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.
        """
        check_is_fitted(self)
        importances = np.zeros(self.n_features_in_)
        self._compute_feature_importances(self._root, importances)
        
        sum_importances = importances.sum()
        if sum_importances > 0:
            importances /= sum_importances
            
        return importances

    def _compute_feature_importances(self, node, importances):
        if node.is_leaf:
            return

        # Weighted impurity reduction
        # Gini reduction = n_node / n_total * (impurity - n_left/n_node * left_imp - n_right/n_node * right_imp)
        # Which simplifies to:
        # (n_node * impurity - n_left * left_imp - n_right * right_imp) / n_total
        
        n_node = node.n_samples
        n_left = node.left.n_samples
        n_right = node.right.n_samples
        
        reduction = (n_node * node.impurity - 
                     n_left * node.left.impurity - 
                     n_right * node.right.impurity)
        
        importances[node.split_feature] += reduction
        
        self._compute_feature_importances(node.left, importances)
        self._compute_feature_importances(node.right, importances)

    def _calculate_lookahead_gain(self, X, y, split_feat, next_feat):
        """
        Calculate total gain of splitting on split_feat, then splitting children on next_feat.
        """
        is_cat = self._categorical_mask[split_feat]
        threshold, missing_go_left, gain1 = find_best_split(
            X[:, split_feat], y, is_categorical=is_cat
        )

        if threshold is None:
            return 0.0

        if is_cat:
            left_mask = np.array([v in threshold for v in X[:, split_feat]])
        else:
            left_mask = X[:, split_feat] <= threshold

        # Handle NaNs in split_feat for children mask
        nan_mask = (
            np.isnan(X[:, split_feat]) if not is_cat else pd.isna(X[:, split_feat])
        )
        if is_cat and X.dtype.kind == "O":
            nan_mask = np.array(
                [
                    (v is None or (isinstance(v, float) and np.isnan(v)))
                    for v in X[:, split_feat]
                ]
            )

        if missing_go_left:
            left_mask = left_mask | nan_mask
        else:
            left_mask = left_mask & ~nan_mask

        y_left = y[left_mask]
        y_right = y[~left_mask]
        X_left = X[left_mask]
        X_right = X[~left_mask]

        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)

        # Gain from second level (next_feat)
        is_cat_next = self._categorical_mask[next_feat]
        _, _, gain2_left = find_best_split(
            X_left[:, next_feat], y_left, is_categorical=is_cat_next
        )
        _, _, gain2_right = find_best_split(
            X_right[:, next_feat], y_right, is_categorical=is_cat_next
        )

        total_gain = gain1 + (n_left / n) * gain2_left + (n_right / n) * gain2_right
        return total_gain

    def _fit_node(self, X, y, depth):
        """Recursive function to grow the tree."""
        n_samples = len(y)
        unique_y = np.unique(y)

        # Calculate majority class and probabilities
        counts = np.bincount(y, minlength=self.n_classes_)
        if np.sum(counts) > 0:
            probabilities = counts / np.sum(counts)
        else:
            probabilities = np.ones(self.n_classes_) / self.n_classes_

        prediction = self.classes_[np.argmax(counts)]
        current_impurity = _gini(y)

        # 2. Check stopping criteria
        if (
            len(unique_y) == 1
            or n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return GuideNode(
                depth=depth,
                is_leaf=True,
                prediction=prediction,
                probabilities=probabilities,
                n_samples=n_samples,
                impurity=current_impurity,
                value_distribution=counts,
            )

        # 3. Variable Selection (GUIDE step 1)
        best_idx, p, all_p_values = select_split_variable(
            X, y, categorical_features=self._categorical_mask
        )

        # Interaction Detection (Fallback)
        interaction_split_override = False
        if p > self.significance_threshold and self.interaction_depth > 0:
            best_int_p = 1.0
            best_int_group = None
            n_features = X.shape[1]

            # Determine candidates for interaction search
            candidates = list(range(n_features))

            # 1. Filter by interaction_features
            if self.interaction_features is not None:
                candidates = [c for c in candidates if c in self.interaction_features]

            # 2. Filter by max_interaction_candidates
            if self.max_interaction_candidates is not None:
                # Sort candidates by their p-value (ascending)
                candidates.sort(key=lambda idx: all_p_values[idx])
                candidates = candidates[: self.max_interaction_candidates]

            # Search interactions (groups of size 2 up to interaction_depth + 1)
            # We limit to candidates to avoid combinatorial explosion
            for size in range(2, self.interaction_depth + 2):
                for group in itertools.combinations(candidates, size):
                    p_int = calc_interaction_p_value(
                        X[:, list(group)],
                        y,
                        categorical_mask=self._categorical_mask[list(group)],
                    )
                    if p_int < best_int_p:
                        best_int_p = p_int
                        best_int_group = group

            if best_int_p < self.significance_threshold:
                # Interaction found! Select the best variable from the group to split on.
                if len(best_int_group) == 2:
                    # For pairs, perform standard look-ahead
                    i, j = best_int_group
                    gain_i_then_j = self._calculate_lookahead_gain(X, y, i, j)
                    gain_j_then_i = self._calculate_lookahead_gain(X, y, j, i)

                    if gain_i_then_j >= gain_j_then_i:
                        best_idx = i
                    else:
                        best_idx = j
                else:
                    # For triplets+, pick the one with the smallest individual p-value
                    best_idx = best_int_group[0]
                    min_p = all_p_values[best_idx]
                    for feat in best_int_group:
                        if all_p_values[feat] < min_p:
                            min_p = all_p_values[feat]
                            best_idx = feat

                interaction_split_override = True

        # Check significance threshold
        if not interaction_split_override and p > self.significance_threshold:
            return GuideNode(
                depth=depth,
                is_leaf=True,
                prediction=prediction,
                probabilities=probabilities,
                n_samples=n_samples,
                impurity=current_impurity,
                value_distribution=counts,
            )

        # 4. Split Point Optimization (GUIDE step 2)
        is_cat = self._categorical_mask[best_idx]

        threshold, missing_go_left, gain = find_best_split(
            X[:, best_idx], y, is_categorical=is_cat
        )

        # 5. If no valid split found, return leaf
        if threshold is None or (gain <= 0 and not interaction_split_override):
            return GuideNode(
                depth=depth,
                is_leaf=True,
                prediction=prediction,
                probabilities=probabilities,
                n_samples=n_samples,
                impurity=current_impurity,
                value_distribution=counts,
            )

        # 6. Create node and recurse
        node = GuideNode(
            depth=depth,
            split_feature=best_idx,
            split_threshold=threshold,
            missing_go_left=missing_go_left,
            probabilities=probabilities,
            n_samples=n_samples,
            impurity=current_impurity,
            value_distribution=counts,
        )

        if is_cat:
            left_mask = np.array([v in threshold for v in X[:, best_idx]])
        else:
            left_mask = X[:, best_idx] <= threshold

        # Handle NaNs
        nan_mask = np.isnan(X[:, best_idx]) if not is_cat else pd.isna(X[:, best_idx])
        if is_cat and X.dtype.kind == "O":
            nan_mask = np.array(
                [
                    (v is None or (isinstance(v, float) and np.isnan(v)))
                    for v in X[:, best_idx]
                ]
            )

        if missing_go_left:
            left_mask = left_mask | nan_mask
        else:
            left_mask = left_mask & ~nan_mask

        node.left = self._fit_node(X[left_mask], y[left_mask], depth + 1)
        node.right = self._fit_node(X[~left_mask], y[~left_mask], depth + 1)

        return node

    def predict(self, X):
        """
        Predict class for X.
        """
        check_is_fitted(self)

        # Use numeric by default unless categorical features were handled in fit
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

        X = check_array(X, dtype=dtype, ensure_all_finite="allow-nan")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.__class__.__name__} is expecting {self.n_features_in_} features as input."
            )

        return np.array([self._predict_single(x, self._root) for x in X])

    def _predict_single(self, x, node):
        """Predict for a single sample by traversing the tree."""
        if node.is_leaf:
            return node.prediction

        is_cat = self._categorical_mask[node.split_feature]
        val = x[node.split_feature]

        # Handle missing values
        is_nan = False
        if is_cat:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                is_nan = True
        else:
            if np.isnan(val):
                is_nan = True

        if is_nan:
            go_left = node.missing_go_left
        else:
            if is_cat:
                go_left = val in node.split_threshold
            else:
                go_left = val <= node.split_threshold

        if go_left:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
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

        X = check_array(X, dtype=dtype, ensure_all_finite="allow-nan")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.__class__.__name__} is expecting {self.n_features_in_} features as input."
            )

        return np.array([self._predict_proba_single(x, self._root) for x in X])

    def _predict_proba_single(self, x, node):
        """Predict probabilities for a single sample."""
        if node.is_leaf:
            return node.probabilities

        is_cat = self._categorical_mask[node.split_feature]
        val = x[node.split_feature]

        # Handle missing values
        is_nan = False
        if is_cat:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                is_nan = True
        else:
            if np.isnan(val):
                is_nan = True

        if is_nan:
            go_left = node.missing_go_left
        else:
            if is_cat:
                go_left = val in node.split_threshold
            else:
                go_left = val <= node.split_threshold

        if go_left:
            return self._predict_proba_single(x, node.left)
        else:
            return self._predict_proba_single(x, node.right)
