import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .interactions import calc_interaction_p_value
from .node import GuideNode
from .selection import select_split_variable
from .splitting import _sse, find_best_split
from .visualization import build_mock_tree


class GuideTreeRegressor(RegressorMixin, BaseEstimator):
    """
    GUIDE (Generalized, Unbiased, Interaction Detection and Estimation) Tree Regressor.

    GUIDE is a decision tree algorithm that separates variable selection from
    split point optimization. This approach prevents the variable selection
    bias common in CART-like algorithms (which favor variables with many
    unique values) and provides built-in interaction detection.

    Parameters
    ----------
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least min_samples_leaf training samples in each of the left and
        right branches.

    significance_threshold : float, default=0.05
        The p-value threshold for variable selection and interaction detection.
        If no variable is individually significant at this level, the algorithm
        searches for interactions. If no interaction is significant either,
        splitting stops.

    interaction_depth : int, default=1
        The maximum order of interactions to search for.
        - 0: No interaction detection.
        - 1: Pairwise interactions.
        - 2: Triplets, etc.

    categorical_features : list of int, default=None
        Indices of features to be treated as categorical. If None, the
        algorithm attempts to identify categorical features automatically
        based on input types (e.g., pandas object/category columns).

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ccp_alpha will be chosen.

    interaction_features : list of int, default=None
        Subset of feature indices to consider for interaction search.
        If None, all features are considered (subject to candidate filtering).

    max_interaction_candidates : int, default=None
        If set, only the top K features (ranked by individual p-values) are
        considered as candidates for interaction tests. This significantly
        speeds up training on high-dimensional datasets.

    max_features : int, float, str or None, default=None
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each split.
        - If "sqrt", then `max_features=sqrt(n_features_in_)`.
        - If "log2", then `max_features=log2(n_features_in_)`.
        - If None, then `max_features=n_features_in_`.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. To obtain a deterministic behaviour during fitting,
        ``random_state`` has to be fixed to an integer.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

    n_nodes_ : int
        Total number of nodes in the fitted tree.

    n_leaves_ : int
        Number of leaf nodes in the fitted tree.

    max_depth_ : int
        The actual maximum depth of the fitted tree.

    feature_importances_ : ndarray of shape (n_features_in_,)
        The feature importances based on weighted impurity reduction (SSE).

    Notes
    -----
    The algorithm follows a two-step process at each node:
    1. Variable Selection: Calculate residuals from the current node mean and
       use Chi-square tests on residual signs to identify the best splitting
       variable.
    2. Split Point Optimization: Given the selected variable, find the
       threshold that minimizes the Sum of Squared Errors (SSE).

    References
    ----------
    Loh, W.-Y. (2002). Regression trees with unbiased variable selection and
    interaction detection. Statistica Sinica, 361-386.
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
        max_features=None,
        random_state=None,
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
        self.max_features = max_features
        self.random_state = random_state

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

        X, y = check_X_y(X, y, dtype=dtype, ensure_all_finite="allow-nan")

        self.n_features_in_ = X.shape[1]
        self._categorical_mask = self._get_categorical_mask(X_orig, self.n_features_in_)

        self.rng_ = check_random_state(self.random_state)
        self.max_features_ = self._resolve_max_features(self.n_features_in_)

        # Build the tree
        self._root = self._fit_node(X, y, depth=0)

        # Post-pruning
        if self.ccp_alpha > 0.0:
            self._prune_tree(self._root, len(y))

        # Assign node IDs
        self.n_nodes_ = self._assign_node_ids(self._root, 0)

        self.is_fitted_ = True
        return self

    def _resolve_max_features(self, n_features):
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, (int, np.integer)):
            return min(n_features, int(self.max_features))
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError(f"Invalid max_features: {self.max_features}")

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
        # For regressor, impurity is SSE. R(t) = SSE(t) / N
        if node.is_leaf:
            return (node.impurity / n_total), 1

        # Recursive call
        left_impurity, left_leaves = self._prune_tree(node.left, n_total)
        right_impurity, right_leaves = self._prune_tree(node.right, n_total)

        subtree_impurity = left_impurity + right_impurity
        subtree_leaves = left_leaves + right_leaves

        # Cost of current node as a leaf
        node_impurity_scaled = node.impurity / n_total

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
        """
        # 1. Fit a full tree
        est = self.__class__(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            significance_threshold=self.significance_threshold,
            interaction_depth=self.interaction_depth,
            categorical_features=self.categorical_features,
            ccp_alpha=0.0,  # Full tree
            interaction_features=self.interaction_features,
            max_interaction_candidates=self.max_interaction_candidates,
        )
        est.fit(X, y)

        n_total = len(y)

        # 2. Compute path
        ccp_alphas = [0.0]
        impurities = [self._calculate_total_impurity(est._root, n_total)]

        while not est._root.is_leaf:
            candidates = []
            self._collect_pruning_candidates(est._root, n_total, candidates)

            if not candidates:
                break

            min_alpha = min(c[1] for c in candidates)
            
            nodes_to_prune = [c[0] for c in candidates if abs(c[1] - min_alpha) < 1e-9]
            
            for node in nodes_to_prune:
                node.is_leaf = True
                node.left = None
                node.right = None

            ccp_alphas.append(min_alpha)
            impurities.append(self._calculate_total_impurity(est._root, n_total))

        return {"ccp_alphas": np.array(ccp_alphas), "impurities": np.array(impurities)}

    def _calculate_total_impurity(self, node, n_total):
        if node.is_leaf:
            return node.impurity / n_total
        return self._calculate_total_impurity(node.left, n_total) + self._calculate_total_impurity(node.right, n_total)

    def _collect_pruning_candidates(self, node, n_total, candidates):
        """
        Recursive helper to find alpha_eff for all internal nodes.
        Returns (R_subtree, n_leaves_subtree) for the node.
        """
        if node.is_leaf:
            R_leaf = node.impurity / n_total
            return R_leaf, 1

        R_left, leaves_left = self._collect_pruning_candidates(node.left, n_total, candidates)
        R_right, leaves_right = self._collect_pruning_candidates(node.right, n_total, candidates)

        R_Tt = R_left + R_right
        leaves_Tt = leaves_left + leaves_right

        R_t = node.impurity / n_total

        if leaves_Tt > 1:
            alpha_eff = (R_t - R_Tt) / (leaves_Tt - 1)
            alpha_eff = max(0.0, alpha_eff)
            candidates.append((node, alpha_eff))

        return R_Tt, leaves_Tt

    @property
    def tree_(self):
        """Returns a scikit-learn compatible MockTree."""
        check_is_fitted(self)
        return build_mock_tree(self._root, is_classifier=False)

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

        # Impurity reduction for regression (SSE)
        # Reduction = SSE(node) - (SSE(left) + SSE(right))
        # This is already what find_best_split calculates as 'gain'

        reduction = node.impurity - node.left.impurity - node.right.impurity

        importances[node.split_feature] += max(0, reduction)  # Ensure non-negative

        self._compute_feature_importances(node.left, importances)
        self._compute_feature_importances(node.right, importances)

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
            left_mask = np.array([v in threshold for v in X[:, split_feat]])
        else:
            left_mask = X[:, split_feat] <= threshold

        # Handle NaNs
        nan_mask = np.isnan(X[:, split_feat]) if not is_cat else pd.isna(X[:, split_feat])
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
        prediction = np.mean(y) if n_samples > 0 else 0.0
        current_impurity = _sse(y) if n_samples > 0 else 0.0

        # 1. Check stopping criteria
        if (
            (n_samples > 0 and np.all(np.abs(y - prediction) < 1e-9))
            or n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return GuideNode(
                depth=depth,
                is_leaf=True,
                prediction=prediction,
                n_samples=n_samples,
                impurity=current_impurity,
                value_distribution=np.array([[prediction]]),
                split_type=None,
                interaction_group=None,
            )

        # 2. Variable Selection (GUIDE step 1)
        z = (y > prediction).astype(int)

        if self.max_features_ < self.n_features_in_:
            feature_indices = self.rng_.choice(
                self.n_features_in_, self.max_features_, replace=False
            )
        else:
            feature_indices = None

        best_idx, p, all_p_values = select_split_variable(
            X, z, categorical_features=self._categorical_mask, feature_indices=feature_indices
        )

        # Interaction Detection (Fallback)
        interaction_split_override = False
        if p > self.significance_threshold and self.interaction_depth > 0:
            best_int_p = 1.0
            best_int_group = None
            n_features = X.shape[1]

            # Determine candidates for interaction search
            if feature_indices is not None:
                candidates = list(feature_indices)
            else:
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
            for size in range(2, self.interaction_depth + 2):
                for group in itertools.combinations(candidates, size):
                    p_int = calc_interaction_p_value(
                        X[:, list(group)],
                        z,
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
                n_samples=n_samples,
                impurity=current_impurity,
                value_distribution=np.array([[prediction]]),
                split_type=None,
                interaction_group=None,
            )

        # 3. Split Point Optimization (GUIDE step 2)
        is_cat = self._categorical_mask[best_idx]
        threshold, missing_go_left, gain = find_best_split(
            X[:, best_idx], y, is_categorical=is_cat, criterion="mse"
        )

        # 4. If no valid split found, return leaf
        if threshold is None or (gain <= 0 and not interaction_split_override):
            return GuideNode(
                depth=depth,
                is_leaf=True,
                prediction=prediction,
                n_samples=n_samples,
                impurity=current_impurity,
                value_distribution=np.array([[prediction]]),
                split_type=None,
                interaction_group=None,
            )

        # 5. Create node and recurse
        node = GuideNode(
            depth=depth,
            split_feature=best_idx,
            split_threshold=threshold,
            missing_go_left=missing_go_left,
            n_samples=n_samples,
            impurity=current_impurity,
            value_distribution=np.array([[prediction]]),
            split_type="interaction" if interaction_split_override else "main",
            interaction_group=best_int_group if interaction_split_override else None,
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
        Predict regression target for X.
        """
        check_is_fitted(self)

        dtype = (
            None
            if (
                self.categorical_features is not None
                or (hasattr(self, "_categorical_mask") and np.any(self._categorical_mask))
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
