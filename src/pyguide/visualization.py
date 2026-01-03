import numpy as np


class MockTree:
    """
    A class that mocks the interface of sklearn.tree._tree.Tree.
    Used for compatibility with sklearn's visualization tools.
    """
    def __init__(self, children_left, children_right, feature, threshold, value, impurity, n_node_samples):
        self.children_left = np.array(children_left, dtype=np.intp)
        self.children_right = np.array(children_right, dtype=np.intp)
        self.feature = np.array(feature, dtype=np.intp)
        self.threshold = np.array(threshold, dtype=np.float64)
        self.value = np.array(value, dtype=np.float64)
        self.impurity = np.array(impurity, dtype=np.float64)
        self.n_node_samples = np.array(n_node_samples, dtype=np.intp)
        self.weighted_n_node_samples = np.array(n_node_samples, dtype=np.float64)
        self.node_count = len(children_left)
        
        # Calculate max_depth
        self.max_depth = self._get_max_depth(0)
        
        # value shape is (n_nodes, n_outputs, n_classes)
        self.n_classes = np.array([self.value.shape[2]], dtype=np.intp)
        self.n_outputs = 1

    def _get_max_depth(self, node_id):
        if self.children_left[node_id] == -1:
            return 0
        return 1 + max(self._get_max_depth(self.children_left[node_id]),
                       self._get_max_depth(self.children_right[node_id]))

def build_mock_tree(root_node, n_classes=1, is_classifier=True):
    """
    Recursively builds arrays for MockTree from a GuideNode structure.
    """
    children_left = []
    children_right = []
    feature = []
    threshold = []
    value = []
    impurity = []
    n_node_samples = []

    def traverse(node):
        node_id = len(children_left)
        # Initialize placeholders
        children_left.append(-1)
        children_right.append(-1)
        feature.append(-2)
        threshold.append(-2.0)
        impurity.append(node.impurity)
        n_node_samples.append(node.n_samples)
        
        # value shape: (n_outputs, n_classes)
        if is_classifier:
            val = np.zeros((1, n_classes))
            if node.value_distribution is not None:
                val[0, :] = node.value_distribution
            else:
                # Fallback
                val[0, int(node.prediction)] = node.n_samples
            value.append(val)
        else:
            # For regressor, value is (1, 1) containing the mean
            value.append(np.array([[node.prediction]]))

        if not node.is_leaf:
            feature[node_id] = node.split_feature
            # sklearn only supports numeric thresholds for plot_tree directly.
            if isinstance(node.split_threshold, (set, frozenset)):
                threshold[node_id] = 0.5 # Dummy
            else:
                threshold[node_id] = node.split_threshold
            
            # Left child
            children_left[node_id] = len(children_left)
            traverse(node.left)
            
            # Right child
            children_right[node_id] = len(children_left)
            traverse(node.right)

    traverse(root_node)

    return MockTree(
        children_left, children_right, feature, threshold, value, impurity, n_node_samples
    )


def plot_tree(decision_tree, **kwargs):
    """
    Plot a GUIDE decision tree.
    This is a wrapper around sklearn.tree.plot_tree.
    """
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.tree import plot_tree as sklearn_plot_tree

    # To bypass sklearn's strict type checking, we use a proxy that
    # inherits from the appropriate base class.
    if hasattr(decision_tree, "classes_"):
        base_class = DecisionTreeClassifier
    else:
        base_class = DecisionTreeRegressor

    class SklearnWrapper(base_class):
        def __init__(self, fitted_tree):
            self.tree_ = fitted_tree.tree_
            self.n_features_in_ = fitted_tree.n_features_in_
            if hasattr(fitted_tree, "classes_"):
                self.classes_ = fitted_tree.classes_
                self.n_classes_ = fitted_tree.n_classes_
            self.n_outputs_ = 1
            self.criterion = "gini" if hasattr(fitted_tree, "classes_") else "mse"
            self.splitter = "best"
            self.max_features_ = fitted_tree.n_features_in_

        def __sklearn_tags__(self):
            # This helps bypass some validation checks in plot_tree
            tags = super().__sklearn_tags__()
            tags.input_tags.allow_nan = True
            return tags

    wrapper = SklearnWrapper(decision_tree)
    return sklearn_plot_tree(wrapper, **kwargs)
