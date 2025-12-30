import numpy as np
from sklearn.tree import export_text, export_graphviz

class MockTree:
    def __init__(self, children_left, children_right, feature, threshold, value, impurity, n_node_samples):
        self.children_left = np.array(children_left, dtype=np.intp)
        self.children_right = np.array(children_right, dtype=np.intp)
        self.feature = np.array(feature, dtype=np.intp)
        self.threshold = np.array(threshold, dtype=np.float64)
        self.value = np.array(value, dtype=np.float64)
        self.impurity = np.array(impurity, dtype=np.float64)
        self.n_node_samples = np.array(n_node_samples, dtype=np.intp)
        self.node_count = len(children_left)
        self.max_depth = 3
        self.n_classes = np.array([self.value.shape[2]], dtype=np.intp)
        self.n_outputs = 1

class MockEstimator:
    def __init__(self, tree, n_features_in_):
        self.tree_ = tree
        self.n_features_in_ = n_features_in_
        self.classes_ = np.array([0, 1])
        self.n_outputs_ = 1
        self.criterion = "gini" # needed for export_text

children_left = [1, -1, -1]
children_right = [2, -1, -1]
feature = [0, -2, -2]
threshold = [0.5, -2, -2]
value = [[[5, 5]], [[5, 0]], [[0, 5]]]
impurity = [0.5, 0.0, 0.0]
n_node_samples = [10, 5, 5]

mock_tree = MockTree(children_left, children_right, feature, threshold, value, impurity, n_node_samples)
mock_estimator = MockEstimator(mock_tree, 2)

print("Testing export_text...")
try:
    print(export_text(mock_estimator))
except Exception as e:
    print(f"export_text failed: {e}")

print("\nTesting export_graphviz...")
try:
    print(export_graphviz(mock_estimator))
except Exception as e:
    print(f"export_graphviz failed: {e}")
