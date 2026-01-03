from sklearn.datasets import load_iris, make_regression

from pyguide import GuideTreeClassifier, GuideTreeRegressor


def test_ccp_alpha_classifier():
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Baseline: no pruning
    clf_unpruned = GuideTreeClassifier(max_depth=5, ccp_alpha=0.0)
    clf_unpruned.fit(X, y)
    nodes_unpruned = clf_unpruned.tree_.node_count
    
    # Pruned tree
    clf_pruned = GuideTreeClassifier(max_depth=5, ccp_alpha=0.1)
    clf_pruned.fit(X, y)
    nodes_pruned = clf_pruned.tree_.node_count
    
    print(f"Classifier: Unpruned nodes: {nodes_unpruned}, Pruned nodes: {nodes_pruned}")
    assert nodes_pruned < nodes_unpruned

def test_ccp_alpha_regressor():
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    
    # Baseline: no pruning
    reg_unpruned = GuideTreeRegressor(max_depth=5, ccp_alpha=0.0)
    reg_unpruned.fit(X, y)
    nodes_unpruned = reg_unpruned.tree_.node_count
    
    # Pruned tree
    # High alpha should lead to a much smaller tree, likely just the root.
    reg_pruned = GuideTreeRegressor(max_depth=5, ccp_alpha=1000.0)
    reg_pruned.fit(X, y)
    nodes_pruned = reg_pruned.tree_.node_count
    
    print(f"Regressor: Unpruned nodes: {nodes_unpruned}, Pruned nodes: {nodes_pruned}")
    assert nodes_pruned < nodes_unpruned