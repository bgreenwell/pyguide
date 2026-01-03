from sklearn.datasets import load_iris, make_regression
from pyguide import GuideTreeClassifier, GuideTreeRegressor
import numpy as np

def verify_classifier():
    print("--- Verifying Classifier Pruning Path ---")
    X, y = load_iris(return_X_y=True)
    clf = GuideTreeClassifier(max_depth=None, min_samples_split=2)
    clf.fit(X, y)
    path = clf.cost_complexity_pruning_path(X, y)
    print(f"Alphas: {path['ccp_alphas']}")
    print(f"Impurities: {path['impurities']}")
    print(f"Valid sequence: {np.all(np.diff(path['ccp_alphas']) >= 0)}")
    print()

def verify_regressor():
    print("--- Verifying Regressor Pruning Path ---")
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    reg = GuideTreeRegressor(max_depth=None, min_samples_split=2)
    reg.fit(X, y)
    path = reg.cost_complexity_pruning_path(X, y)
    print(f"Alphas: {path['ccp_alphas']}")
    print(f"Impurities: {path['impurities']}")
    print(f"Valid sequence: {np.all(np.diff(path['ccp_alphas']) >= 0)}")
    print()

if __name__ == "__main__":
    verify_classifier()
    verify_regressor()
