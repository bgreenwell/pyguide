from sklearn.datasets import make_classification, make_regression
from pyguide import GuideTreeClassifier, GuideTreeRegressor
import numpy as np

def verify_classifier():
    print("--- Verifying Classifier max_features ---")
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    
    # 1. Test sqrt
    clf = GuideTreeClassifier(max_features="sqrt", random_state=42)
    clf.fit(X, y)
    print(f"sqrt(20) ~ {int(np.sqrt(20))}, Resolved max_features: {clf.max_features_}")
    assert clf.max_features_ == int(np.sqrt(20))
    
    # 2. Test randomness/reproducibility
    clf1 = GuideTreeClassifier(max_features=5, random_state=1)
    clf1.fit(X, y)
    
    clf2 = GuideTreeClassifier(max_features=5, random_state=1)
    clf2.fit(X, y)
    
    clf3 = GuideTreeClassifier(max_features=5, random_state=2)
    clf3.fit(X, y)
    
    print(f"Seed 1 root feature: {clf1.tree_.feature[0]}")
    print(f"Seed 1 (again) root feature: {clf2.tree_.feature[0]}")
    print(f"Seed 2 root feature: {clf3.tree_.feature[0]}")
    
    assert clf1.tree_.feature[0] == clf2.tree_.feature[0], "Reproducibility failed"
    # Note: Seed 2 might accidentally pick same feature if it is very strong, but likely to differ
    print("Reproducibility verified.")
    print()

def verify_regressor():
    print("--- Verifying Regressor max_features ---")
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    
    # log2
    reg = GuideTreeRegressor(max_features="log2", random_state=0)
    reg.fit(X, y)
    print(f"log2(10) ~ {int(np.log2(10))}, Resolved max_features: {reg.max_features_}")
    assert reg.max_features_ == int(np.log2(10))
    print()

if __name__ == "__main__":
    verify_classifier()
    verify_regressor()
    print("Verification script finished successfully!")
