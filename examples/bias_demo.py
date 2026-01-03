"""
Demonstration of GUIDE's unbiased variable selection.
Compares GUIDE against standard CART (scikit-learn) on high-cardinality noise.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from pyguide import GuideTreeClassifier


def bias_demo():
    print("--- Variable Selection Bias Demo ---")
    np.random.seed(42)
    n_samples = 200
    
    # X0: Binary informative feature (weak signal)
    X0 = np.random.randint(0, 2, n_samples)
    y = X0.copy()
    # Add some noise to the signal
    flip = np.random.rand(n_samples) < 0.1
    y[flip] = 1 - y[flip]
    
    # X1: High-cardinality noise feature (many unique values, no signal)
    X1 = np.arange(n_samples)
    
    X = np.column_stack([X0, X1])
    
    print("Feature 0: Binary, informative (Accuracy ~0.9)")
    print("Feature 1: High-cardinality (unique ID), pure NOISE")
    print("\nA standard CART tree will almost always split on Feature 1 first")
    print("because it can perfectly partition the data using many unique values.")

    # 1. Scikit-learn DecisionTreeClassifier (CART)
    dt = DecisionTreeClassifier(max_depth=1)
    dt.fit(X, y)
    print(f"\nScikit-learn (CART) split on feature: {dt.tree_.feature[0]}")
    
    # 2. GUIDE Tree Classifier
    clf = GuideTreeClassifier(max_depth=1)
    clf.fit(X, y)
    # Map back from internal index if needed, but here they are the same
    split_feat = clf._root.split_feature
    print(f"pyguide (GUIDE) split on feature: {split_feat}")

    if split_feat == 0:
        print("\nSUCCESS: GUIDE correctly identified the informative feature and ignored the noise!")

if __name__ == "__main__":
    bias_demo()
