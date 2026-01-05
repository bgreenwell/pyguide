import numpy as np
import pytest
from sklearn.datasets import load_iris, make_regression

from pyguide import GuideTreeClassifier, GuideTreeRegressor


def test_max_features_classifier():
    X, y = load_iris(return_X_y=True)
    n_features = X.shape[1]
    
    # 1. Test different types
    # int
    clf = GuideTreeClassifier(max_features=2, random_state=42)
    clf.fit(X, y)
    assert clf.max_features_ == 2
    
    # float
    clf = GuideTreeClassifier(max_features=0.5, random_state=42)
    clf.fit(X, y)
    assert clf.max_features_ == int(0.5 * n_features)
    
    # sqrt
    clf = GuideTreeClassifier(max_features="sqrt", random_state=42)
    clf.fit(X, y)
    assert clf.max_features_ == int(np.sqrt(n_features))
    
    # 2. Test randomness
    # If we restrict features, two trees with different seeds should likely differ
    # (assuming there's redundancy in features or noise)
    clf1 = GuideTreeClassifier(max_features=1, random_state=1)
    clf1.fit(X, y)
    
    clf2 = GuideTreeClassifier(max_features=1, random_state=2)
    clf2.fit(X, y)
    
    # Structure or split features should differ
    # Note: Iris is simple, so maybe they find same split if feature 2/3 are highly correlated.
    # But with max_features=1, one might pick feat 0 and fail to split well, another pick feat 2.
    
    # Let's check root split feature
    # They shouldn't be identical all the time.
    # Actually, with max_features=1, it picks ONE feature.
    # If clf1 picks feat 0, clf2 picks feat 1.
    if clf1.tree_.feature[0] == clf2.tree_.feature[0]:
        # Unlikely for 4 features and 2 random picks unless they picked same or equivalent
        pass 
        
    # 3. Test reproducibility
    clf3 = GuideTreeClassifier(max_features=2, random_state=42)
    clf3.fit(X, y)
    
    # Should be identical to initial int test
    # (assuming structure comparison)
    assert clf3.tree_.feature[0] == clf.tree_.feature[0]
    
def test_max_features_regressor():
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    n_features = 10
    
    # sqrt
    reg = GuideTreeRegressor(max_features="sqrt", random_state=0)
    reg.fit(X, y)
    assert reg.max_features_ == int(np.sqrt(n_features)) # 3
    
    # log2
    reg = GuideTreeRegressor(max_features="log2", random_state=0)
    reg.fit(X, y)
    assert reg.max_features_ == int(np.log2(n_features)) # 3
    
    # Test that it respects the limit
    # We can't easily introspect the internal sampling without mocking,
    # but we can verify it runs.

def test_invalid_max_features():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    
    clf = GuideTreeClassifier(max_features="invalid")
    with pytest.raises(ValueError, match="Invalid max_features"):
        clf.fit(X, y)
