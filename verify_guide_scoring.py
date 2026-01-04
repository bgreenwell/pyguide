import numpy as np
from pyguide import GuideTreeClassifier, GuideTreeRegressor
from scipy.stats import chi2

def verify_classifier_guide_scoring():
    print("--- Verifying Classifier GUIDE Importance (Eq 1) ---")
    np.random.seed(42)
    X = np.random.rand(100, 5)
    # y depends strongly on X0 and weakly on X1
    y = (X[:, 0] > 0.5).astype(int)
    
    clf = GuideTreeClassifier(max_depth=2, significance_threshold=1.0, max_features=None)
    clf.fit(X, y)
    
    scores = clf.guide_importances_
    print(f"GUIDE Scores: {scores}")
    
    assert scores.shape == (5,)
    assert np.all(scores >= 0), "Scores must be non-negative"
    
    # X0 should have the highest score
    assert np.argmax(scores) == 0
    print("Classifier GUIDE scoring verification successful!")
    print()

def verify_regressor_guide_scoring():
    print("--- Verifying Regressor GUIDE Importance (Eq 1) ---")
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = X[:, 0] * 10 + X[:, 1] * 2 + np.random.normal(0, 0.1, 100)
    
    reg = GuideTreeRegressor(max_depth=2, significance_threshold=1.0, max_features=None)
    reg.fit(X, y)
    
    scores = reg.guide_importances_
    print(f"GUIDE Scores: {scores}")
    
    assert scores.shape == (5,)
    assert np.argmax(scores) == 0
    print("Regressor GUIDE scoring verification successful!")
    print()

if __name__ == "__main__":
    verify_classifier_guide_scoring()
    verify_regressor_guide_scoring()
    print("All GUIDE importance verifications passed!")
