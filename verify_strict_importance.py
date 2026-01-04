import numpy as np
from pyguide import GuideTreeClassifier, GuideTreeRegressor

def verify():
    print("--- Verifying GuideTreeClassifier ---")
    np.random.seed(42)
    X = np.random.rand(500, 3)
    # X0 is the signal, X1 and X2 are noise
    y = (X[:, 0] > 0.5).astype(int)
    
    clf = GuideTreeClassifier()
    clf.fit(X, y)
    
    print("Computing importance (this may take a few seconds)...")
    scores = clf.compute_guide_importance(X, y, n_permutations=50, random_state=42)
    print(f"Scores (X0, X1, X2): {scores}")
    
    assert np.argmax(scores) == 0, "X0 should be the most important feature"
    assert scores[0] > 2.0, "Signal variable should have high importance"
    print("Classifier verification: SUCCESS")

    print("\n--- Verifying GuideTreeRegressor ---")
    # Linear signal on X1
    y_reg = 10 * X[:, 1] + np.random.normal(0, 0.1, 500)
    
    reg = GuideTreeRegressor()
    reg.fit(X, y_reg)
    
    print("Computing importance...")
    scores_reg = reg.compute_guide_importance(X, y_reg, n_permutations=50, random_state=42)
    print(f"Scores (X0, X1, X2): {scores_reg}")
    
    assert np.argmax(scores_reg) == 1, "X1 should be the most important feature"
    assert scores_reg[1] > 2.0, "Signal variable should have high importance"
    print("Regressor verification: SUCCESS")

if __name__ == "__main__":
    verify()
