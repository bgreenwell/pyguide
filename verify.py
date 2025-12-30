import numpy as np
from pyguide import GuideTreeRegressor

def verify_selection():
    # X0 is predictive, X1 is noise
    X = np.array([
        [0, 10],
        [0, 20],
        [1, 10],
        [1, 20]
    ], dtype=float)
    y = np.array([10, 10, 50, 50], dtype=float)

    # Force split by setting high significance threshold
    reg = GuideTreeRegressor(max_depth=1, significance_threshold=1.0)
    reg.fit(X, y)
    
    print(f"Is leaf: {reg.tree_.is_leaf}")
    print(f"Split feature: {reg.tree_.split_feature}")
    
    assert reg.tree_.is_leaf is False
    assert reg.tree_.split_feature == 0, f"Expected feature 0, got {reg.tree_.split_feature}"
    print("Variable selection verification successful!")

if __name__ == "__main__":
    verify_selection()