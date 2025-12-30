import numpy as np
from pyguide import GuideTreeRegressor

def verify_recursive_regressor():
    # Create a step-like relationship
    # x < 5 -> y = 10, x >= 5 -> y = 100
    X = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=float).reshape(-1, 1)
    y = np.array([10, 10, 10, 10, 100, 100, 100, 100], dtype=float)

    reg = GuideTreeRegressor(max_depth=2, significance_threshold=1.0)
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    print(f"Predictions: {y_pred}")
    
    assert np.allclose(y_pred, y)
    print("Recursive growth verification successful!")

if __name__ == "__main__":
    verify_recursive_regressor()