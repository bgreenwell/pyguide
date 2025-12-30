import numpy as np
from pyguide import GuideTreeRegressor
from sklearn.utils.validation import check_is_fitted

def verify():
    X = np.array([[10], [20], [30]])
    y = np.array([1, 2, 3])

    reg = GuideTreeRegressor(max_depth=2)
    reg.fit(X, y)
    
    # Confirm it's fitted
    check_is_fitted(reg)
    
    # Confirm prediction
    y_pred = reg.predict(X)
    print(f"Predictions: {y_pred}")
    print(f"Shape: {y_pred.shape}")
    
    # Assertions for verification
    expected_mean = np.mean(y)
    assert np.allclose(y_pred, expected_mean), f"Expected mean {expected_mean}, got {y_pred}"
    assert y_pred.shape == (3,), f"Expected shape (3,), got {y_pred.shape}"
    print("Verification successful!")

if __name__ == "__main__":
    verify()
