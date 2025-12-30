import numpy as np
from pyguide import GuideTreeClassifier, GuideTreeRegressor

# 1. Test Classifier routing with NaNs
# Create enough data for statistical tests to work
# Pattern: X[0] > 0.5 -> class 1. NaNs also -> class 1.
X_clf = np.array([
    [1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0],
    [0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0],
    [np.nan, 1.0], [np.nan, 0.0]
])
y_clf = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1])

clf = GuideTreeClassifier(max_depth=1, min_samples_split=2)
clf.fit(X_clf, y_clf)
pred_clf = clf.predict([[np.nan, 1.0]])
print(f"Classifier prediction for NaN: {pred_clf}")

# 2. Test Regressor routing with NaNs
# Pattern: X[0] > 0.5 -> y=100. NaNs also -> y=100.
X_reg = np.array([
    [1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0],
    [0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0],
    [np.nan, 1.0], [np.nan, 0.0]
])
y_reg = np.array([100.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0])

reg = GuideTreeRegressor(max_depth=1, min_samples_split=2)
reg.fit(X_reg, y_reg)
pred_reg = reg.predict([[np.nan, 1.0]])
print(f"Regressor prediction for NaN: {pred_reg}")

# Simple validation
assert pred_clf[0] == 1
# Tolerance for float comparison
assert np.abs(pred_reg[0] - 100.0) < 1e-5
print("\nSUCCESS: Missing values are routed correctly based on training data.")