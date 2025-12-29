import numpy as np
from pyguide import GuideTreeClassifier
from pyguide.stats import calc_curvature_p_value
from pyguide.selection import select_split_variable
from pyguide.splitting import find_best_split

X = np.random.rand(10, 2)
y = np.random.randint(0, 2, 10)
clf = GuideTreeClassifier()
clf.fit(X, y)
print("Success")

# Case 1: Perfectly correlated (N=20)
x1 = np.array([0]*10 + [1]*10)
z1 = np.array([0]*10 + [1]*10)
p1 = calc_curvature_p_value(x1, z1, is_categorical=True)
print(f"P-value (correlated, N=20): {p1:.4f}")

# Case 2: Random
x2 = np.random.rand(100)
z2 = np.random.randint(0, 2, 100)
p2 = calc_curvature_p_value(x2, z2, is_categorical=False)
print(f"P-value (random): {p2:.4f}")

# Case 3: Perfectly correlated (N=20) with categorical mask
np.random.seed(42)
x0 = np.array([0]*10 + [1]*10)
y = x0.copy()
x1 = np.random.rand(20)
X = np.column_stack([x0, x1])

# We tell it that feature 0 is categorical
cat_mask = np.array([True, False])

best_idx, p = select_split_variable(X, y, categorical_features=cat_mask)
print(f"Best feature index (with mask): {best_idx}, P-value: {p:.4f}")

# Numerical Split
x = np.array([1, 2, 8, 9])
y = np.array([0, 0, 1, 1])
t, g = find_best_split(x, y, is_categorical=False)
print(f"Num Split: threshold={t}, gain={g:.4f}")

# Categorical Split
x_cat = np.array([0, 0, 1, 1])
y_cat = np.array([0, 0, 1, 1])
c, g_cat = find_best_split(x_cat, y_cat, is_categorical=True)
print(f"Cat Split: category={c}, gain={g_cat:.4f}")

# 2D data that needs 2 splits (Doubled to avoid N=2 issues)
# Split 1: x0 <= 0.5
# Split 2 (if x0 > 0.5): x1 <= 0.5
X = np.array([
    [0.2, 0.2], [0.2, 0.8], # y=0
    [0.2, 0.2], [0.2, 0.8], # y=0
    [0.8, 0.2], [0.8, 0.2], # y=1
    [0.8, 0.8], [0.8, 0.8]  # y=2
])
y = np.array([0, 0, 0, 0, 1, 1, 2, 2])
clf = GuideTreeClassifier(max_depth=2)
clf.fit(X, y)
print(f"Predictions: {clf.predict(X)}")
# Should be [0, 0, 0, 0, 1, 1, 2, 2]

# Create a node that cannot split but is impure
X = np.array([[1], [1], [1]])
y = np.array([0, 0, 1])
clf = GuideTreeClassifier(min_samples_split=10)
clf.fit(X, y)
proba = clf.predict_proba(X)
print(f"Probabilities: {proba[0]}")
# Expected: [0.666..., 0.333...]
pred = clf.predict(X)
print(f"Prediction: {pred[0]}")
# Expected: 0
