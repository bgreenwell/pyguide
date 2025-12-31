import numpy as np
from pyguide import GuideTreeClassifier
X = np.array([[0], [1]]).reshape(-1, 1)
y = np.array([0, 1])
clf = GuideTreeClassifier(max_depth=1, significance_threshold=1.0).fit(X, y)
dp = clf.decision_path(X)
print(f"Decision Path Sparse Matrix:\n{dp.toarray()}")
assert dp.shape == (2, 3) # root + 2 leaves
# Confirm output:
#      - The matrix should show 1 for the root node (index 0) for both samples.
#      - Each sample should have exactly two 1s in its row (one for root, one for leaf).