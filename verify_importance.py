import numpy as np
from pyguide import GuideTreeClassifier
X = np.random.rand(100, 5)
y = (X[:, 0] > 0.5).astype(int)
clf = GuideTreeClassifier(max_depth=2).fit(X, y)
print(f"Importances: {clf.feature_importances_}")
print(f"Sum: {np.sum(clf.feature_importances_)}")
assert np.isclose(np.sum(clf.feature_importances_), 1.0)
# Confirm output:
#      - The first feature should have high importance.
#      - The sum should be exactly 1.0.