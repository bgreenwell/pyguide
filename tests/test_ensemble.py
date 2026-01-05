import numpy as np
import pytest
from sklearn.datasets import load_digits, load_iris, make_regression
from sklearn.utils.estimator_checks import check_estimator

from pyguide import GuideRandomForestClassifier, GuideRandomForestRegressor


def test_rf_classifier_basic():
    X, y = load_iris(return_X_y=True)
    clf = GuideRandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X, y)
    
    assert hasattr(clf, "classes_")
    assert len(clf.classes_) == 3
    
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    # Accuracy should be high on Iris
    acc = np.mean(y_pred == y)
    assert acc > 0.9

def test_rf_regressor_basic():
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    reg = GuideRandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    reg.fit(X, y)
    
    y_pred = reg.predict(X)
    assert y_pred.shape == y.shape
    
    # R2 should be reasonable
    r2 = reg.score(X, y)
    assert r2 > 0.5

@pytest.mark.parametrize("estimator", [
    GuideRandomForestClassifier(n_estimators=2),
    GuideRandomForestRegressor(n_estimators=2)
])
def test_rf_compatibility(estimator):
    # n_estimators=2 to speed up checks
    check_estimator(estimator)

def test_rf_classifier_improvement():
    # Random Forest should generally outperform a single deep tree on complex data
    X, y = load_digits(return_X_y=True)
    
    from pyguide import GuideTreeClassifier
    
    # Single tree
    clf_tree = GuideTreeClassifier(max_depth=5, random_state=42)
    clf_tree.fit(X, y)
    score_tree = clf_tree.score(X, y)
    
    # Forest
    clf_rf = GuideRandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
    clf_rf.fit(X, y)
    score_rf = clf_rf.score(X, y)
    
    print(f"Digits Score - Tree: {score_tree:.4f}, RF: {score_rf:.4f}")
    # RF should be better or at least comparable on training set
    assert score_rf >= score_tree
