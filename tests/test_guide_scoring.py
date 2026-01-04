import numpy as np
from pyguide import GuideTreeClassifier, GuideTreeRegressor
from scipy.stats import chi2

def test_classifier_guide_scoring_basic():
    X = np.random.rand(50, 5)
    y = (X[:, 0] > 0.5).astype(int)
    
    clf = GuideTreeClassifier(max_depth=2, significance_threshold=1.0, max_features=None)
    clf.fit(X, y)
    
    # Check property existence
    scores = clf.guide_importances_
    assert scores.shape == (5,)
    assert np.all(scores >= 0)
    
    # X[0] should have the highest score
    assert np.argmax(scores) == 0

def test_regressor_guide_scoring_basic():
    X = np.random.rand(50, 5)
    y = X[:, 0] * 5 + np.random.normal(0, 0.1, 50)
    
    reg = GuideTreeRegressor(max_depth=2, significance_threshold=1.0, max_features=None)
    reg.fit(X, y)
    
    scores = reg.guide_importances_
    assert scores.shape == (5,)
    assert np.all(scores >= 0)
    assert np.argmax(scores) == 0

def test_guide_scoring_calculation_logic():
    # Verify manual calculation against property
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 1, 1])
    clf = GuideTreeClassifier(max_depth=1, significance_threshold=1.0, max_features=None)
    clf.fit(X, y)
    
    root = clf._root
    n_t = root.n_samples
    
    # Manually calculate score
    stat_0 = root.curvature_stats[0]
    stat_1 = root.curvature_stats[1]
    
    expected_score_0 = np.sqrt(n_t) * stat_0
    expected_score_1 = np.sqrt(n_t) * stat_1
    
    scores = clf.guide_importances_
    
    assert np.isclose(scores[0], expected_score_0)
    assert np.isclose(scores[1], expected_score_1)