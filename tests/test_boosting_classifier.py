import numpy as np
import pytest
from sklearn.datasets import make_classification

from pyguide import GuideGradientBoostingClassifier


def test_gbm_classifier_binary():
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
    
    gbm = GuideGradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=42)
    gbm.fit(X, y)
    
    assert len(gbm.estimators_) == 10
    assert gbm.classes_.shape[0] == 2
    
    # Check predictions
    preds = gbm.predict(X)
    assert preds.shape == y.shape
    
    # Check probabilities
    probas = gbm.predict_proba(X)
    assert probas.shape == (100, 2)
    assert np.allclose(np.sum(probas, axis=1), 1.0)
    
    # Accuracy should be decent
    acc = np.mean(preds == y)
    assert acc > 0.8

def test_gbm_classifier_subsample():
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
    
    gbm = GuideGradientBoostingClassifier(n_estimators=10, max_depth=2, subsample=0.5, random_state=42)
    gbm.fit(X, y)
    
    assert len(gbm.estimators_) == 10
    acc = np.mean(gbm.predict(X) == y)
    assert acc > 0.8

def test_gbm_classifier_multiclass_error():
    X, y = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=5, random_state=42)
    
    gbm = GuideGradientBoostingClassifier()
    with pytest.raises(ValueError, match="only supports binary classification"):
        gbm.fit(X, y)
