import pytest
import numpy as np
import pandas as pd
from pyguide import GuideTreeClassifier
from sklearn.utils.estimator_checks import check_estimator

def test_classifier_init():
    clf = GuideTreeClassifier(max_depth=3, min_samples_split=5)
    assert clf.max_depth == 3
    assert clf.min_samples_split == 5

def test_classifier_fit_predict_basic():
    X = np.array([[1, 2], [3, 4], [1, 3], [3, 5]])
    y = np.array([0, 1, 0, 1])
    
    clf = GuideTreeClassifier()
    clf.fit(X, y)
    
    # Check if it can predict
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    
    # Check if it can predict_proba
    y_proba = clf.predict_proba(X)
    assert y_proba.shape == (4, 2)

def test_classifier_pandas_input():
    X = pd.DataFrame({'a': [1, 3, 1, 3], 'b': [2, 4, 3, 5]})
    y = pd.Series([0, 1, 0, 1])
    
    clf = GuideTreeClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert isinstance(y_pred, np.ndarray)
