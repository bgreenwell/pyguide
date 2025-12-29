import pytest
from sklearn.utils.estimator_checks import check_estimator
from pyguide import GuideTreeClassifier

def test_sklearn_compatibility():
    # check_estimator runs a suite of tests to ensure the estimator
    # adheres to scikit-learn conventions.
    check_estimator(GuideTreeClassifier())
