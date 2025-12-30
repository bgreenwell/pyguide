from sklearn.utils.estimator_checks import check_estimator

from pyguide import GuideTreeClassifier, GuideTreeRegressor



def test_classifier_compatibility():

    # check_estimator runs a suite of tests to ensure the estimator

    # adheres to scikit-learn conventions.

    check_estimator(GuideTreeClassifier())



def test_regressor_compatibility():

    check_estimator(GuideTreeRegressor())
