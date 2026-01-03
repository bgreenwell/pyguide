"""
Ensemble models based on GUIDE trees.
"""
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from .classifier import GuideTreeClassifier
from .regressor import GuideTreeRegressor

class GuideRandomForestClassifier(ClassifierMixin, BaseEstimator):
    """
    Random Forest Classifier using GUIDE trees as base estimators.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
        
    max_depth : int, default=None
        The maximum depth of the trees.
        
    max_features : int, float, str or None, default="sqrt"
        The number of features to consider when looking for the best split.
        
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
        
    n_jobs : int, default=None
        The number of jobs to run in parallel for both `fit` and `predict`.
        
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    significance_threshold : float, default=0.05
        The p-value threshold for variable selection.

    interaction_depth : int, default=1
        The maximum order of interactions to search for.
    """
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=None,
        random_state=None,
        significance_threshold=0.05,
        interaction_depth=1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.significance_threshold = significance_threshold
        self.interaction_depth = interaction_depth

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.base_estimator_ = GuideTreeClassifier(
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=self.random_state,
            significance_threshold=self.significance_threshold,
            interaction_depth=self.interaction_depth,
        )
        self.bagging_ = BaggingClassifier(
            estimator=self.base_estimator_,
            n_estimators=self.n_estimators,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        self.bagging_.fit(X, y)
        self.classes_ = self.bagging_.classes_
        self.n_classes_ = len(self.classes_)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.bagging_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.bagging_.predict_proba(X)

class GuideRandomForestRegressor(RegressorMixin, BaseEstimator):
    """
    Random Forest Regressor using GUIDE trees as base estimators.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
        
    max_depth : int, default=None
        The maximum depth of the trees.
        
    max_features : int, float, str or None, default=1.0
        The number of features to consider when looking for the best split.
        
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
        
    n_jobs : int, default=None
        The number of jobs to run in parallel for both `fit` and `predict`.
        
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    significance_threshold : float, default=0.05
        The p-value threshold for variable selection.

    interaction_depth : int, default=1
        The maximum order of interactions to search for.
    """
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        max_features=1.0,
        bootstrap=True,
        n_jobs=None,
        random_state=None,
        significance_threshold=0.05,
        interaction_depth=1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.significance_threshold = significance_threshold
        self.interaction_depth = interaction_depth

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.base_estimator_ = GuideTreeRegressor(
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=self.random_state,
            significance_threshold=self.significance_threshold,
            interaction_depth=self.interaction_depth,
        )
        self.bagging_ = BaggingRegressor(
            estimator=self.base_estimator_,
            n_estimators=self.n_estimators,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        self.bagging_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.bagging_.predict(X)