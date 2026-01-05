"""
Gradient Boosting with GUIDE trees.
"""
import numpy as np
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets

from .regressor import GuideTreeRegressor


class GuideGradientBoostingRegressor(RegressorMixin, BaseEstimator):
    """
    Gradient Boosting for regression using GUIDE trees as base learners.

    This implementation uses the GUIDE algorithm for unbiased variable selection
    and interaction detection at each boosting stage.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting stages to perform.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
    max_depth : int, default=3
        Maximum depth of the individual regression estimators.
    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient Boosting.
    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to each Tree estimator at each
        boosting iteration.
    
    # GUIDE-specific parameters
    significance_threshold : float, default=0.05
        The p-value threshold for variable selection in GUIDE trees.
    interaction_depth : int, default=1
        The maximum order of interactions to search for.
    max_interaction_candidates : int, default=None
        Limit on candidates for interaction search.
    """
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        random_state=None,
        significance_threshold=0.05,
        interaction_depth=1,
        max_interaction_candidates=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.significance_threshold = significance_threshold
        self.interaction_depth = interaction_depth
        self.max_interaction_candidates = max_interaction_candidates

    def fit(self, X, y):
        """
        Fit the gradient boosting model.
        """
        X, y = check_X_y(X, y, dtype=None)
        self.n_features_in_ = X.shape[1]
        rng = check_random_state(self.random_state)
        
        # 1. Initialize with constant mean
        self.init_ = np.mean(y)
        
        # Current predictions (initially just the mean)
        y_pred = np.full(y.shape[0], self.init_)
        
        self.estimators_ = []
        
        for i in range(self.n_estimators):
            # 2. Compute negative gradient (residuals for LS)
            residuals = y - y_pred
            
            # 3. Fit GUIDE tree to residuals
            tree = GuideTreeRegressor(
                max_depth=self.max_depth,
                random_state=rng, 
                significance_threshold=self.significance_threshold,
                interaction_depth=self.interaction_depth,
                max_interaction_candidates=self.max_interaction_candidates
            )
            
            # Subsampling
            if self.subsample < 1.0:
                n_samples = X.shape[0]
                n_sub = int(self.subsample * n_samples)
                indices = rng.choice(n_samples, n_sub, replace=False)
                X_train = X[indices]
                r_train = residuals[indices]
            else:
                X_train = X
                r_train = residuals
            
            tree.fit(X_train, r_train)
            
            # 4. Update predictions
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            
            self.estimators_.append(tree)
            
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=None)
        
        y_pred = np.full(X.shape[0], self.init_)
        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)
            
        return y_pred


class GuideGradientBoostingClassifier(ClassifierMixin, BaseEstimator):
    """
    Gradient Boosting for binary classification using GUIDE trees.

    This implementation uses the GUIDE algorithm for unbiased variable selection
    at each boosting stage. Currently supports binary classification (Log Loss).

    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting stages to perform.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
    max_depth : int, default=3
        Maximum depth of the individual regression estimators.
    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
    
    # GUIDE-specific parameters
    significance_threshold : float, default=0.05
        The p-value threshold for variable selection.
    interaction_depth : int, default=1
        The maximum order of interactions to search for.
    max_interaction_candidates : int, default=None
        Limit on candidates for interaction search.
    """
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        random_state=None,
        significance_threshold=0.05,
        interaction_depth=1,
        max_interaction_candidates=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.significance_threshold = significance_threshold
        self.interaction_depth = interaction_depth
        self.max_interaction_candidates = max_interaction_candidates

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=None)
        check_classification_targets(y)
        
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"GuideGradientBoostingClassifier currently only supports binary classification. Found {len(self.classes_)} classes.")
            
        self.n_features_in_ = X.shape[1]
        rng = check_random_state(self.random_state)
        
        # Map y to 0/1
        y_encoded = (y == self.classes_[1]).astype(float)
        
        # 1. Initialize with log-odds of positive class probability
        pos_prop = np.mean(y_encoded)
        # Avoid log(0) or log(1)
        pos_prop = np.clip(pos_prop, 1e-10, 1 - 1e-10)
        self.init_raw_ = logit(pos_prop)
        
        # Current predictions (log-odds)
        raw_pred = np.full(y.shape[0], self.init_raw_)
        
        self.estimators_ = []
        
        for i in range(self.n_estimators):
            # 2. Compute negative gradient (residuals: y - p)
            # p = sigmoid(raw_pred)
            prob_pred = expit(raw_pred)
            residuals = y_encoded - prob_pred
            
            # 3. Fit GUIDE tree to residuals
            tree = GuideTreeRegressor(
                max_depth=self.max_depth,
                random_state=rng, 
                significance_threshold=self.significance_threshold,
                interaction_depth=self.interaction_depth,
                max_interaction_candidates=self.max_interaction_candidates
            )
            
            # Subsampling
            if self.subsample < 1.0:
                n_samples = X.shape[0]
                n_sub = int(self.subsample * n_samples)
                indices = rng.choice(n_samples, n_sub, replace=False)
                X_train = X[indices]
                r_train = residuals[indices]
            else:
                X_train = X
                r_train = residuals
            
            tree.fit(X_train, r_train)
            
            # 4. Update predictions
            update = tree.predict(X)
            raw_pred += self.learning_rate * update
            
            self.estimators_.append(tree)
            
        self.is_fitted_ = True
        return self

    def decision_function(self, X):
        """Compute the decision function of X (log-odds)."""
        check_is_fitted(self)
        X = check_array(X, dtype=None)
        
        raw_pred = np.full(X.shape[0], self.init_raw_)
        for tree in self.estimators_:
            raw_pred += self.learning_rate * tree.predict(X)
            
        return raw_pred

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        raw_pred = self.decision_function(X)
        prob_pos = expit(raw_pred)
        return np.vstack([1 - prob_pos, prob_pos]).T

    def predict(self, X):
        """Predict class labels for X."""
        raw_pred = self.decision_function(X)
        indices = (raw_pred > 0).astype(int)
        return self.classes_[indices]