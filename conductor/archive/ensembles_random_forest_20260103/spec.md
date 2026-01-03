# Spec: Tree Ensembles: Random Forest

## Overview
This track implements Random Forest ensembles for `pyguide`. Instead of writing the ensemble logic from scratch, we will leverage `sklearn.ensemble.BaggingClassifier` and `BaggingRegressor`. To achieve true Random Forest behavior (random feature subsets at each split), we must first implement the `max_features` parameter in our base `GuideTreeClassifier` and `GuideTreeRegressor`.

## Goals
- **Base Estimator Enhancement:** Implement `max_features` in `GuideTreeClassifier` and `GuideTreeRegressor`.
- **Ensemble Implementation:** Create `GuideRandomForestClassifier` and `GuideRandomForestRegressor`.
- **Validation:** Verify that the ensembles work correctly and provide improved performance over single trees.

## Key Components

### 1. `max_features` Support
- **Parameter:** Add `max_features` to `__init__` (int, float, "sqrt", "log2", None).
- **Logic:** In `select_split_variable`, randomly sample a subset of features before computing Chi-square tests.
- **Constraints:** Ensure `interaction_features` (if set) intersects correctly with the random subset.

### 2. Ensemble Classes
- **Structure:** Inherit from `sklearn.base.BaseEstimator` and `ClassifierMixin`/`RegressorMixin`.
- **Composition:** Internally use `sklearn.ensemble.BaggingClassifier` with `GuideTreeClassifier` as `base_estimator` (and similarly for Regressor).
- **API:** Expose standard Random Forest parameters (`n_estimators`, `max_samples`, `bootstrap`, `n_jobs`).

## Success Criteria
- `GuideTreeClassifier(max_features="sqrt")` runs and produces different trees on different runs (even with same data, if seed varies).
- `GuideRandomForestClassifier` achieves higher accuracy on complex datasets (e.g., Digits) than a single GUIDE tree.
- Full scikit-learn compatibility (check_estimator).
