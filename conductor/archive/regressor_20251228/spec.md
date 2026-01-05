# Spec: Implement GuideTreeRegressor with residual-based selection

## Overview
This track implements the regression version of the GUIDE algorithm. While the tree-growing and interaction logic are shared with the classifier, the variable selection and split optimization steps are specific to the regression task.

## Goals
- **Scikit-learn Compatibility:** Implement `GuideTreeRegressor` inheriting from `RegressorMixin` and `BaseEstimator`.
- **Residual-based Variable Selection:** Implement the transformation of regression residuals into a categorical target for Chi-square testing.
- **SSE Split Optimization:** Find the split threshold that minimizes the Sum of Squared Errors (SSE) for the selected feature.
- **Unified Infrastructure:** Leverage existing `GuideNode` and shared statistical utilities where possible.

## Key Components

### 1. `GuideTreeRegressor` Class
- Parameters: `max_depth`, `min_samples_split`, `min_samples_leaf`, `significance_threshold`, `interaction_depth`.
- Methods: `fit(X, y)`, `predict(X)`.

### 2. Regression Variable Selection
- In each node, calculate residuals: $r = y - \bar{y}$.
- Create temporary class variable $z$:
  - $z = 1$ if $r > 0$
  - $z = 0$ if $r \le 0$
- Use the existing `select_split_variable(X, z)` to pick the best feature.

### 3. Regression Split Optimization
- Implement `_find_best_threshold_sse` for numerical features.
- Implement `_find_best_split_categorical_sse` for categorical features.

## Success Criteria
- Pass `check_estimator` for `GuideTreeRegressor`.
- Accurate predictions on standard regression datasets (e.g., California Housing, Diabetes).
- Test coverage > 80% for new code.
