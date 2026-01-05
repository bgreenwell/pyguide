# Specification: Gradient Boosting with GUIDE Trees

## Overview
Implement Gradient Boosting algorithms (`GuideGradientBoostingClassifier` and `GuideGradientBoostingRegressor`) using `GuideTreeRegressor` as the base learner. This allows the boosting machine to benefit from GUIDE's unbiased variable selection and interaction detection capabilities.

## Architecture

### Base Classes
Since `sklearn.ensemble.GradientBoosting*` does not support custom base estimators, we will implement a custom boosting loop inheriting from `sklearn.base.BaseEstimator` and `EnsembleMixin`.

### GuideGradientBoostingRegressor
- **Loss Function:** Least Squares (LS) initially. Residuals $r_i = y_i - \hat{y}_i$.
- **Algorithm:**
    1. Initialize $F_0(x) = \bar{y}$.
    2. For $m = 1$ to $M$:
        a. Compute pseudo-residuals $r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F(x)=F_{m-1}(x)}$.
        b. Fit a `GuideTreeRegressor` $h_m(x)$ to residuals $r_{im}$.
        c. Update $F_m(x) = F_{m-1}(x) + \nu h_m(x)$ (where $\nu$ is learning rate).
- **Parameters:** `n_estimators`, `learning_rate`, `max_depth`, `subsample` (stochastic gradient boosting).

### GuideGradientBoostingClassifier
- **Loss Function:** Deviance (Log Loss).
- **Binary Classification:**
    - Fit `GuideTreeRegressor` to gradients of log-odds.
    - Transform predictions via sigmoid for probability.
- **Multiclass Classification (Phase 2/Future):**
    - One-vs-Rest or Multinomial Deviance (fitting $K$ trees per iteration).

## Dependencies
- `sklearn.base`
- `src/pyguide/regressor.py` (GuideTreeRegressor)

## Technical Challenges
- **Performance:** Fitting `GuideTreeRegressor` in Python $M$ times can be slow. We must leverage `max_interaction_candidates` and possibly efficient data handling.
- **Line Search:** Standard GBMs perform a line search for the optimal leaf values. For MVP, we might skip this or implement a simple version (e.g., just using the tree's mean prediction scaled by learning rate).

## Testing Strategy
- Unit tests for shape, params, output types.
- Integration tests on `make_regression` and `make_classification`.
- Comparison with `sklearn.ensemble.GradientBoosting*` on simple datasets (should achieve comparable or better test error, though likely slower).
