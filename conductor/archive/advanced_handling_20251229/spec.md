# Spec: Advanced Categorical and Missing Value Handling

## Overview
This track enhances the robustness of the `pyguide` library by implementing strategies for handling missing values (NaNs) and improving how categorical splits are determined. Currently, the library assumes complete data and uses a simple one-vs-rest strategy for categorical splits.

## Goals
- **Missing Value Support:** Allow `fit` and `predict` to handle input data containing `np.nan` or `None`.
- **GUIDE Missing Value Logic:** Implement the strategy where missing values are sent to the child node that minimizes impurity, or treated as a separate category.
- **Advanced Categorical Splits:** For binary classification/regression, order categories by the proportion of class 1 (or mean target value) to find the optimal split in $O(N)$ instead of $O(2^N)$. This is a standard optimization in CART/GUIDE.

## Key Components

### 1. Missing Value Handling
- **Update `find_best_split`:** Modify to calculate impurity reduction when missing values are sent left vs. right.
- **Update `GuideNode`:** Store a `missing_go_left` boolean flag.
- **Update `predict`:** Route samples with missing values based on the flag.

### 2. Optimized Categorical Splitting
- **Classification:** Sort categories by $P(Y=1|X=c)$.
- **Regression:** Sort categories by mean of $Y$.
- **Implementation:** Reduce search space from $2^{K-1}$ to $K-1$ splits for binary targets/regression. For multiclass, one-vs-rest is possibly sufficient or we can use the "mean of class probabilities" heuristic.

## Success Criteria
- Pass `check_estimator` checks related to missing values (if applicable tags are set).
- Correctly classify/predict samples with missing values in synthetic tests.
- Improved performance (or identical performance with faster fit time) on high-cardinality categorical data.
