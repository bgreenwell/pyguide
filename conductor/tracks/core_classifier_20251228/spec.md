# Spec: Build the core GuideTreeClassifier infrastructure

## Overview
This track focuses on implementing the fundamental architecture and core logic of the `GuideTreeClassifier`. The goal is to create a robust, scikit-learn compatible classifier that implements the unique GUIDE splitting mechanism: unbiased variable selection followed by split point optimization.

## Goals
- **Scikit-learn Compatibility:** Implement `fit`, `predict`, and `predict_proba` methods. Pass `check_estimator` validation where possible for the core functionality.
- **Unbiased Variable Selection:** Implement the Chi-square testing framework for selecting the best splitting variable.
- **Efficient Split Optimization:** Implement optimized search for split thresholds once a variable is selected.
- **Recursive Tree Construction:** Build the tree structure using a recursive growing algorithm.
- **Data Compatibility:** Support both NumPy arrays and Pandas DataFrames as input.

## Key Components

### 1. `GuideTreeClassifier` Class
- Inherits from `BaseEstimator` and `ClassifierMixin`.
- Parameters: `max_depth`, `min_samples_split`, `min_samples_leaf`, `significance_threshold`, `interaction_depth`.

### 2. Variable Selection Engine
- `calc_curvature_p_value`: Computes p-values for main effects using Chi-square tests on binned continuous data or categorical counts.
- `select_split_variable`: Iterates through features and selects the one with the most significant p-value.

### 3. Split Optimizer
- For the selected variable, finds the threshold $t$ that minimizes Gini impurity.

### 4. Tree Structure
- `GuideNode`: Stores split information (feature, threshold) or prediction values for leaves.
- Recursive `_fit_node` function to grow the tree.

## Success Criteria
- Successful training on standard datasets (e.g., Iris, Wine).
- Verification that variable selection is unbiased compared to standard CART (verified via specific synthetic test cases).
- Test coverage > 80%.
- No linting or type-checking errors.
