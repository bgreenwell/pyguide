# Plan: Scikit-Learn Parity and Variable Importance

## Phase 1: Structural Attributes [checkpoint: ab2f006]
- [x] Task: Implement `n_leaves_` and `get_depth()` (exposed as `max_depth_`) [4b95a82]
- [x] Task: Create tests for structural attributes (verify counts on known tree structures) [4b95a82]
- [x] Task: Conductor - User Manual Verification 'Phase 1: Structural Attributes' (Protocol in workflow.md) [ab2f006]

## Phase 2: Variable Importance [checkpoint: eeb2764]
- [x] Task: Implement `feature_importances_` based on weighted impurity reduction [8cab993]
- [x] Task: Create tests for feature importance (verify that predictive features have higher scores) [8cab993]
- [x] Task: Verify compatibility with `sklearn.feature_selection.SelectFromModel` [8cab993]
- [x] Task: Conductor - User Manual Verification 'Phase 2: Variable Importance' (Protocol in workflow.md) [eeb2764]

## Phase 3: Diagnostic Methods [checkpoint: 9d1369f]
- [x] Task: Implement `apply(X)` to return leaf indices [bf8e137]
- [x] Task: Implement `decision_path(X)` using a CSR matrix [3fde1a5]
- [x] Task: Create tests for `apply` and `decision_path` [3fde1a5]
- [x] Task: Conductor - User Manual Verification 'Phase 3: Diagnostic Methods' (Protocol in workflow.md) [9d1369f]
