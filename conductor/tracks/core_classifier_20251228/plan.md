# Plan: Build the core GuideTreeClassifier infrastructure

## Phase 1: Project Scaffolding & CI/CD [checkpoint: 2b14280]
- [x] Task: Initialize Python project with `uv` and `pyproject.toml` (Poetry style) [dbdedd1]
- [x] Task: Configure `ruff` and `pytest` with coverage requirements [c9768e7]
- [x] Task: Create basic directory structure (`src/pyguide`, `tests/`) [c8d9c43]
- [x] Task: Conductor - User Manual Verification 'Phase 1: Project Scaffolding & CI/CD' (Protocol in workflow.md) [2b14280]

## Phase 2: Scikit-learn Compliance & Base Structure [checkpoint: 526a9ad]
- [x] Task: Write tests for `GuideTreeClassifier` basic interface (init, fit, predict) [09b9618]
- [x] Task: Implement `GuideTreeClassifier` shell with scikit-learn validation (`check_X_y`, `check_array`) [2a7ea6c]
- [x] Task: Create `GuideNode` class for tree structure [0ca1ffa]
- [x] Task: Conductor - User Manual Verification 'Phase 2: Scikit-learn Compliance & Base Structure' (Protocol in workflow.md) [526a9ad]

## Phase 3: Statistical Testing Utility [checkpoint: 2c78f09]
- [x] Task: Write tests for Chi-square p-value calculation (main effects) [1ff3067]
- [x] Task: Implement `calc_curvature_p_value` helper using `scipy.stats.chi2_contingency` [a2b6219]
- [x] Task: Implement data binning for continuous variables (GUIDE style) [a2b6219]
- [x] Task: Conductor - User Manual Verification 'Phase 3: Statistical Testing Utility' (Protocol in workflow.md) [2c78f09]

## Phase 4: Variable Selection Logic [checkpoint: 7a9e6e0]
- [x] Task: Write tests for `select_split_variable` [447234d]
- [x] Task: Implement variable selection loop across all features [3ad63a1]
- [x] Task: Implement handling of categorical vs numerical features in selection [3ad63a1]
- [x] Task: Conductor - User Manual Verification 'Phase 4: Variable Selection Logic' (Protocol in workflow.md) [7a9e6e0]

## Phase 5: Split Point Optimization [checkpoint: 9dbbd16]
- [x] Task: Write tests for finding the optimal Gini split on a single feature [0f24202]
- [x] Task: Implement `_find_best_threshold` for numerical features [ce28635]
- [x] Task: Implement basic categorical split search (binary split) [ce28635]
- [x] Task: Conductor - User Manual Verification 'Phase 5: Split Point Optimization' (Protocol in workflow.md) [9dbbd16]

## Phase 6: Recursive Tree Growth [checkpoint: 67c1c0b]
- [x] Task: Write tests for tree growing (depth, min_samples stopping) [1d2cb82]
- [x] Task: Implement recursive `_fit_node` algorithm [8d1479c]
- [x] Task: Implement leaf prediction logic (majority class) [8d1479c]
- [x] Task: Conductor - User Manual Verification 'Phase 6: Recursive Tree Growth' (Protocol in workflow.md) [67c1c0b]

## Phase 7: Prediction & Probability
- [ ] Task: Write tests for `predict` and `predict_proba`
- [ ] Task: Implement tree traversal for prediction
- [ ] Task: Implement `predict_proba` returning class frequencies
- [ ] Task: Conductor - User Manual Verification 'Phase 7: Prediction & Probability' (Protocol in workflow.md)

## Phase 8: Basic Interaction Detection (Fallback)
- [ ] Task: Write tests for simple interaction detection fallback
- [ ] Task: Implement interaction p-value calculation for variable pairs
- [ ] Task: Integrate interaction check if main effects are not significant
- [ ] Task: Conductor - User Manual Verification 'Phase 8: Basic Interaction Detection (Fallback)' (Protocol in workflow.md)

## Phase 9: Final Verification & Benchmarking
- [ ] Task: Run `check_estimator` from `sklearn.utils.estimator_checks`
- [ ] Task: Create a benchmark script comparing against `DecisionTreeClassifier` on bias-prone synthetic data
- [ ] Task: Final code cleanup and documentation pass
- [ ] Task: Conductor - User Manual Verification 'Phase 9: Final Verification & Benchmarking' (Protocol in workflow.md)
