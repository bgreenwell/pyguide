# Plan: Build the core GuideTreeClassifier infrastructure

## Phase 1: Project Scaffolding & CI/CD
- [x] Task: Initialize Python project with `uv` and `pyproject.toml` (Poetry style) [dbdedd1]
- [x] Task: Configure `ruff` and `pytest` with coverage requirements [c9768e7]
- [ ] Task: Create basic directory structure (`src/pyguide`, `tests/`)
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Project Scaffolding & CI/CD' (Protocol in workflow.md)

## Phase 2: Scikit-learn Compliance & Base Structure
- [ ] Task: Write tests for `GuideTreeClassifier` basic interface (init, fit, predict)
- [ ] Task: Implement `GuideTreeClassifier` shell with scikit-learn validation (`check_X_y`, `check_array`)
- [ ] Task: Create `GuideNode` class for tree structure
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Scikit-learn Compliance & Base Structure' (Protocol in workflow.md)

## Phase 3: Statistical Testing Utility
- [ ] Task: Write tests for Chi-square p-value calculation (main effects)
- [ ] Task: Implement `calc_curvature_p_value` helper using `scipy.stats.chi2_contingency`
- [ ] Task: Implement data binning for continuous variables (GUIDE style)
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Statistical Testing Utility' (Protocol in workflow.md)

## Phase 4: Variable Selection Logic
- [ ] Task: Write tests for `select_split_variable`
- [ ] Task: Implement variable selection loop across all features
- [ ] Task: Implement handling of categorical vs numerical features in selection
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Variable Selection Logic' (Protocol in workflow.md)

## Phase 5: Split Point Optimization
- [ ] Task: Write tests for finding the optimal Gini split on a single feature
- [ ] Task: Implement `_find_best_threshold` for numerical features
- [ ] Task: Implement basic categorical split search (binary split)
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Split Point Optimization' (Protocol in workflow.md)

## Phase 6: Recursive Tree Growth
- [ ] Task: Write tests for tree growing (depth, min_samples stopping)
- [ ] Task: Implement recursive `_fit_node` algorithm
- [ ] Task: Implement leaf prediction logic (majority class)
- [ ] Task: Conductor - User Manual Verification 'Phase 6: Recursive Tree Growth' (Protocol in workflow.md)

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
