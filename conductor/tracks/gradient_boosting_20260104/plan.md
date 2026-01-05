# Plan: Gradient Boosting

## Phase 1: Gradient Boosting Regressor [checkpoint: 42bc225]
- [x] Task: Implement `GuideGradientBoostingRegressor` skeleton and parameters
- [x] Task: Implement Least Squares boosting loop using `GuideTreeRegressor`
- [x] Task: Add support for `subsample` (Stochastic Gradient Boosting)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Regressor' (Protocol in workflow.md)

## Phase 2: Gradient Boosting Classifier (Binary)
- [ ] Task: Implement `GuideGradientBoostingClassifier` skeleton
- [ ] Task: Implement Binary Deviance boosting loop
- [ ] Task: Implement `predict_proba` using sigmoid transformation
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Classifier' (Protocol in workflow.md)

## Phase 3: Benchmarking & Docs
- [ ] Task: Add `GuideGradientBoosting*` to `pyguide.ensemble` module
- [ ] Task: Create benchmarks comparing against sklearn GBM
- [ ] Task: Update Documentation and Examples
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Benchmarking & Docs' (Protocol in workflow.md)
