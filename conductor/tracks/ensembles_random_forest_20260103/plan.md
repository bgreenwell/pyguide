# Plan: Tree Ensembles: Random Forest

## Phase 1: Implement `max_features`
- [ ] Task: Add `max_features` parameter to `GuideTreeClassifier` and `GuideTreeRegressor`
- [ ] Task: Update `select_split_variable` to handle feature subsetting
- [ ] Task: Verify `max_features` logic with unit tests
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Implement max_features' (Protocol in workflow.md)

## Phase 2: Implement Ensemble Classes
- [ ] Task: Create `src/pyguide/ensemble.py`
- [ ] Task: Implement `GuideRandomForestClassifier` wrapping `BaggingClassifier`
- [ ] Task: Implement `GuideRandomForestRegressor` wrapping `BaggingRegressor`
- [ ] Task: Verify ensembles with unit tests and check_estimator
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Implement Ensemble Classes' (Protocol in workflow.md)

## Phase 3: Benchmarking & Docs
- [ ] Task: Add Random Forest benchmark to `benchmarks/main_benchmark.py`
- [ ] Task: Update documentation to include Ensembles
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Benchmarking & Docs' (Protocol in workflow.md)
