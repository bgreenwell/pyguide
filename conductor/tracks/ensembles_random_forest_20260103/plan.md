# Plan: Tree Ensembles: Random Forest

## Phase 1: Implement `max_features` [checkpoint: 9859363]
- [x] Task: Add `max_features` parameter to `GuideTreeClassifier` and `GuideTreeRegressor` [9b66157]
- [x] Task: Update `select_split_variable` to handle feature subsetting [9b66157]
- [x] Task: Verify `max_features` logic with unit tests [9b66157]
- [x] Task: Conductor - User Manual Verification 'Phase 1: Implement max_features' (Protocol in workflow.md) [9859363]

## Phase 2: Implement Ensemble Classes
- [x] Task: Create `src/pyguide/ensemble.py` [07d2190]
- [x] Task: Implement `GuideRandomForestClassifier` wrapping `BaggingClassifier` [07d2190]
- [x] Task: Implement `GuideRandomForestRegressor` wrapping `BaggingRegressor` [07d2190]
- [x] Task: Verify ensembles with unit tests and check_estimator [07d2190]
- [~] Task: Conductor - User Manual Verification 'Phase 2: Implement Ensemble Classes' (Protocol in workflow.md)
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Implement Ensemble Classes' (Protocol in workflow.md)

## Phase 3: Benchmarking & Docs
- [x] Task: Add Random Forest benchmark to `benchmarks/main_benchmark.py` [a1b2c3d]
- [~] Task: Update documentation to include Ensembles
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Benchmarking & Docs' (Protocol in workflow.md)
