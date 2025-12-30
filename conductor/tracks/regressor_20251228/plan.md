# Plan: Implement GuideTreeRegressor with residual-based selection

## Phase 1: Regressor Foundation
- [x] Task: Write tests for `GuideTreeRegressor` interface (init, fit, predict) [7c154c3]
- [ ] Task: Implement `GuideTreeRegressor` shell with `RegressorMixin` and basic validation
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Regressor Foundation' (Protocol in workflow.md)

## Phase 2: Regression Variable Selection
- [ ] Task: Write tests for residual-to-class transformation logic
- [ ] Task: Implement residual-based target creation in `_fit_node`
- [ ] Task: Integrate `select_split_variable` with the new target
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Regression Variable Selection' (Protocol in workflow.md)

## Phase 3: SSE Split Optimization
- [ ] Task: Write tests for finding the optimal SSE split
- [ ] Task: Implement SSE calculation and optimization helper (`_find_best_threshold_sse`)
- [ ] Task: Implement categorical SSE split search
- [ ] Task: Conductor - User Manual Verification 'Phase 3: SSE Split Optimization' (Protocol in workflow.md)

## Phase 4: Recursive Tree Growth & Prediction
- [ ] Task: Write tests for recursive regression tree fitting
- [ ] Task: Implement `_fit_node` specific to regression (mean prediction)
- [ ] Task: Implement `predict` for regression tree traversal
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Recursive Tree Growth & Prediction' (Protocol in workflow.md)

## Phase 5: Final Verification & Benchmarking
- [ ] Task: Run `check_estimator` for `GuideTreeRegressor`
- [ ] Task: Create a benchmark script comparing against `DecisionTreeRegressor`
- [ ] Task: Code cleanup and documentation
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Final Verification & Benchmarking' (Protocol in workflow.md)
