# Plan: Implement GuideTreeRegressor with residual-based selection

## Phase 1: Regressor Foundation [checkpoint: 4cfe7d4]
- [x] Task: Write tests for `GuideTreeRegressor` interface (init, fit, predict) [7c154c3]
- [x] Task: Implement `GuideTreeRegressor` shell with `RegressorMixin` and basic validation [7c154c3]
- [x] Task: Conductor - User Manual Verification 'Phase 1: Regressor Foundation' (Protocol in workflow.md) [4cfe7d4]

## Phase 2: Regression Variable Selection [checkpoint: 1fdcafa]
- [x] Task: Write tests for residual-to-class transformation logic [c7381e3]
- [x] Task: Implement residual-based target creation in `_fit_node` [1540fd4]
- [x] Task: Integrate `select_split_variable` with the new target [1540fd4]
- [x] Task: Conductor - User Manual Verification 'Phase 2: Regression Variable Selection' (Protocol in workflow.md) [1fdcafa]

## Phase 3: SSE Split Optimization [checkpoint: 1b2c460]
- [x] Task: Write tests for finding the optimal SSE split [3a3d97f]
- [x] Task: Implement SSE calculation and optimization helper (`_find_best_threshold_sse`) [3a3d97f]
- [x] Task: Implement categorical SSE split search [3a3d97f]
- [x] Task: Conductor - User Manual Verification 'Phase 3: SSE Split Optimization' (Protocol in workflow.md) [1b2c460]

## Phase 4: Recursive Tree Growth & Prediction [checkpoint: 56ba075]
- [x] Task: Write tests for recursive regression tree fitting [0bf88ac]
- [x] Task: Implement `_fit_node` specific to regression (mean prediction) [0bf88ac]
- [x] Task: Implement `predict` for regression tree traversal [0bf88ac]
- [x] Task: Conductor - User Manual Verification 'Phase 4: Recursive Tree Growth & Prediction' (Protocol in workflow.md) [56ba075]

## Phase 5: Final Verification & Benchmarking
- [x] Task: Run `check_estimator` for `GuideTreeRegressor` [4761945]
- [x] Task: Create a benchmark script comparing against `DecisionTreeRegressor` [e7745a0]
- [ ] Task: Code cleanup and documentation
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Final Verification & Benchmarking' (Protocol in workflow.md)
