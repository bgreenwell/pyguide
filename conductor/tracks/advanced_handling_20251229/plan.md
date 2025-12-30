# Plan: Advanced Categorical and Missing Value Handling

## Phase 1: Missing Value Support (Infrastructure) [checkpoint: a78b271]
- [x] Task: Create tests for missing value handling in `fit` and `predict` (expect failure) [f6e384a]
- [x] Task: Update `GuideNode` to store missing value direction [80f002c]
- [x] Task: Update `find_best_split` to optimize missing value direction [6cb60b5]
- [x] Task: Update `predict` to handle missing values using the new node logic [e20a30b]
- [x] Task: Conductor - User Manual Verification 'Phase 1: Missing Value Support (Infrastructure)' (Protocol in workflow.md) [a78b271]

## Phase 2: Optimized Categorical Splitting
- [ ] Task: Create tests for high-cardinality categorical splits (performance/correctness)
- [ ] Task: Implement ordered categorical splitting for Regression (mean target)
- [ ] Task: Implement ordered categorical splitting for Binary Classification (class probability)
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Optimized Categorical Splitting' (Protocol in workflow.md)

## Phase 3: Final Integration & Verification
- [ ] Task: Update `check_estimator` tags to allow NaN
- [ ] Task: Verify scikit-learn compatibility with missing values
- [ ] Task: Final code cleanup and documentation
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Final Integration & Verification' (Protocol in workflow.md)
