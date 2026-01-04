# Plan: Strict GUIDE Variable Importance

## Phase 1: Core Algorithm
- [ ] Task: Add `compute_guide_importance` method to `GuideTreeClassifier` and `GuideTreeRegressor`
- [ ] Task: Implement auxiliary tree growth logic (depth-restricted, unpruned)
- [ ] Task: Implement permutation loop for bias correction
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Core Algorithm' (Protocol in workflow.md)

## Phase 2: Verification & Benchmarking
- [ ] Task: Create `examples/strict_importance_demo.py` comparing adjusted vs unadjusted scores
- [ ] Task: Verify that noise variables get normalized scores near 1.0
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Verification & Benchmarking' (Protocol in workflow.md)

## Phase 3: Documentation
- [ ] Task: Update API documentation for the new method
- [ ] Task: Update User Guide with details from the Loh & Zhou (2021) paper
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Documentation' (Protocol in workflow.md)
