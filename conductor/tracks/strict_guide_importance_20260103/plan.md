# Plan: Strict GUIDE Variable Importance

## Phase 1: Core Algorithm [checkpoint: 17d479b]
- [x] Task: Add `compute_guide_importance` method to `GuideTreeClassifier` and `GuideTreeRegressor`
- [x] Task: Implement auxiliary tree growth logic (depth-restricted, unpruned)
- [x] Task: Implement permutation loop for bias correction
- [x] Task: Conductor - User Manual Verification 'Phase 1: Core Algorithm' (Protocol in workflow.md)

## Phase 2: Verification & Benchmarking
- [x] Task: Create `examples/strict_importance_demo.py` comparing adjusted vs unadjusted scores
- [x] Task: Verify that noise variables get normalized scores near 1.0
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Verification & Benchmarking' (Protocol in workflow.md)

## Phase 3: Documentation
- [ ] Task: Update API documentation for the new method
- [ ] Task: Update User Guide with details from the Loh & Zhou (2021) paper
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Documentation' (Protocol in workflow.md)
