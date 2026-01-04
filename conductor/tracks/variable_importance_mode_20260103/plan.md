# Plan: Variable Importance Mode

## Phase 1: Metadata Tracking
- [x] Task: Update `GuideNode` to store `split_type` and `interaction_group` fb624fc
- [x] Task: Update `_fit_node` in `Classifier` and `Regressor` to populate this metadata 393d00c
- [~] Task: Conductor - User Manual Verification 'Phase 1: Metadata Tracking' (Protocol in workflow.md)

## Phase 2: Importance Calculation
- [ ] Task: Implement `interaction_importances_` calculation logic
- [ ] Task: Expose `interaction_importances_` property in estimators
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Importance Calculation' (Protocol in workflow.md)

## Phase 3: Reporting & Docs
- [ ] Task: Create `examples/importance_demo.py` showcasing the new metrics on interaction datasets
- [ ] Task: Update documentation to explain the difference between main and interaction importance
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Reporting & Docs' (Protocol in workflow.md)
