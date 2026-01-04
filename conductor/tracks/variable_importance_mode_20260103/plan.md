# Plan: Variable Importance Mode

## Phase 1: Metadata Tracking [checkpoint: 35fa5b6]
- [x] Task: Update `GuideNode` to store `split_type` and `interaction_group` fb624fc
- [x] Task: Update `_fit_node` in `Classifier` and `Regressor` to populate this metadata 393d00c
- [x] Task: Conductor - User Manual Verification 'Phase 1: Metadata Tracking' (Protocol in workflow.md) 35fa5b6

## Phase 2: Importance Calculation
- [x] Task: Implement `interaction_importances_` calculation logic 1588b7b
- [x] Task: Expose `interaction_importances_` property in estimators 1588b7b
- [~] Task: Conductor - User Manual Verification 'Phase 2: Importance Calculation' (Protocol in workflow.md)
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Importance Calculation' (Protocol in workflow.md)

## Phase 3: Reporting & Docs
- [ ] Task: Create `examples/importance_demo.py` showcasing the new metrics on interaction datasets
- [ ] Task: Update documentation to explain the difference between main and interaction importance
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Reporting & Docs' (Protocol in workflow.md)
