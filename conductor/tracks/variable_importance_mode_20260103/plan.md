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

## Phase 3: True GUIDE Scoring [checkpoint: cb5d785]
- [x] Task: Update `GuideNode` to store `curvature_p_values` from variable selection 8424e51
- [x] Task: Update `_fit_node` to save these p-values 7efd63c
- [x] Task: Implement `guide_importances_` (Eq 1 from paper) 30c1902
- [x] Task: Conductor - User Manual Verification 'Phase 3: True GUIDE Scoring' (Protocol in workflow.md) cb5d785
- [x] Task: Conductor - User Manual Verification 'Phase 3: True GUIDE Scoring' (Protocol in workflow.md)

## Phase 4: Reporting & Docs
- [x] Task: Create `examples/importance_demo.py` showcasing the new metrics on interaction datasets cb5d785
- [~] Task: Update documentation to explain the difference between main, interaction, and GUIDE importance
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Reporting & Docs' (Protocol in workflow.md)
