# Plan: Pruning, Visualization, and Optimization

## Phase 1: Visualization Compatibility [checkpoint: d59ace6]
- [x] Task: Investigate and prototype `sklearn.tree.Tree` structure mapping [eaa3959]
- [x] Task: Implement `_build_sklearn_tree` method to convert `GuideNode` to `sklearn.tree.Tree` [5611cb4]
- [x] Task: Verify visualization with `sklearn.tree.plot_tree` using a manual script [5611cb4]
- [x] Task: Conductor - User Manual Verification 'Phase 1: Visualization Compatibility' (Protocol in workflow.md) [d59ace6]

## Phase 2: Cost-Complexity Pruning
- [x] Task: Create tests for `ccp_alpha` (verify tree depth reduction) [b364bbd]
- [x] Task: Implement `ccp_alpha` parameter and `cost_complexity_pruning_path` stub [42175cd]
- [ ] Task: Implement pruning logic (post-pruning) in `fit`
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Cost-Complexity Pruning' (Protocol in workflow.md)

## Phase 3: Performance Optimization
- [ ] Task: Create a comprehensive performance benchmark (profiling baseline)
- [ ] Task: Optimize `calc_curvature_p_value` (vectorize contingency table creation if possible)
- [ ] Task: Optimize `find_best_split` (vectorize loop over thresholds)
- [ ] Task: Verify performance improvements against baseline
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Performance Optimization' (Protocol in workflow.md)
