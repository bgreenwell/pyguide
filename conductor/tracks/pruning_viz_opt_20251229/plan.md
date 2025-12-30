# Plan: Pruning, Visualization, and Optimization

## Phase 1: Visualization Compatibility
- [ ] Task: Investigate and prototype `sklearn.tree.Tree` structure mapping
- [ ] Task: Implement `_build_sklearn_tree` method to convert `GuideNode` to `sklearn.tree.Tree`
- [ ] Task: Verify visualization with `sklearn.tree.plot_tree` using a manual script
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Visualization Compatibility' (Protocol in workflow.md)

## Phase 2: Cost-Complexity Pruning
- [ ] Task: Create tests for `ccp_alpha` (verify tree depth reduction)
- [ ] Task: Implement `ccp_alpha` parameter and `cost_complexity_pruning_path` stub
- [ ] Task: Implement pruning logic (post-pruning) in `fit`
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Cost-Complexity Pruning' (Protocol in workflow.md)

## Phase 3: Performance Optimization
- [ ] Task: Create a comprehensive performance benchmark (profiling baseline)
- [ ] Task: Optimize `calc_curvature_p_value` (vectorize contingency table creation if possible)
- [ ] Task: Optimize `find_best_split` (vectorize loop over thresholds)
- [ ] Task: Verify performance improvements against baseline
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Performance Optimization' (Protocol in workflow.md)
