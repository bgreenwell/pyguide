# Plan: Refining Interaction Detection

## Phase 1: Constraints and Filtering [checkpoint: 146d692]
- [x] Task: Implement `interaction_features` and `max_interaction_candidates` parameters [106d42e]
- [x] Task: Update `_select_split_variable` to use the new filtering logic [106d42e]
- [x] Task: Create tests for interaction constraints (verify restricted search space) [106d42e]
- [x] Task: Conductor - User Manual Verification 'Phase 1: Constraints and Filtering' (Protocol in workflow.md) [146d692]

## Phase 2: Higher-Order Interactions
- [ ] Task: Generalize `calc_interaction_p_value` to support more than 2 variables (or implement recursive grouping)
- [ ] Task: Update `_select_split_variable` to handle `interaction_depth > 1`
- [ ] Task: Create tests for triplet interactions
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Higher-Order Interactions' (Protocol in workflow.md)

## Phase 3: Final Integration & Benchmarking
- [ ] Task: Benchmark the impact of candidate filtering on large feature sets
- [ ] Task: Final code cleanup and documentation update
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Final Integration & Benchmarking' (Protocol in workflow.md)
