# Spec: Refining Interaction Detection

## Overview
This track focuses on improving the efficiency and flexibility of the interaction detection mechanism in `pyguide`. Currently, the library performs an exhaustive $O(d^2)$ search of all pairwise interactions when individual variable selection fails to find a significant split. This can be slow for datasets with many features.

## Goals
- **Efficiency:** Reduce the number of feature pairs tested for interactions by implementing pre-filtering.
- **Flexibility:** Allow users to constrain the interaction search to specific feature sets.
- **Capability:** Support higher-order interactions beyond simple pairs (depth > 1).

## Key Components

### 1. Interaction Constraints
- **Attribute:** `interaction_features` (list of feature indices or names).
- **Behavior:** Only features in this list will be considered as candidates for interaction tests.

### 2. Candidate Pre-filtering
- **Mechanism:** Before performing exhaustive pairwise tests, rank features based on their individual Chi-square p-values. Only test interactions between the top $K$ features (where $K$ is configurable).
- **Attribute:** `max_interaction_candidates` (default: None, meaning all).

### 3. Higher-Order Interactions
- **Mechanism:** Generalize the search logic to support `interaction_depth` > 1. This involves testing triplets, etc., using recursive or iterative grouping.
- **Note:** This will likely be limited to a small number of top candidates to avoid combinatorial explosion.

## Success Criteria
- Large feature sets (e.g., 100+ features) show faster training times when `max_interaction_candidates` is used.
- Models correctly identify and split on higher-order interactions in synthetic tests.
- `interaction_features` correctly restricts the search space.
