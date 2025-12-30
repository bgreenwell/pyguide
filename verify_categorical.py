import numpy as np
from pyguide import GuideTreeRegressor

# Test categorical split storage as a set
# Use enough samples to ensure Chi-square test doesn't fail due to small size
# Or force the split with significance_threshold=1.0
X = np.array(["A"]*10 + ["B"]*10 + ["C"]*10).reshape(-1, 1)
y = np.concatenate([np.zeros(10), np.ones(10) * 10, np.ones(10) * 10])

# Force split by allowing any p-value
reg = GuideTreeRegressor(max_depth=1, significance_threshold=1.0, min_samples_split=2)
reg.fit(X, y)

print(f"Split threshold type: {type(reg.tree_.split_threshold)}")
print(f"Split threshold value: {reg.tree_.split_threshold}")

if reg.tree_.split_threshold is None:
    print("FAILURE: No split found.")
else:
    # Verification
    assert isinstance(reg.tree_.split_threshold, (set, frozenset)), "Threshold should be a set"
    # Given the data: A -> 0, B -> 10, C -> 10.
    # Optimal split separates A from {B, C}.
    # So threshold should be {'A'} (left) or {'B', 'C'} (left).
    # Based on sorting means: Mean(A)=0, Mean(B)=10, Mean(C)=10.
    # Sorted order: A, B, C (or A, C, B).
    # Split 1: Left={A}. Gain maximizes separation.
    assert reg.tree_.split_threshold == {"A"}

    print("\nSUCCESS: Categorical split thresholds are correctly stored as sets.")