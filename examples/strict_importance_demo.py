"""
Demonstration of Strict GUIDE Variable Importance.

This example shows how GUIDE's importance scores handle:
1. Interaction detection (capturing associative signals).
2. Bias correction (preventing high-cardinality variables from appearing important).
3. Normalization (scores near 1.0 indicate noise).
"""
import numpy as np
import pandas as pd
from pyguide import GuideTreeClassifier

def generate_biased_data(n_samples=1000, seed=42):
    rng = np.random.default_rng(seed)
    
    # 1. Main signal: X0
    x0 = rng.uniform(0, 1, n_samples)
    
    # 2. Interaction signal: X1 and X2 (XOR-like)
    x1 = rng.uniform(0, 1, n_samples)
    x2 = rng.uniform(0, 1, n_samples)
    
    # 3. High-cardinality noise: X3 (50 levels)
    # This usually trips up CART-based importance (e.g., Gini importance)
    x3 = rng.choice([f"cat_{i}" for i in range(50)], n_samples)
    
    # 4. Standard continuous noise: X4-X6
    x_noise = rng.uniform(0, 1, (n_samples, 3))
    
    # Target: depends on X0 (main) and XOR(X1, X2)
    # Note: X1 and X2 have NO marginal effect individually.
    y = ((x0 > 0.5) | ((x1 > 0.5) ^ (x2 > 0.5))).astype(int)
    
    df = pd.DataFrame({
        'signal_main': x0,
        'signal_int_1': x1,
        'signal_int_2': x2,
        'noise_high_card': x3,
        'noise_cont_1': x_noise[:, 0],
        'noise_cont_2': x_noise[:, 1],
        'noise_cont_3': x_noise[:, 2],
    })
    
    return df, y

def demo():
    print("Generating synthetic data with interactions and bias traps...")
    X, y = generate_biased_data()
    
    # Initialize GUIDE Classifier with interaction detection enabled
    clf = GuideTreeClassifier(interaction_depth=1, random_state=42)
    
    print("\n1. Calculating UNADJUSTED Importance (bias_correction=False)")
    print("This shows raw associative statistics...")
    unadjusted_scores = clf.compute_guide_importance(X, y, bias_correction=False)
    
    print("\n2. Calculating ADJUSTED Importance (bias_correction=True)")
    print("This normalizes scores against a permutation-based null (may take a moment)...")
    # Reduced permutations slightly for the demo speed, but default is 300
    adjusted_scores = clf.compute_guide_importance(X, y, bias_correction=True, n_permutations=100)
    
    # Prepare results for display
    results = pd.DataFrame({
        'Variable': X.columns,
        'Unadjusted': unadjusted_scores,
        'Adjusted (VI)': adjusted_scores
    }).sort_values('Adjusted (VI)', ascending=False)
    
    print("\nResults Table:")
    print("-" * 60)
    print(f"{ 'Variable':<20} | { 'Unadjusted':<12} | { 'Adjusted (VI)':<12}")
    print("-" * 60)
    for _, row in results.iterrows():
        v_name = str(row['Variable'])
        v_unadj = float(row['Unadjusted'])
        v_adj = float(row['Adjusted (VI)'])
        print(f"{v_name:<20} | {v_unadj:<12.2f} | {v_adj:<12.2f}")
    print("-" * 60)
    
    print("\nObservations:")
    print("- 'signal_main' (X0) has high scores in both.")
    print("- 'signal_int_1' and 'signal_int_2' (X1/X2) are correctly identified as important")
    print("  thanks to interaction detection, even though they have no main effect.")
    print("- 'noise_high_card' has its score brought down significantly in the Adjusted column.")
    print("- Adjusted scores near 1.0 (or below) reliably indicate noise variables.")

if __name__ == "__main__":
    demo()
