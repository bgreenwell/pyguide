import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from pyguide import GuideTreeClassifier

def generate_bias_data(n_samples=500, random_seed=42):
    """
    Generate data where Feature 0 is binary and predictive,
    and Feature 1 is continuous and noise.
    Standard CART usually biases towards Feature 1.
    """
    np.random.seed(random_seed)
    
    # Feature 0: Binary, perfect predictor (with 5% noise)
    x0 = np.random.randint(0, 2, n_samples)
    y = x0.copy()
    flip_mask = np.random.rand(n_samples) < 0.05
    y[flip_mask] = 1 - y[flip_mask]
    
    # Feature 1: Continuous noise
    x1 = np.random.rand(n_samples)
    
    X = np.column_stack([x0, x1])
    return X, y

def run_benchmark():
    print("Running Selection Bias Benchmark...")
    print("-" * 40)
    X, y = generate_bias_data()
    
    # 1. Standard CART (Sklearn)
    cart = DecisionTreeClassifier(max_depth=1, random_state=42)
    start = time.time()
    cart.fit(X, y)
    cart_time = time.time() - start
    cart_feat = cart.tree_.feature[0]
    
    # 2. GUIDE
    guide = GuideTreeClassifier(max_depth=1)
    start = time.time()
    guide.fit(X, y)
    guide_time = time.time() - start
    guide_feat = guide.tree_.split_feature
    
    print(f"CART selected feature: {cart_feat} (Expected: 0, biased to 1)")
    print(f"GUIDE selected feature: {guide_feat} (Expected: 0)")
    print("-" * 40)
    print(f"CART fit time: {cart_time:.4f}s")
    print(f"GUIDE fit time: {guide_time:.4f}s")
    
    if guide_feat == 0 and cart_feat != 0:
        print("\nSUCCESS: GUIDE avoided the variable selection bias that CART fell for.")
    elif guide_feat == 0:
        print("\nSUCCESS: GUIDE correctly selected the predictive feature.")
    else:
        print("\nFAILURE: GUIDE was also biased.")

if __name__ == "__main__":
    run_benchmark()
