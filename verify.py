import numpy as np
from pyguide.splitting import find_best_split

def verify_sse_splitting():
    # Clear split point for SSE
    x = np.array([10, 20, 30, 40], dtype=float)
    y = np.array([5, 5, 25, 25], dtype=float)
    
    # Numerical split
    threshold, gain = find_best_split(x, y, is_categorical=False, criterion="mse")
    print(f"Numerical Threshold: {threshold}, Gain: {gain}")
    assert threshold == 25.0
    assert gain > 0
    
    # Categorical split
    xc = np.array(['low', 'low', 'high', 'high'])
    cat, gain_c = find_best_split(xc, y, is_categorical=True, criterion="mse")
    print(f"Categorical Split: {cat}, Gain: {gain_c}")
    assert cat in ['low', 'high']
    assert gain_c > 0
    
    print("SSE splitting verification successful!")

if __name__ == "__main__":
    verify_sse_splitting()
