import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_digits, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from pyguide import GuideTreeClassifier, GuideTreeRegressor

def benchmark_classifier(name, X, y):
    print(f"--- Benchmarking Classifier: {name} ({X.shape[0]} samples, {X.shape[1]} features) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []
    
    # 1. Scikit-learn CART
    clf_cart = DecisionTreeClassifier(max_depth=5, random_state=42)
    start = time.time()
    clf_cart.fit(X_train, y_train)
    train_time_cart = time.time() - start
    
    start = time.time()
    score_cart = clf_cart.score(X_test, y_test)
    test_time_cart = time.time() - start
    
    results.append({
        "Model": "sklearn (CART)",
        "Train Time (s)": train_time_cart,
        "Test Time (s)": test_time_cart,
        "Accuracy": score_cart
    })
    
    # 2. pyguide GUIDE
    clf_guide = GuideTreeClassifier(max_depth=5, interaction_depth=0)
    start = time.time()
    clf_guide.fit(X_train, y_train)
    train_time_guide = time.time() - start
    
    start = time.time()
    score_guide = clf_guide.score(X_test, y_test)
    test_time_guide = time.time() - start
    
    results.append({
        "Model": "pyguide (GUIDE)",
        "Train Time (s)": train_time_guide,
        "Test Time (s)": test_time_guide,
        "Accuracy": score_guide
    })
    
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    print()
    return df

def benchmark_regressor(name, X, y):
    print(f"--- Benchmarking Regressor: {name} ({X.shape[0]} samples, {X.shape[1]} features) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []
    
    # 1. Scikit-learn CART
    reg_cart = DecisionTreeRegressor(max_depth=5, random_state=42)
    start = time.time()
    reg_cart.fit(X_train, y_train)
    train_time_cart = time.time() - start
    
    start = time.time()
    score_cart = reg_cart.score(X_test, y_test)
    test_time_cart = time.time() - start
    
    results.append({
        "Model": "sklearn (CART)",
        "Train Time (s)": train_time_cart,
        "Test Time (s)": test_time_cart,
        "R2 Score": score_cart
    })
    
    # 2. pyguide GUIDE
    reg_guide = GuideTreeRegressor(max_depth=5, interaction_depth=0)
    start = time.time()
    reg_guide.fit(X_train, y_train)
    train_time_guide = time.time() - start
    
    start = time.time()
    score_guide = reg_guide.score(X_test, y_test)
    test_time_guide = time.time() - start
    
    results.append({
        "Model": "pyguide (GUIDE)",
        "Train Time (s)": train_time_guide,
        "Test Time (s)": test_time_guide,
        "R2 Score": score_guide
    })
    
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    print()
    return df

if __name__ == "__main__":
    # Classification
    X_iris, y_iris = load_iris(return_X_y=True)
    benchmark_classifier("Iris", X_iris, y_iris)
    
    X_digits, y_digits = load_digits(return_X_y=True)
    benchmark_classifier("Digits", X_digits, y_digits)
    
    # Regression
    X_diabetes, y_diabetes = load_diabetes(return_X_y=True)
    benchmark_regressor("Diabetes", X_diabetes, y_diabetes)
