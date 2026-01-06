import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_digits, load_iris
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from pyguide import (
    GuideGradientBoostingClassifier,
    GuideGradientBoostingRegressor,
    GuideRandomForestClassifier,
    GuideRandomForestRegressor,
    GuideTreeClassifier,
    GuideTreeRegressor,
)


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

    # 3. Scikit-learn Random Forest
    rf_sklearn = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
    start = time.time()
    rf_sklearn.fit(X_train, y_train)
    train_time_rf = time.time() - start
    
    start = time.time()
    score_rf = rf_sklearn.score(X_test, y_test)
    test_time_rf = time.time() - start
    
    results.append({
        "Model": "sklearn (Random Forest)",
        "Train Time (s)": train_time_rf,
        "Test Time (s)": test_time_rf,
        "Accuracy": score_rf
    })

    # 4. pyguide Random Forest
    rf_guide = GuideRandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
    start = time.time()
    rf_guide.fit(X_train, y_train)
    train_time_rf_guide = time.time() - start
    
    start = time.time()
    score_rf_guide = rf_guide.score(X_test, y_test)
    test_time_rf_guide = time.time() - start
    
    results.append({
        "Model": "pyguide (Random Forest)",
        "Train Time (s)": train_time_rf_guide,
        "Test Time (s)": test_time_rf_guide,
        "Accuracy": score_rf_guide
    })

    # 5. Scikit-learn Gradient Boosting
    # Note: sklearn GBM is slower than RF but more accurate usually
    gbm_sklearn = GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=42)
    start = time.time()
    gbm_sklearn.fit(X_train, y_train)
    train_time_gbm = time.time() - start
    
    start = time.time()
    score_gbm = gbm_sklearn.score(X_test, y_test)
    test_time_gbm = time.time() - start
    
    results.append({
        "Model": "sklearn (Gradient Boosting)",
        "Train Time (s)": train_time_gbm,
        "Test Time (s)": test_time_gbm,
        "Accuracy": score_gbm
    })

    # 6. pyguide Gradient Boosting
    # Only binary supported for now!
    if len(np.unique(y)) == 2:
        gbm_guide = GuideGradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=42)
        start = time.time()
        gbm_guide.fit(X_train, y_train)
        train_time_gbm_guide = time.time() - start
        
        start = time.time()
        score_gbm_guide = gbm_guide.score(X_test, y_test)
        test_time_gbm_guide = time.time() - start
        
        results.append({
            "Model": "pyguide (Gradient Boosting)",
            "Train Time (s)": train_time_gbm_guide,
            "Test Time (s)": test_time_gbm_guide,
            "Accuracy": score_gbm_guide
        })
    else:
        results.append({
            "Model": "pyguide (Gradient Boosting)",
            "Train Time (s)": 0.0,
            "Test Time (s)": 0.0,
            "Accuracy": 0.0 # Not supported
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

    # 3. Scikit-learn Random Forest
    rf_sklearn = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
    start = time.time()
    rf_sklearn.fit(X_train, y_train)
    train_time_rf = time.time() - start
    
    start = time.time()
    score_rf = rf_sklearn.score(X_test, y_test)
    test_time_rf = time.time() - start
    
    results.append({
        "Model": "sklearn (Random Forest)",
        "Train Time (s)": train_time_rf,
        "Test Time (s)": test_time_rf,
        "R2 Score": score_rf
    })

    # 4. pyguide Random Forest
    rf_guide = GuideRandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
    start = time.time()
    rf_guide.fit(X_train, y_train)
    train_time_rf_guide = time.time() - start
    
    start = time.time()
    score_rf_guide = rf_guide.score(X_test, y_test)
    test_time_rf_guide = time.time() - start
    
    results.append({
        "Model": "pyguide (Random Forest)",
        "Train Time (s)": train_time_rf_guide,
        "Test Time (s)": test_time_rf_guide,
        "R2 Score": score_rf_guide
    })

    # 5. Scikit-learn Gradient Boosting
    gbm_sklearn = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=42)
    start = time.time()
    gbm_sklearn.fit(X_train, y_train)
    train_time_gbm = time.time() - start
    
    start = time.time()
    score_gbm = gbm_sklearn.score(X_test, y_test)
    test_time_gbm = time.time() - start
    
    results.append({
        "Model": "sklearn (Gradient Boosting)",
        "Train Time (s)": train_time_gbm,
        "Test Time (s)": test_time_gbm,
        "R2 Score": score_gbm
    })

    # 6. pyguide Gradient Boosting
    gbm_guide = GuideGradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=42)
    start = time.time()
    gbm_guide.fit(X_train, y_train)
    train_time_gbm_guide = time.time() - start
    
    start = time.time()
    score_gbm_guide = gbm_guide.score(X_test, y_test)
    test_time_gbm_guide = time.time() - start
    
    results.append({
        "Model": "pyguide (Gradient Boosting)",
        "Train Time (s)": train_time_gbm_guide,
        "Test Time (s)": test_time_gbm_guide,
        "R2 Score": score_gbm_guide
    })
    
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    print()
    return df

if __name__ == "__main__":
    # Classification
    # Note: Iris and Digits are multiclass, so GBM won't run for pyguide
    X_iris, y_iris = load_iris(return_X_y=True)
    benchmark_classifier("Iris", X_iris, y_iris)
    
    X_digits, y_digits = load_digits(return_X_y=True)
    benchmark_classifier("Digits", X_digits, y_digits)
    
    # Regression
    X_diabetes, y_diabetes = load_diabetes(return_X_y=True)
    benchmark_regressor("Diabetes", X_diabetes, y_diabetes)
