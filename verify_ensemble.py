from sklearn.datasets import load_digits, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from pyguide import GuideRandomForestClassifier, GuideRandomForestRegressor
import numpy as np

def verify_rf_classifier():
    print("--- Verifying GuideRandomForestClassifier ---")
    X, y = load_digits(return_X_y=True)
    
    # pyguide
    clf = GuideRandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
    clf.fit(X, y)
    score = clf.score(X, y)
    print(f"pyguide Accuracy: {score:.4f}")
    
    # sklearn
    clf_sk = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
    clf_sk.fit(X, y)
    score_sk = clf_sk.score(X, y)
    print(f"sklearn Accuracy: {score_sk:.4f}")
    print()

def verify_rf_regressor():
    print("--- Verifying GuideRandomForestRegressor ---")
    X, y = load_diabetes(return_X_y=True)
    
    # pyguide
    reg = GuideRandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
    reg.fit(X, y)
    score = reg.score(X, y)
    print(f"pyguide R2 Score: {score:.4f}")
    
    # sklearn
    reg_sk = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
    reg_sk.fit(X, y)
    score_sk = reg_sk.score(X, y)
    print(f"sklearn R2 Score: {score_sk:.4f}")
    print()

if __name__ == "__main__":
    verify_rf_classifier()
    verify_rf_regressor()
    print("Verification script finished successfully!")
