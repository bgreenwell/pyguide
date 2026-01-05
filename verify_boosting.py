from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from pyguide import GuideGradientBoostingRegressor
from sklearn.metrics import r2_score

def verify_boosting():
    print("Loading Diabetes dataset...")
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    print("Training GuideGradientBoostingRegressor (n_estimators=100, lr=0.1, max_depth=3)...")
    gbm = GuideGradientBoostingRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42
    )
    gbm.fit(X_train, y_train)
    
    y_pred = gbm.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score: {r2:.4f}")
    
    if r2 > 0.4:
        print("SUCCESS: R2 Score is reasonable for this dataset.")
    else:
        print("WARNING: R2 Score is lower than expected.")

if __name__ == "__main__":
    verify_boosting()
