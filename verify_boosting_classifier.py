from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from pyguide import GuideGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

def verify_boosting_classifier():
    print("Loading Breast Cancer dataset...")
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    print("Training GuideGradientBoostingClassifier (n_estimators=50, lr=0.1, max_depth=2)...")
    gbm = GuideGradientBoostingClassifier(
        n_estimators=50, 
        learning_rate=0.1, 
        max_depth=2, 
        random_state=42
    )
    gbm.fit(X_train, y_train)
    
    y_pred = gbm.predict(X_test)
    y_prob = gbm.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC:  {roc:.4f}")
    
    if acc > 0.90:
        print("SUCCESS: Accuracy is good.")
    else:
        print("WARNING: Accuracy is lower than expected.")

if __name__ == "__main__":
    verify_boosting_classifier()
