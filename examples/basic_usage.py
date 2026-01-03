"""
Basic usage example for pyguide showing Classification and Regression.
"""
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from pyguide import GuideTreeClassifier, GuideTreeRegressor

def classification_demo():
    print("--- Classification Demo (Iris Dataset) ---")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = GuideTreeClassifier(max_depth=3, interaction_depth=1)
    clf.fit(X_train, y_train)

    print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
    print(f"Tree structure: {clf.n_leaves_} leaves, depth {clf.max_depth_}")
    print(f"Feature importances: {clf.feature_importances_}")
    print()

def regression_demo():
    print("--- Regression Demo (Diabetes Dataset) ---")
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    reg = GuideTreeRegressor(max_depth=4, interaction_depth=1)
    reg.fit(X_train, y_train)

    print(f"R2 Score: {reg.score(X_test, y_test):.4f}")
    print(f"Tree structure: {reg.n_leaves_} leaves, depth {reg.max_depth_}")
    print()

if __name__ == "__main__":
    classification_demo()
    regression_demo()
