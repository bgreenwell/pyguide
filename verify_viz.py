import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from pyguide import GuideTreeClassifier, plot_tree

iris = load_iris()
X, y = iris.data, iris.target

clf = GuideTreeClassifier(max_depth=3)
clf.fit(X, y)

print("Tree built. Attempting pyguide.plot_tree...")

try:
    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=False)
    plt.savefig("iris_guide_tree_final.png")
    print("SUCCESS: iris_guide_tree_final.png has been generated.")
except Exception as e:
    print(f"pyguide.plot_tree failed: {e}")
    import traceback
    traceback.print_exc()