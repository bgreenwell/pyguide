class GuideNode:
    """
    A node in the GUIDE tree.
    """
    def __init__(
        self,
        depth,
        is_leaf=False,
        prediction=None,
        probabilities=None, # Add probabilities
        split_feature=None,
        split_threshold=None,
        left=None,
        right=None
    ):
        self.depth = depth
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.probabilities = probabilities
        self.split_feature = split_feature
        self.split_threshold = split_threshold
        self.left = left
        self.right = right