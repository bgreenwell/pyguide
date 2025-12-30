class GuideNode:
    """
    A node in the GUIDE tree.
    """

    def __init__(
        self,
        depth,
        is_leaf=False,
        prediction=None,
        probabilities=None,
        split_feature=None,
        split_threshold=None,
        missing_go_left=True,
        left=None,
        right=None,
    ):
        self.depth = depth
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.probabilities = probabilities
        self.split_feature = split_feature
        self.split_threshold = split_threshold
        self.missing_go_left = missing_go_left
        self.left = left
        self.right = right
