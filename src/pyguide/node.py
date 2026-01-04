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
        n_samples=0,
        impurity=0.0,
        value_distribution=None,
        node_id=None,
        split_type=None,
        interaction_group=None,
        curvature_stats=None,
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
        self.n_samples = n_samples
        self.impurity = impurity
        self.value_distribution = value_distribution
        self.node_id = node_id
        self.split_type = split_type
        self.interaction_group = interaction_group
        self.curvature_stats = curvature_stats

    def is_leaf_node(self):
        return self.is_leaf
