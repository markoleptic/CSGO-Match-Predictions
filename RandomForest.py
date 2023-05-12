import numpy as np
from scipy.stats import mode
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import encode_labels

class DecisionTree:
    """
    Binary decision tree using the Gini impurity as the splitting
    criterion.
    """
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self.build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self.predict_one_sample(x, self.tree) for x in X])
    
    def predict_one_sample(self, x, node):
        """
        Returns binary classification if node is a leaf node, 
        otherwise it recursively calls itself until it finds 
        a leaf node.
        """
        if node.is_leaf:
            if node.value >= 0.5:
                return 1
            else:
                return 0
        if x[node.feature] <= node.threshold:
            return self.predict_one_sample(x, node.left)
        else:
            return self.predict_one_sample(x, node.right)
    
    def build_tree(self, X, y, depth):
        """
        Find the best split by iterating over all the features. 
        Create a new node with the best split, and recursively 
        call build_tree on the split until all are leaf nodes.
        Return the node containing the split feature, value, and
        the left & right child nodes.
        """
        # check if max depth is reached
        if depth == self.max_depth or len(y) == 0:
            return Node(True, np.mean(y))
        best_split = self.get_best_split(X, y)
        if best_split is None:
            return Node(True, np.mean(y))
        best_feature, best_threshold = best_split
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold
        left_tree = self.build_tree(X[left_idx], y[left_idx], depth+1)
        right_tree = self.build_tree(X[right_idx], y[right_idx], depth+1)
        return Node(False, None, best_feature, best_threshold, left_tree, right_tree)

    def get_best_split(self, X, y):
        """
        Finds the best feature/threshold to split the data
        based on minimizing Gini impurity.
        """
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]
        for feature in range(n_features):
            feature_values = X[:, feature]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                left_idx = feature_values <= threshold
                right_idx = feature_values > threshold
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                gain = self.gini_gain(y, left_idx, right_idx)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        if best_feature is None or best_threshold is None:
            return None
        return best_feature, best_threshold
    
    def gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs**2)
    
    def gini_gain(self, y, left_idx, right_idx):
        """
        Calculates the Gini impurity gain for a given split, which 
        is a measure of how effective the split is in seperating 
        the two classes.
        """
        p = len(y[left_idx]) / len(y)
        left_gini = self.gini(y[left_idx])
        right_gini = self.gini(y[right_idx])
        return p*left_gini + (1-p)*right_gini
    
class Node:
    def __init__(self, is_leaf, value, feature=None, threshold=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        
    def fit(self, X, y):
        for i in range(self.n_estimators):
            print(i)
            tree = DecisionTree(max_depth=self.max_depth)
            # Randomly select a subset of features for each tree
            indices = np.random.choice(X.shape[1], size=int(np.sqrt(X.shape[1])), replace=False)
            tree.fit(X[:, indices], y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            indices = np.random.choice(X.shape[1], size=int(np.sqrt(X.shape[1])), replace=False)
            predictions[:, i] = tree.predict(X[:, indices])
        return mode(predictions, axis=1, keepdims=True)[0].ravel()
    
    def pre_process(self, features, predict_outcome=["winner"], 
                    use_dummies=True, use_scalar=True, 
                    filePath = "roundMoneyWinners2.csv",
                    labels_to_encode=['team_1', 'team_2', 't1_side', 't2_side'],
                    train_size=0.25):
        data = pd.read_csv("roundMoneyWinners2.csv", header=0, index_col=False)
        if use_dummies:
            data = pd.get_dummies(data, columns=["map"])
        X = data.loc[:, features]
        X = encode_labels(X, labels_to_encode)
        if use_scalar:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        X = np.array(X)
        y = data.loc[:, predict_outcome]
        y = np.array(y).ravel()
        y[y == 1] = 0
        y[y == 2] = 1
        return train_test_split(X, y, random_state = 0, train_size = train_size)
