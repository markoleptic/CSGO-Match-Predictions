import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import matplotlib.pyplot as plt
from utils import encode_labels

class DecisionTree:
    """
    Binary decision tree using the Gini impurity as the splitting
    criterion.
    """
    def __init__(self, max_depth=None, min_samples_leaf=None, verbose=True):
        self.verbose = verbose
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.feature_labels = []

    def fit(self, X, y, feature_labels):
        self.feature_labels = feature_labels
        if self.verbose:
            print(f'Fitting tree with {X.shape[1]} features and {X.shape[0]} samples.')
        self.tree = self.build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self.predict_single(x, self.tree) for x in X])
    
    def predict_single(self, x, node):
        """
        Returns binary classification if node is a leaf node, 
        otherwise it recursively calls itself until it finds 
        a leaf node.
        """
        if node.is_leaf:
            return np.argmax(node.proba)
        if x[node.feature] <= node.threshold:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)
        
    def predict_proba(self, X):
        """
        Returns a list of of probabilities at a given node, with probability of 0 being
        the first index and 1 being the second index.
        """
        probas = np.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            probas[i] = self.predict_proba_single(self.tree, X[i])
        return probas
    
    def predict_proba_single(self, node, x):
        """
        Returns the two probabilities if node is a leaf node, 
        otherwise it recursively calls itself until it finds 
        a leaf node.
        """
        if node.is_leaf:
            return node.proba
        if x[node.feature] <= node.threshold:
            return self.predict_proba_single(node.left, x)
        else:
            return self.predict_proba_single(node.right, x)
    
    def build_tree(self, X, y, depth):
        """
        Find the best split by iterating over all the features. 
        Create a new node with the best split, and recursively 
        call build_tree on the split until all are leaf nodes.
        Return the node containing the split feature, value, and
        the left & right child nodes.
        """
        n_samples, n_features = X.shape
        n_class0 = np.sum(y == 0)
        n_class1 = np.sum(y == 1)

        if depth == self.max_depth or n_samples < self.min_samples_leaf or np.unique(y).size == 1:
            return Node(is_leaf=True, value=max(n_class1, n_class0)/n_samples, proba=[n_class0/n_samples, n_class1/n_samples])

        best_feature_threshold = self.get_best_split(X, y)
        if best_feature_threshold is None:
            return Node(is_leaf=True, value=max(n_class1, n_class0)/n_samples, proba=[n_class0/n_samples, n_class1/n_samples])
        
        best_feature, best_threshold = best_feature_threshold

        if best_feature is None or best_threshold is None:
            return Node(is_leaf=True, value=max(n_class1, n_class0)/n_samples, proba=[n_class0/n_samples, n_class1/n_samples])

        # seperate the tree based on the thresholds of the best feature
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_tree = self.build_tree(X[left_indices, :], y[left_indices], depth+1)
        right_tree = self.build_tree(X[right_indices, :], y[right_indices], depth+1)

        if depth < 1:
            print("Best Feature at Root Node: ", self.feature_labels[best_feature])
        return Node(is_leaf=False, value=None, proba=None, feature=best_feature, feature_label=self.feature_labels[best_feature], threshold=best_threshold,
                    left=left_tree, right=right_tree)

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
        parent_gini = self.gini(y)
    
        # calculate weighted average of the gini impurity of left and right children
        left_gini = self.gini(y[left_idx])
        right_gini = self.gini(y[right_idx])
        left_size = len(y[left_idx])
        right_size = len(y[right_idx])
        total_size = left_size + right_size
        weighted_avg_gini = (left_size / total_size) * left_gini + (right_size / total_size) * right_gini
    
        # calculate gini impurity gain
        return parent_gini - weighted_avg_gini
    
class Node:
    def __init__(self, is_leaf, value, proba, feature=None, threshold=None, left=None, right=None, feature_label=None):
        self.is_leaf = is_leaf
        self.feature = feature
        self.feature_label = feature_label
        self.threshold = threshold
        self.left = left
        self.right = right
        # two element list containing probability of 0/1
        self.proba = proba
        self.value = value
    def count_nodes(self, root):
        if root is None:
            return 0
        return 1 + self.count_nodes(root.left) + self.count_nodes(root.right)

class RandomForest:
    def __init__(self, n_estimators=10, max_features ='sqrt', max_features_num = 0, max_depth=None, min_samples_split=2, min_samples_leaf=1, verbose=True):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_features_num = max_features_num
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.n_nodes_list = []
        self.verbose = verbose
        self.feature_labels = []
        
    def fit(self, X, y):
        for i in range(self.n_estimators):
            tree = DecisionTree(self.max_depth, self.min_samples_leaf, self.verbose)

            # Randomly select a subset of features for each tree
            indices = self.get_indices(X)

            X_bootstrap, y_bootstrap = self.bootstrap_sampling(X, y)
            X_bootstrap = X_bootstrap[:, indices]
            tree.fit(X_bootstrap, y_bootstrap, [self.feature_labels[i] for i in indices])
            self.trees.append(tree)
            self.n_nodes_list.append(tree.tree.count_nodes(tree.tree))
            if self.verbose:
               print(f'Tree {i} has {self.get_num_nodes(tree.tree)} nodes.')


    def bootstrap_sampling(self, X, y):
        """
        Create a new dataset by randomly selecting n samples with
        replacement from the original dataset, where n is the number
        of samples in the original dataset.
        """
        indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        return X[indices], y[indices]
    
    def get_indices(self, X):
        # Randomly select a subset of features for each tree
        indices = np.random.choice(X.shape[1], size=self.max_features_num, replace=False)
        if self.max_features == 'sqrt':
            indices = np.random.choice(X.shape[1], size=int(np.sqrt(X.shape[1])), replace=False)
        elif self.max_features == 'log2':
            indices = np.random.choice(X.shape[1], size=int(np.log2(X.shape[1])), replace=False)
        return indices

    def predict_proba(self, X):
        predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            indices = self.get_indices(X)
            predictions[:, i] = tree.predict_proba(X[:, indices])[:, 1] 
        avg_predictions = np.mean(predictions, axis=1)
        binary_predictions = np.round(avg_predictions).astype(int)
        return binary_predictions

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            indices = self.get_indices(X)
            predictions[:, i] = tree.predict(X[:, indices])
        avg_predictions = np.mean(predictions, axis=1)
        binary_predictions = np.round(avg_predictions).astype(int)
        return binary_predictions
    
    def get_num_nodes(self, node):
        """
        Recursively count the number of nodes in the tree starting from the given node.
        """
        if node is None:
            return 0
        return 1 + self.get_num_nodes(node.left) + self.get_num_nodes(node.right)

    def pre_process(self, features, predict_outcome=["winner"], 
                    use_dummies=True, use_scalar=True, 
                    filePath = "roundMoneyWinners2.csv",
                    labels_to_encode=['team_1', 'team_2', 't1_side', 't2_side'],
                    train_size=0.25):
        self.feature_labels = features
        data = pd.read_csv("roundMoneyWinners2.csv", header=0, index_col=False)
        data = data.sample(30000)
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


