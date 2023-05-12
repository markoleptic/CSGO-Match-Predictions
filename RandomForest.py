import numpy as np
from scipy.stats import mode
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class DecisionTree:
    def __init__(self, max_depth=2, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.tree = self.build_tree(X, y, depth=0)
        
    def predict(self, X):
        return np.array([self.predict_one_sample(x, self.tree) for x in X])
    
    def predict_one_sample(self, x, node):
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
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        
    def fit(self, X, y):
        for i in range(self.n_estimators):
            print(i)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
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
    

# data preprocessing
features = [
    "t1_money",
    "t2_money",
    "t1_rank",
    "t2_rank",
    "map_0",
    "map_1",
    "map_2",
    "map_3",
    "map_4",
    "map_5",
    "map_6",
    "map_7",
    "map_8",
    "map_9",
]
newFeatures = [
    "team_1",
    "team_2",
    "t1_side",
    "t2_side",
    "t1_money",
    "t2_money",
    "t1_rank",
    "t2_rank",
    "map_0",
    "map_1",
    "map_2",
    "map_3",
    "map_4",
    "map_5",
    "map_6",
    "map_7",
    "map_8",
    "map_9",
]

featuresNoMap = [
    "t1_money", 
    "t2_money", 
    "t1_rank", 
    "t2_rank",
]

tinyFeatures = [
    "t1_money",
    "t2_money",
]

matchWinOutcome = ["match_winner"]
outcome = ["winner"]

data = pd.read_csv("roundMoneyWinners2.csv", header=0, index_col=False)
data = pd.get_dummies(data, columns=["map"])

# split into features and outcomes
#X = data.loc[:, features]
X = data.loc[:, newFeatures]
#y = data.loc[:, matchWinOutcome]
y = data.loc[:, outcome]

le = LabelEncoder()
X['team_1'] = le.fit_transform(np.array(X.loc[:,['team_1']]).ravel())
X['team_2'] = le.fit_transform(np.array(X.loc[:,['team_2']]).ravel())
X['t1_side'] = le.fit_transform(np.array(X.loc[:,['t1_side']]).ravel())
X['t2_side'] = le.fit_transform(np.array(X.loc[:,['t2_side']]).ravel())

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = np.array(X)
y = np.array(y).ravel()

# convert binary outcome to -1 or 1
y[y == 1] = 0
y[y == 2] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# train the model
forest = RandomForest()
forest.fit(X_train, y_train)

# use the trained model to make predictions
predictions = forest.predict(X_test)
print(predictions, y_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

count = np.count_nonzero(predictions == -1)
print(count, predictions.__len__())
y_pred = forest.predict(X_test)

exit()
# calculate FPR and TPR for different probability thresholds
tpr_list = []
fpr_list = []
for threshold in np.arange(0, 1, 0.01):
    y_prob = forest.predict_proba(X_test, threshold)
    y_pred = np.where(y_prob >= threshold, 1, -1)
    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == -1) & (y_pred == -1))
    fp = np.sum((y_test == -1) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == -1))
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    tpr_list.append(tpr)
    fpr_list.append(fpr)

# plot the ROC curve
plt.plot(fpr_list, tpr_list)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# print accuracy of predictions


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train,y_train)
predictions = rf_clf.predict(X_test)
print(rf_clf.get_params())
print(rf_clf.score(X_test, y_test))
print(classification_report(y_test, predictions))

svc_disp = RocCurveDisplay.from_estimator(rf_clf, X_test, y_test)
plt.ylabel(ylabel="True Positive Rate")
plt.xlabel(xlabel="False Positive Rate")
plt.legend('', frameon=False)
plt.show()
values = []
coef = rf_clf.coefs_[0]
values.append((abs(coef[0])+abs(coef[1])) / 2)
values.append((abs(coef[2])+abs(coef[3])) / 2)
values.append((abs(coef[4])+abs(coef[5])) / 2)
values.append((abs(coef[6])+abs(coef[7])) / 2)
values.append((np.mean(abs(coef[8:17]))/(17-8)))
exit()
