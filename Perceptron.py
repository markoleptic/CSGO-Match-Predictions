import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, learning_rate=0.001, num_iterations=1000, batch_size=5):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.weights = None
        self.bias = 1.0

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        # gradient descent
        for i in range(self.num_iterations):
            batch_indices = np.random.choice(n_samples, self.batch_size, replace=False)
            X_batch = X[batch_indices]
            y_batch = np.squeeze(y[batch_indices])

            # calculate the predicted values
            y_predicted = self.predict(X_batch)

            # calculate the gradients
            dw, db = self.compute_gradients(X_batch, y_batch, y_predicted)

            # update the weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self.weights

    def binary_cross_entropy_loss(self, X, y, weights, bias):
        loss = 0
        for i in range(X.shape[0]):
            linear_output = np.dot(X[i], weights) + bias
            sigmoid_output = 1 / (1 + np.exp(-linear_output))
            current_loss = -(y[i] * np.log(sigmoid_output) + (1 - y[i]) * np.log(1 - sigmoid_output))
            loss += current_loss
        return np.mean(loss / X.shape[0])

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)
    
    def predict_proba(self, X, threshold=0):
        linear_output = np.dot(X, self.weights) + self.bias
        probabilities = 1 / (1 + np.exp(-linear_output))
        return np.where(probabilities >= threshold, 1, -1)

    def compute_gradients(self, X, y, y_predicted):
        dw = (1 / X.shape[0]) * np.dot(X.T, y_predicted - y)
        db = (1 / X.shape[0]) * np.sum(y_predicted - y)
        return dw, db

    def compute_loss(self, y_predicted, y_batch):
        eps = 1e-15
        return np.mean(
            -y_batch * np.log(y_predicted + eps)
            - (1 - y_batch) * np.log(1 - y_predicted + eps)
        )

    def compute_hinge_loss(self, X, y):
        loss = 0
        for i in range(X.shape[0]):
            linear_output = np.dot(X[i], self.weights) + self.bias
            hinge_loss = max(0, 1 - y[i] * linear_output)
            loss += hinge_loss
        print(np.mean(loss / X.shape[0]))
        return np.mean(loss / X.shape[0])

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
matchWinOutcome = ["match_winner"]
outcome = ["winner"]
featuresNoMap = ["t1_money", "t2_money", "t1_rank", "t2_rank"]

data = pd.read_csv("roundMoneyWinners2.csv", header=0, index_col=False)
data = pd.get_dummies(data, columns=["map"])
print(data)

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
y[y == 1] = 1
y[y == 2] = -1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# train the model
perceptron = Perceptron()
weights = perceptron.fit(X_train, y_train)
print(f"Weights: {weights}")

# use the trained model to make predictions
predictions = perceptron.predict(X_test)
print(predictions)

count = np.count_nonzero(predictions == -1)
print(count, predictions.__len__())
y_pred = perceptron.predict(X_test)

# calculate FPR and TPR for different probability thresholds
tpr_list = []
fpr_list = []
for threshold in np.arange(0, 1, 0.01):
    y_prob = perceptron.predict_proba(X_test, threshold)
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
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")