import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=10000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize parameters
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent algorithm
        for i in range(self.num_iterations):
            # Compute predictions
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_cls
    def predict_proba(self, X, threshold=0):
        linear_output = np.dot(X, self.weights) + self.bias
        probabilities = 1 / (1 + np.exp(-linear_output))
        return np.where(probabilities >= threshold, 1, -1)
    
    def accuracy(self, y_true, y_pred):
        accuracy = np.mean(y_true == y_pred)
        return accuracy

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def pre_process(self, features, outcome = ["winner"]):
        data = pd.read_csv("roundMoneyWinners2.csv", header=0, index_col=False)
        data = pd.get_dummies(data, columns=["map"])

        X = data.loc[:, features]
        y = data.loc[:, outcome]

        le = LabelEncoder()
        X["team_1"] = le.fit_transform(np.array(X.loc[:, ["team_1"]]).ravel())
        X["team_2"] = le.fit_transform(np.array(X.loc[:, ["team_2"]]).ravel())
        X["t1_side"] = le.fit_transform(np.array(X.loc[:, ["t1_side"]]).ravel())
        X["t2_side"] = le.fit_transform(np.array(X.loc[:, ["t2_side"]]).ravel())

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X = np.array(X)
        y = np.array(y).ravel()

        # convert binary outcome to -1 or 1
        y[y == 1] = 0
        y[y == 2] = 1

        #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        return train_test_split(X, y, random_state=0)