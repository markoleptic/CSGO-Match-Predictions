import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import calculate_FPR_TPR, plot_ROC_curve, encode_labels, plot_coefficents_curve, plot_All_coefficents_curve

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
    
    def pre_process(self, features, predict_outcome = ["winner"], 
                    use_dummies = True, use_scalar = True, 
                    filePath = "roundMoneyWinners2.csv",
                    labels_to_encode = ['team_1', 'team_2', 't1_side', 't2_side'],
                    train_size = 0.25):
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
        y[y == 1] = -1
        y[y == 2] = 1
        return train_test_split(X, y, random_state = 0, train_size = train_size)
