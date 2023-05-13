import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class NaiveBayes:
    def __init__(self):
        self.class_priors = None
        self.class_means = None
        self.class_variances = None

    def fit(self, X, y):
        # Get the unique class labels and their indices
        classes, class_indices = np.unique(y, return_inverse=True)
        n_classes = len(classes)

        # Compute the class priors
        self.class_priors = np.bincount(class_indices) / len(y)

        # Compute the class means and variances
        self.class_means = np.zeros((n_classes, X.shape[1]))
        self.class_variances = np.zeros((n_classes, X.shape[1]))
        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_means[i, :] = np.mean(X_c, axis=0)
            self.class_variances[i, :] = np.var(X_c, axis=0)

    def gaussian_pd(self, X, mean, variance):
        # Compute the Gaussian probability density function for each feature
        return np.exp(-(X - mean)**2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)
    
    def predict(self, X):
        # Compute the log-likelihoods
        log_likelihoods = []
        for i in range(len(self.class_priors)):
            prior = np.log(self.class_priors[i])
            self.class_variances[i] += 1e-9
            likelihood = np.sum(np.log(self.gaussian_pd(X, self.class_means[i], self.class_variances[i])), axis=1)
            log_likelihoods.append(prior + likelihood)
        log_likelihoods = np.array(log_likelihoods).T

        # Return the class with the highest log-likelihood for each sample
        return np.argmax(log_likelihoods, axis=1)

    def accuracy(self, y_true, y_pred):
        accuracy = np.mean(y_true == y_pred)
        return accuracy
    
    def pre_process(self, features, outcome = ["winner"]):
        # data preprocessing
        data = pd.read_csv("roundMoneyWinners2.csv", header=0, index_col=False)
        data = pd.get_dummies(data, columns=["map"])
        print(data)

        X = data.loc[:, features]
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

        return train_test_split(X, y, random_state=0)