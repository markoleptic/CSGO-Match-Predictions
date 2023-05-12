import numpy as np
import math
import pandas as pd
from utils import to_categorical,divide_on_feature, train_test_split, standardize, mean_squared_error, calculate_entropy, accuracy_score, calculate_variance
from sklearn.preprocessing import LabelEncoder

def calculate_entropy(y):
    """ Calculate the entropy of label array y """
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy

def mean_squared_error(y_true, y_pred):
    """ Returns the mean squared error between y_true and y_pred """
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse

def calculate_variance(X):
    """ Return the variance of the features in dataset X """
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
    
    return variance

def calculate_std_dev(X):
    """ Calculate the standard deviations of the features in dataset X """
    std_dev = np.sqrt(calculate_variance(X))
    return std_dev

def euclidean_distance(x1, x2):
    """ Calculates the l2 distance between two vectors """
    distance = 0
    # Squared distance between each coordinate
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)

def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def calculate_covariance_matrix(X, Y=None):
    """ Calculate the covariance matrix for the dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    return np.array(covariance_matrix, dtype=float)
 
def calculate_correlation_matrix(X, Y=None):
    """ Calculate the correlation matrix for the dataset X """
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance = (1 / n_samples) * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(Y), 1)
    correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_y.T))

    return np.array(correlation_matrix, dtype=float)

class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0

class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)

class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)

class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)

class TanH():
    def __call__(self, x):
        return 2 / (1 + np.exp(-2*x)) - 1

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)

class ReLU():
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)

class LeakyReLU():
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)

class ELU():
    def __init__(self, alpha=0.1):
        self.alpha = alpha 

    def __call__(self, x):
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, x):
        return np.where(x >= 0.0, 1, self.__call__(x) + self.alpha)

class SELU():
    # Reference : https://arxiv.org/abs/1706.02515,
    # https://github.com/bioinf-jku/SNNs/blob/master/SelfNormalizingNetworks_MLP_MNIST.ipynb
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946 

    def __call__(self, x):
        return self.scale * np.where(x >= 0.0, x, self.alpha*(np.exp(x)-1))

    def gradient(self, x):
        return self.scale * np.where(x >= 0.0, 1, self.alpha * np.exp(x))

class SoftPlus():
    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        return 1 / (1 + np.exp(-x))
    
class LogisticLoss():
    def __init__(self):
        sigmoid = Sigmoid()
        self.log_func = sigmoid
        self.log_grad = sigmoid.gradient

    def loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        p = self.log_func(y_pred)
        return y * np.log(p) + (1 - y) * np.log(1 - p)

    # gradient w.r.t y_pred
    def gradient(self, y, y_pred):
        p = self.log_func(y_pred)
        return -(y - p)

    # w.r.t y_pred
    def hess(self, y, y_pred):
        p = self.log_func(y_pred)
        return p * (1 - p)
N
class Perceptron():
    """The Perceptron. One layer neural network classifier.
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    activation_function: class
        The activation that shall be used for each neuron.
        Possible choices: Sigmoid, ExpLU, ReLU, LeakyReLU, SoftPlus, TanH
    loss: class
        The loss function used to assess the model's performance.
        Possible choices: SquareLoss, CrossEntropy
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, n_iterations=1000, activation_function=Sigmoid, loss=SquareLoss, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.loss = loss()
        self.activation_func = activation_function()
        self.W = None
        self.w0 = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = np.shape(X)
        _, n_outputs = np.shape(y)
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, n_outputs))
        self.w0 = np.zeros((1, n_outputs))
        for i in range(self.n_iterations):
            if i % 100 == 0:
                print(self.W)
            # Calculate outputs
            linear_output = X.dot(self.W) + self.w0
            y_pred = self.activation_func(linear_output)
            # Calculate the loss gradient w.r.t the input of the activation function
            error_gradient = self.loss.gradient(y, y_pred) * self.activation_func.gradient(linear_output)
            # Calculate the gradient of the loss with respect to each weight
            grad_wrt_w = X.T.dot(error_gradient)
            grad_wrt_w0 = np.sum(error_gradient, axis=0, keepdims=True)
            # Update weights
            self.W  -= self.learning_rate * grad_wrt_w
            self.w0 -= self.learning_rate  * grad_wrt_w0

    # Use the trained model to predict labels of X
    def predict(self, X):
        y_pred = self.activation_func(X.dot(self.W) + self.w0)
        return y_pred

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = pd.read_csv('roundMoneyWinners.csv',header=0, index_col=False)
data = data.head(1000)
#data = pd.get_dummies(data, columns = ['map'])
#features = ['t1_money', 't2_money', 't1_rank', 't2_rank', 'map_0', 'map_1',  'map_2',  'map_3',  'map_4',  'map_5',  'map_6',  'map_7',  'map_8',  'map_9']
#outcome = ['winner']
#featuresNoMap = ['t1_money', 't2_money', 't1_rank', 't2_rank']
##X = data.loc[:, features]
#print(X)
##y = data.loc[:, outcome]
#X = np.array(X)##
#X = scaler.fit_transform(X)

le = LabelEncoder()
#X['team_1'] = le.fit_transform(np.array(X.loc[:,['team_1']]).ravel())
#X['team_2'] = le.fit_transform(np.array(X.loc[:,['team_2']]).ravel())
#X['t1_side'] = le.fit_transform(np.array(X.loc[:,['t1_side']]).ravel())
#X['t2_side'] = le.fit_transform(np.array(X.loc[:,['t2_side']]).ravel())

#y = le.fit_transform(np.array(y))

tinyFeatures = [
    "t1_money",
    "t2_money",
]

matchWinOutcome = ["match_winner"]
outcome = ["winner"]

# split into features and outcomes
#X = data.loc[:, features]
X = data.loc[:, tinyFeatures]
#y = data.loc[:, matchWinOutcome]
y = data.loc[:, outcome]



scaler = StandardScaler()
X = scaler.fit_transform(X)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

perceptron = Perceptron(activation_function=Sigmoid,loss=SquareLoss)
perceptron.fit(X_train, y_train)
predictions = perceptron.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")