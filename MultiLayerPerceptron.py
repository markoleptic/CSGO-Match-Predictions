import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Hinge
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

#df80 = data.head(int(len(data)*0.8))
#df20 = data.tail(int(len(data)*0.2))
data = pd.read_csv('roundMoneyWinners2.csv',header=0, index_col=False)
#print(data)
#data = pd.get_dummies(data, columns = ['map'])
#features = ['t1_money', 't2_money', 't1_rank', 't2_rank', 'map_0', 'map_1',  'map_2',  'map_3',  'map_4',  'map_5',  'map_6',  'map_7',  'map_8',  'map_9']
features = ['team_1', 'team_2', 't1_money', 't2_money', 't1_rank', 't2_rank', 'map_0', 'map_1',  
            'map_2',  'map_3',  'map_4',  'map_5',  'map_6',  'map_7',  'map_8',  'map_9']
features2 = ['team_1', 'team_2', 't1_money', 't2_money', 't1_rank', 't2_rank', 'map_0', 'map_1',
            'map_2',  'map_3',  'map_4',  'map_5',  'map_6',  'map_7',  'map_8',  'map_9', 'winner']

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

newFeatures1map = [
    "team_1",
    "team_2",
    "t1_side",
    "t2_side",
    "t1_money",
    "t2_money",
    "t1_rank",
    "t2_rank",
    "map",
]

newFeatures1mapnosideorteam = [
    "t1_money",
    "t2_money",
    "t1_rank",
    "t2_rank",
    "map",
]

def encodeLabel(X, labels):
    le = LabelEncoder()
    for label in labels:
        X[label] = le.fit_transform(np.array(X.loc[:,[label]]).ravel())
    return X

X = data.loc[:, newFeatures]
y = data.loc[:,['winner']]

labelsToEncode = ['team_1', 'team_2,', 't1_side', 't2_side']
X = encodeLabel(X, labelsToEncode)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = np.array(X)
y = le.fit_transform(np.array(y).ravel())

# convert binary outcome to 0 or 1
y[y == 1] = 0
y[y == 2] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

mlp_clf = MLPClassifier(alpha=1,hidden_layer_sizes=(1,))
mlp_clf.fit(X_train,y_train)
predictions = mlp_clf.predict(X_test)
print(mlp_clf.get_params())
print(mlp_clf.score(X_train, y_train))
print(mlp_clf.coefs_)
print(classification_report(y_test, predictions))
svc_disp = RocCurveDisplay.from_estimator(mlp_clf, X_test, y_test)
plt.ylabel(ylabel="True Positive Rate")
plt.xlabel(xlabel="False Positive Rate")
plt.legend('', frameon=False)
plt.show()
values = []
coef = mlp_clf.coefs_[0]
values.append((abs(coef[0])+abs(coef[1])) / 2)
values.append((abs(coef[2])+abs(coef[3])) / 2)
values.append((abs(coef[4])+abs(coef[5])) / 2)
values.append((abs(coef[6])+abs(coef[7])) / 2)
values.append((np.mean(abs(coef[8:17]))/(17-8)))

svc_disp = RocCurveDisplay.from_estimator(mlp_clf, X_test, y_test)
plt.ylabel(ylabel="True Positive Rate")
plt.xlabel(xlabel="False Positive Rate")
plt.legend('', frameon=False)
plt.show()
features = ['team','side', 'money', 'rank', 'map']
plt.scatter(features, values)
plt.show()
exit()