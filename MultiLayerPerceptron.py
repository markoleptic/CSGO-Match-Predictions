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
from utils import encode_labels
from sklearn.tree import export_text

#df80 = data.head(int(len(data)*0.8))
#df20 = data.tail(int(len(data)*0.2))

Features = [
    "round_number",
    "team_1",
    "team_2",
    "t1_side",
    "t2_side",
    "t1_money",
    "t2_money",
    "t1_rank",
    "t2_rank",
]

data = pd.read_csv('roundMoneyWinners2.csv',header=0, index_col=False)
#data = pd.get_dummies(data, columns = ['map'])
X = data.loc[:, Features]
y = data.loc[:,['winner']]
X = encode_labels(X, ['team_1', 'team_2', 't1_side', 't2_side'])
scaler = StandardScaler()
X = scaler.fit_transform(X)
# X = np.array(X)

# convert binary outcome to 0 or 1
y[y == 1] = 0
y[y == 2] = 1
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print(X_train)
print(y_train)

rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train,y_train)
predictions = rf.predict(X_test)
print(rf.get_params())
print("\n")
print(rf.score(X_test, y_test))
# print("\n")
# print(rf.decision_path(X_train[0]))
print("\n Classes: ")
print(rf.classes_)
print("\n feature_importances: ")
print(rf.feature_importances_)

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

print(export_text(rf.estimators_[0], 
                  spacing=3, decimals=3,
                  feature_names=Features))

fig = plt.figure(figsize=(15, 10))
plot_tree(rf.estimators_[0], 
          feature_names= Features,
          class_names=["Win","Lose"], 
          filled=True, impurity=True, 
          rounded=True)

svc_disp = RocCurveDisplay.from_estimator(rf, X_test, y_test)
plt.ylabel(ylabel="True Positive Rate")
plt.xlabel(xlabel="False Positive Rate")
plt.legend('', frameon=False)
plt.show()