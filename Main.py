from sklearn.metrics import accuracy_score
from utils import calculate_FPR_TPR, plot_ROC_curve, plot_coefficents_curve, plot_All_coefficents_curve
from RandomForest import RandomForest
from Perceptron import Perceptron
import numpy as np
import math

Features = [
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

# ----------- #
# Random Forest
# ----------- #

forest = RandomForest()
X_train_rf, X_test_rf, y_train_rf, y_test_rf = forest.pre_process(Features)
forest.fit(X_train_rf, y_train_rf)

# use the trained model to make predictions
predictions_rf = forest.predict(X_test_rf)
print(predictions_rf, y_test_rf)
accuracy = accuracy_score(y_test_rf, predictions_rf)

print(f"Accuracy: {accuracy}")
count = np.count_nonzero(predictions_rf == -1)
print(count, predictions_rf.__len__())


# --------- #
# Perceptron
# --------- #

perceptron = Perceptron()
X_train_p, X_test_p, y_train_p, y_test_p = perceptron.pre_process(Features)
perceptron.fit(X_train_p, y_train_p)

# use the trained model to make predictions
predictions_p = perceptron.predict(X_test_p)
accuracy = accuracy_score(y_test_p, predictions_p)
print(predictions_p)
print(f"Accuracy: {accuracy}")

# TPR, FPR, ROC
fpr_list, tpr_list = calculate_FPR_TPR(X_test_p, y_test_p, perceptron.predict_proba)
plot_ROC_curve(fpr_list, tpr_list)

# weights chart
weights = perceptron.weights
np.abs(weights, out=weights)
values = []
values.append(np.sum(weights[0:1]) / 2)
values.append(np.sum(weights[2:3]) / 2)
values.append(np.sum(weights[4:5]) / 2)
values.append(np.sum(weights[6:7]) / 2)
values.append(np.sum(weights[8:17]) / (17-8))
print(values)
values = values / np.linalg.norm(values)
features = ['Team Name','Side (CT/T)', 'Money', 'Team Rank', 'Map']
plot_coefficents_curve(features, values)