from sklearn.metrics import accuracy_score
from utils import calculate_FPR_TPR, plot_ROC_curve, plot_coefficents_chart, plot_All_coefficents_chart
from RandomForest import RandomForest
from Perceptron import Perceptron
import numpy as np
import math

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
    "map"
]

# ----------- #
# Random Forest
# ----------- #

forest = RandomForest(n_estimators = 10, max_features = "number", max_features_num = 3, max_depth = 10, min_samples_split = 2, min_samples_leaf = 1)

X_train, X_test, y_train, y_test = forest.pre_process(Features, use_scalar=False)

forest.fit(X_train, y_train)

# use the trained model to make predictions
predictions_rf = forest.predict(X_test)
print(predictions_rf)
print(predictions_rf, y_test)
accuracy = accuracy_score(y_test, predictions_rf)

print(f"Accuracy: {accuracy}")
count = np.count_nonzero(predictions_rf == 1)
print(count, predictions_rf.__len__())

exit()
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
plot_coefficents_chart(features, values)