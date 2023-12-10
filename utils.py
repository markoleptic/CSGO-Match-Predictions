import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from matplotlib.pyplot import cm
import numpy as np

def encode_labels(X, labels):
    le = LabelEncoder()
    for label in labels:
        X[label] = le.fit_transform(np.array(X.loc[:,[label]]).ravel())
    return X

def calculate_FPR_TPR(X_test, y_test, predict_proba_func, binary_value_1 = -1, binary_value_2 = 1, step = 0.01):
    """
    Calculate FPR and TPR for different probability thresholds
    """
    tpr_list = []
    fpr_list = []
    for threshold in np.arange(0, 1, step):
        y_prob = predict_proba_func(X_test, threshold)
        y_pred = np.where(y_prob >= threshold, binary_value_2, binary_value_1)
        tp = np.sum((y_test == binary_value_2) & (y_pred == binary_value_2))
        tn = np.sum((y_test == binary_value_1) & (y_pred == binary_value_1))
        fp = np.sum((y_test == binary_value_1) & (y_pred == binary_value_2))
        fn = np.sum((y_test == binary_value_2) & (y_pred == binary_value_1))
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return fpr_list, tpr_list

def plot_ROC_curve(fpr_list, tpr_list):
    """
    plot ROC curve using values obtained from calculate_FPR_TPR
    """
    plt.plot(fpr_list, tpr_list)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title("ROC Curve")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

def plot_all_ROC_curve(tpr_lists, fpr_lists, algos):
    colors = cm.rainbow(np.linspace(0, 1, 6))
    for algo, fpr, tpr, color in zip(algos, fpr_lists, tpr_lists, colors):
        plt.plot(fpr, tpr, color = color, label=algo)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title("ROC Curve")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="upper left")
    plt.rcParams['figure.figsize'] = [8, 8]
    plt.show()

def plot_coefficents_chart(features, values):
    """
    Plots the weighting of the features / coefficients.\n
    example usage:
    weights = perceptron.weights
    np.abs(weights, out=weights)
    values = []
    values.append(np.sum(weights[0:1]) / 2)
    values.append(np.sum(weights[2:3]) / 2)
    values.append(np.sum(weights[4:5]) / 2)
    values.append(np.sum(weights[6:7]) / 2)
    values.append(np.sum(weights[8:17]) / (17-8))
    values = values / np.linalg.norm(values)
    features = ['Team Name','Side (CT/T)', 'Money', 'Team Rank', 'Map']
    plot_coefficents_curve(features, values) 
    """
    plt.bar(features, values, align='center', alpha=0.9)
    plt.title("Coefficients (combined) vs Feature Weight (normalized)")
    plt.ylabel(ylabel = "Feature Weight (normalized)")
    plt.xlabel(xlabel = "Coefficients (combined)")
    plt.legend('', frameon=False)
    plt.show()

def plot_all_coefficents_chart(features, algos, valuesArray):
    """
    Plots weighting of the features / coefficients for multiple algos
    """
    index = np.arange(features.__len__())
    bar_width = 1 / (algos.__len__() * 1.1)
    opacity = 0.8
    colors = cm.rainbow(np.linspace(0, 1, features.__len__()))
    for i, (algo, value, color) in enumerate(zip(algos, valuesArray, colors)):
        plt.bar((bar_width * i) + index, value, bar_width,
        alpha = opacity,
        color = color,
        label = algo,
        align = 'edge')
    plt.title("Coefficients (combined) vs Feature Weight (normalized)")
    plt.ylabel(ylabel = "Feature Weight (normalized)")
    plt.xlabel(xlabel = "Coefficients (combined)")
    plt.xticks(index + (bar_width * algos.__len__() / 2), features, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()