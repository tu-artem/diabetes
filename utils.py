import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score,
                             roc_auc_score,
                             roc_curve,
                             auc,
                             precision_score,
                             recall_score)


def evaluate_model(y_true, y_pred_prob, threshhold=0.5):
    accuracy = accuracy_score(y_true, y_pred_prob > threshhold)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    precision = precision_score(y_true, y_pred_prob > threshhold)
    recall = recall_score(y_true, y_pred_prob > threshhold)
    return({
        "accuracy": accuracy,
        "auc": auc_score,
        "precision": precision,
        "recall": recall
    })


def plot_curve(y_true, **predictions):
    """Plots ROC curve for 1 or more classifiers

    predictions: name_of_model=[list of predicted probabilities]
    """
    plt.figure()
    lw = 2
    for name, prediction in predictions.items():
        fpr, tpr, _ = roc_curve(y_true, prediction)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, label=f'{name} (area = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
