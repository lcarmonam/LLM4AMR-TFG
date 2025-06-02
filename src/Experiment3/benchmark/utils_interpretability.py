from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

####################################################################
# FUNCTIONS TO ANALYZE THE RESULTS (SCORES) AND PROBABILITIES #
####################################################################
def get_metrics(y_test_df, y_pred_df):
    """
    Calculate metrics for single-output predictions.
    
    Args:
        - y_test_df: DataFrame containing the real values (1D).
        - y_pred_df: DataFrame containing the predicted probabilities (1D).
        - threshold: Threshold to convert probabilities to binary predictions.
    
    Returns:
        - metrics_df: DataFrame containing the calculated metrics.
    """
    # Extract the true labels and predicted probabilities
    y_test = y_test_df.iloc[:, 0].values
    y_pred_probs = y_pred_df.iloc[:, 0].values

    # Round the predictions to get binary values
    y_pred_binary = np.round(y_pred_probs).astype(int)

    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary, labels=[0, 1]).ravel()

    # Calculate specificity and recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_probs) if len(np.unique(y_test)) > 1 else np.nan

    # Create a DataFrame to store the metrics
    metrics_df = pd.DataFrame({
        'TN': [tn],
        'FP': [fp],
        'FN': [fn],
        'TP': [tp],
        'Specificity': [specificity],
        'Recall': [recall],
        'ROC AUC': [roc_auc]
    })

    return metrics_df

def plot_metrics(metrics_df):
    """
    Plot metrics for single-output predictions.
    
    Args:
        - metrics_df: DataFrame containing the calculated metrics.
    """
    plt.figure(figsize=(12, 6))
    
    # Extract metrics for plotting
    specificity = metrics_df['Specificity'][0]
    recall = metrics_df['Recall'][0]
    roc_auc = metrics_df['ROC AUC'][0]

    # Plot Specificity, Recall and ROC AUC
    plt.bar(['Specificity', 'Recall', 'ROC AUC'], [specificity, recall, roc_auc])
    plt.ylim(0, 1)
    plt.title('Metrics for Single-Output Predictions')
    plt.ylabel('Score')
    plt.grid(axis='y')
    plt.show()

def plot_roc_curve(y_test_df, y_pred_df):
    """
    Plot ROC curve for single-output predictions.
    
    Args:
        - y_test_df: DataFrame containing the real values (1D).
        - y_pred_df: DataFrame containing the predicted probabilities (1D).
    """
    y_test = y_test_df.iloc[:, 0].values
    y_pred_probs = y_pred_df.iloc[:, 0].values

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    
    plt.figure(figsize=(12, 6))
    plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc_score(y_test, y_pred_probs)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()