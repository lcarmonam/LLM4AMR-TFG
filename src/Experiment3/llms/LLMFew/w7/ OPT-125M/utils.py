import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, f1_score


####################################################################
# FUNCTIONS TO ANALYZE THE RESULTS (SCORES) AND PROBABILITIES #
####################################################################

def get_metrics_(y_true, y_pred_probs):
    y_pred = np.round(y_pred_probs).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) 
    sensitivity = tp / (tp + fn)  # Changed from Recall to Sensitivity
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_probs) 
    f1 = f1_score(y_true, y_pred)  # Added F1-score

    metrics_df = pd.DataFrame({
        'TN': [tn],
        'FP': [fp],
        'FN': [fn],
        'TP': [tp],
        'Accuracy': [accuracy],
        'Sensitivity': [sensitivity],  # Updated label
        'Specificity': [specificity],
        'ROC AUC': [roc_auc],
        'F1 Score': [f1]  # Added F1-score
    })

    return metrics_df

def plot_metrics(metrics_df):
    """
    Plot bar chart for Specificity, Sensitivity, ROC AUC, and F1-score.
    
    Args:
        - metrics_df: DataFrame containing calculated metrics.
    """
    plt.figure(figsize=(12, 6))
    
    # Extract metrics for plotting
    specificity = metrics_df['Specificity'].values[0]
    sensitivity = metrics_df['Sensitivity'].values[0]
    roc_auc = metrics_df['ROC AUC'].values[0]
    f1 = metrics_df['F1 Score'].values[0]

    # Plot Specificity, Sensitivity, ROC AUC, and F1-score
    plt.bar(['Specificity', 'Sensitivity', 'ROC AUC', 'F1 Score'], [specificity, sensitivity, roc_auc, f1])
    plt.ylim(0, 1)
    plt.title('Metrics for Single-Output Predictions')
    plt.ylabel('Score')
    plt.grid(axis='y')
    plt.show()

def plot_roc_curve(y_test, y_pred_probs):
    """
    Plot ROC curve for single-output predictions.
    
    Args:
        - y_test: Array of true labels (1D).
        - y_pred_probs: Array of predicted probabilities (1D).
    """
    # Ensure y_test and y_pred_probs are 1D arrays
    y_test = np.array(y_test).flatten()
    y_pred_probs = np.array(y_pred_probs).flatten()

    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    
    plt.figure(figsize=(12, 6))
    plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(roc_auc_score(y_test, y_pred_probs)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
