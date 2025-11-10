
"""Model evaluation utilities (ROC, PR, metrics)."""
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_pred_proba))
    }
    return metrics
