"""
Evaluation and Metrics Module
Calculates standard biomedical diagnostic performance metrics from actual test sets:
- Accuracy
- Area Under the ROC Curve (AUC-ROC)
- Sensitivity (True Positive Rate / Recall)
- Specificity (True Negative Rate)
- Precision (Positive Predictive Value)
- F1-Score
- Confusion Matrix
- Multi-modality ROC curves overlay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_curve,
    classification_report
)

def calculate_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """
    Computes empirical classification metrics from actual test predictions.
    Strictly calculates numbers from input arrays without hardcoded defaults.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0)) # Sensitivity
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Specificity = TN / (TN + FP)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    auc = None
    fpr, tpr, thresholds = None, None, None
    if y_prob is not None:
        try:
            if len(y_prob.shape) == 2:
                prob_pos = y_prob[:, 1]
            else:
                prob_pos = y_prob
            auc = float(roc_auc_score(y_true, prob_pos))
            fpr, tpr, thresholds = roc_curve(y_true, prob_pos)
        except Exception:
            auc = None

    return {
        'accuracy': acc,
        'sensitivity': rec,
        'specificity': spec,
        'precision': prec,
        'f1_score': f1,
        'auc_roc': auc,
        'confusion_matrix': cm,
        'roc_curve': (fpr, tpr) if fpr is not None else None,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }

def plot_confusion_matrix_heatmap(cm: np.ndarray, title: str = "Confusion Matrix", class_names=['Control', 'ALS']):
    """
    Renders clean academic confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Biological Label',
        xlabel='Predicted Label',
        title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Text annotations in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "#222222",
                fontsize=12, fontweight='bold'
            )
    plt.tight_layout()
    return fig

def plot_multi_model_roc(roc_dict: dict):
    """
    Plots comparative ROC curves for EMG, Ultrasound, Early Fusion, and Late Fusion.
    roc_dict format: { 'Model Name': (fpr, tpr, auc_score) }
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (name, val) in enumerate(roc_dict.items()):
        if val is None or val[0] is None or val[1] is None:
            continue
        fpr, tpr, auc_val = val
        auc_str = f"AUC = {auc_val:.3f}" if auc_val is not None else "AUC = N/A"
        ax.plot(fpr, tpr, label=f"{name} ({auc_str})", color=colors[idx % len(colors)], linewidth=2.0)

    # Diagonal chance line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance (AUC = 0.500)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=10)
    ax.set_title('Comparative Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    return fig

def create_summary_metrics_dataframe(metrics_by_model: dict) -> pd.DataFrame:
    """
    Aggregates metrics for EMG, Ultrasound, Early Fusion, and Late Fusion into a clean summary table.
    """
    rows = []
    for model_name, m in metrics_by_model.items():
        if m is None:
            continue
        rows.append({
            'Model Configuration': model_name,
            'Accuracy': f"{m['accuracy']:.3f}",
            'Sensitivity (Recall)': f"{m['sensitivity']:.3f}",
            'Specificity': f"{m['specificity']:.3f}",
            'Precision': f"{m['precision']:.3f}",
            'F1-Score': f"{m['f1_score']:.3f}",
            'AUC-ROC': f"{m['auc_roc']:.3f}" if m['auc_roc'] is not None else "N/A"
        })
    return pd.DataFrame(rows)
