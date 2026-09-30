"""
Evaluation metrics for FireSightNet.
Includes the novel Fire Spread Index (FSI).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)


CLASSES = ['No_Fire', 'Smoke', 'Fire', 'Fire_Smoke']


def fire_spread_index(y_true, y_pred, classes=None):
    """
    Fire Spread Index (FSI) — Novel metric proposed in FireSightNet paper.

    Defined as the geometric mean of per-class recall values.
    FSI → 0 if any single class recall → 0, ensuring safety-critical reliability.

    Args:
        y_true: Ground truth labels (array-like)
        y_pred: Predicted labels (array-like)
        classes: List of class indices (default: all unique classes)

    Returns:
        fsi (float): FSI score in [0, 1]
        per_class_recall (dict): Recall per class
    """
    if classes is None:
        classes = np.unique(y_true)

    per_class_recall = {}
    recalls = []
    cm = confusion_matrix(y_true, y_pred)

    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_recall[CLASSES[i] if i < len(CLASSES) else str(cls)] = recall
        recalls.append(recall)

    # Geometric mean
    recalls = np.array(recalls)
    if np.any(recalls == 0):
        fsi = 0.0
    else:
        fsi = float(np.exp(np.mean(np.log(recalls))))

    return fsi * 100, per_class_recall


def compute_all_metrics(y_true, y_pred, y_prob=None):
    """
    Compute all evaluation metrics used in the FireSightNet paper.

    Returns:
        dict with accuracy, precision, recall, f1, fsi, mAP@0.5 (if y_prob given)
    """
    metrics = {
        'accuracy':  accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'recall':    recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
        'f1_score':  f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100,
    }

    fsi, per_class = fire_spread_index(y_true, y_pred)
    metrics['fsi'] = fsi
    metrics['per_class_recall'] = per_class

    if y_prob is not None:
        try:
            auc = roc_auc_score(
                np.eye(len(CLASSES))[y_true], y_prob,
                multi_class='ovr', average='weighted'
            )
            metrics['mAP_0_5'] = auc * 100
        except Exception:
            metrics['mAP_0_5'] = None

    return metrics


def print_metrics(metrics):
    print("\n" + "=" * 50)
    print("FireSightNet Evaluation Results")
    print("=" * 50)
    print(f"  Accuracy  : {metrics['accuracy']:.2f}%")
    print(f"  Precision : {metrics['precision']:.2f}%")
    print(f"  Recall    : {metrics['recall']:.2f}%")
    print(f"  F1-Score  : {metrics['f1_score']:.2f}%")
    print(f"  FSI       : {metrics['fsi']:.2f}%")
    if metrics.get('mAP_0_5'):
        print(f"  mAP@0.5   : {metrics['mAP_0_5']:.2f}%")
    print("\n  Per-Class Recall:")
    for cls, rec in metrics['per_class_recall'].items():
        print(f"    {cls}: {rec * 100:.2f}%")
    print("=" * 50)
