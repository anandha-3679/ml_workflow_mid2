"""Clinical evaluation diagnostics, metrics, and threshold tuning."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def evaluate_clinical_predictions(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute comprehensive clinical metrics, prioritizing sensitivity/recall."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Clinical metrics
    sensitivity = float(recall_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "accuracy": acc,
        "roc_auc": auc,
        "brier_score": brier,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


def find_optimal_clinical_threshold(
    pipeline: Pipeline,
    x_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    min_sensitivity: float = 0.85,
) -> float:
    """Find the threshold maximizing specificity while maintaining min sensitivity."""
    y_prob = pipeline.predict_proba(x_val)[:, 1]
    candidate_thresholds = np.linspace(0.1, 0.9, 81)

    best_threshold = 0.5
    best_specificity = -1.0

    for thresh in candidate_thresholds:
        metrics = evaluate_clinical_predictions(y_val, y_prob, threshold=thresh)
        if metrics["sensitivity"] >= min_sensitivity:
            if metrics["specificity"] > best_specificity:
                best_specificity = metrics["specificity"]
                best_threshold = float(thresh)

    return best_threshold
