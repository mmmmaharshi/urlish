"""Threshold selection and the metrics that actually matter for a filter.

Accuracy is not one of them. A URL filter that blocks 1 in 20 legitimate
visits is unusable, so everything is reported at a fixed false-positive rate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def threshold_at_fpr(scores: ArrayLike, y: ArrayLike, target: float = 0.01) -> float:
    """Highest threshold whose false-positive rate on this set is <= target."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y)
    negatives = np.sort(scores[y == 0])
    if negatives.size == 0:
        return float(np.nextafter(scores.max() if scores.size else 0.0, np.inf))
    allowed = int(np.floor(target * negatives.size))
    if allowed <= 0:
        return float(np.nextafter(negatives[-1], np.inf))
    idx = negatives.size - allowed - 1
    return float(np.nextafter(negatives[idx], np.inf))


def at_threshold(scores: ArrayLike, y: ArrayLike, threshold: float) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y)
    pred = scores >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    if len(np.unique(y)) == 2:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, scores))
    else:
        auc = float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "accuracy": accuracy,
        "auc": auc,
    }
