"""Threshold selection and the metrics that actually matter for a filter.

Accuracy is not one of them. A URL filter that blocks 1 in 20 legitimate
visits is unusable, so everything is reported at a fixed false-positive rate.
"""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike


class Metrics(TypedDict):
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    fpr: float
    auc: float


TARGET_FPR = 0.01


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval - honest error bars on the 300-URL test set."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


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


def at_threshold(scores: ArrayLike, y: ArrayLike, threshold: float) -> Metrics:
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
        "auc": auc,
    }
