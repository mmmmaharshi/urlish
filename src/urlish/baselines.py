"""Published-style baselines measured on the same splits as the benchmark.

These are URL-only detectors in the spirit of URLNet and URLTran: they see
the raw URL string and nothing else. They run on scikit-learn so the
benchmark stays dependency-light; faithful CNN/transformer reproductions
need PyTorch or TensorFlow and are out of scope here.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import metrics
from .data import Row, random_split
from .report import Result


def run_char_ngram(
    rows: list[Row],
    y: np.ndarray,
    train_idx: list[int],
    test_idx: list[int],
) -> Result:
    """Character n-gram logistic regression, calibrated like every other row."""
    t0 = time.time()
    urls = [rows[i].url for i in range(len(rows))]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=5000)
    X = vectorizer.fit_transform(urls)
    X = np.asarray(X.todense(), dtype=np.float32)

    # random split of the training half into fit and calibration
    rng_idx = list(train_idx)
    import random as _r

    _r.Random(1).shuffle(rng_idx)
    cut = int(len(rng_idx) * 0.25)
    cal_idx = sorted(rng_idx[:cut])
    fit_idx = sorted(rng_idx[cut:])
    model = LogisticRegression(max_iter=200, class_weight="balanced")
    model.fit(X[fit_idx], y[fit_idx])
    threshold = metrics.threshold_at_fpr(
        model.predict_proba(X[cal_idx])[:, 1], y[cal_idx], metrics.TARGET_FPR
    )
    elapsed = time.time() - t0
    print(f"  char n-gram baseline trained in {elapsed:.1f}s", file=sys.stderr)

    metrics_out = metrics.at_threshold(
        model.predict_proba(X[test_idx])[:, 1], y[test_idx], threshold
    )
    return Result.measured(
        "char n-gram logistic regression",
        len(fit_idx),
        len(test_idx),
        float(y[test_idx].mean()),
        metrics_out,
    )
