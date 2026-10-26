"""Published-style baselines measured on the same splits as the benchmark.

These are URL-only detectors in the spirit of URLNet and URLTran: they see
the raw URL string and nothing else. They run on scikit-learn (and optionally
PyTorch) so the benchmark stays dependency-light.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import metrics
from .data import Row
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
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=1000)
    X = vectorizer.fit_transform(urls)
    X = np.asarray(X.todense(), dtype=np.float32)

    fit_idx, cal_idx = _split_fit_cal(rows, train_idx)
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


def run_transformer(
    rows: list[Row],
    y: np.ndarray,
    train_idx: list[int],
    test_idx: list[int],
) -> Result | None:
    """Small character transformer (URLTran-style). Returns None if torch is missing."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  transformer baseline skipped: torch not installed", file=sys.stderr)
        return None

    t0 = time.time()
    urls = [rows[i].url for i in range(len(rows))]
    # character vocabulary
    chars = sorted(set("".join(urls)))
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 = pad
    max_len = 128

    def encode(s: str) -> list[int]:
        return [char_to_idx.get(c, 0) for c in s[:max_len]] + [0] * (max_len - min(len(s), max_len))

    X = np.array([encode(u) for u in urls], dtype=np.int64)
    Xte = torch.tensor(X[test_idx], dtype=torch.long)

    class TinyTransformer(nn.Module):
        def __init__(self, vocab: int, dim: int = 64, heads: int = 4, layers: int = 2):
            super().__init__()
            self.emb = nn.Embedding(vocab + 1, dim, padding_idx=0)
            self.pos = nn.Embedding(max_len, dim)
            enc = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 2, batch_first=True
            )
            self.enc = nn.TransformerEncoder(enc, num_layers=layers)
            self.head = nn.Linear(dim, 1)

        def forward(self, x):
            mask = x == 0
            h = self.emb(x) + self.pos(torch.arange(x.size(1), device=x.device).unsqueeze(0))
            h = self.enc(h, src_key_padding_mask=mask)
            pooled = h[:, 0, :]
            return self.head(pooled).squeeze(-1)

    model = TinyTransformer(len(chars))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    fit_idx, cal_idx = _split_fit_cal(rows, train_idx)
    Xfit = torch.tensor(X[fit_idx], dtype=torch.long)
    yfit = torch.tensor(y[fit_idx], dtype=torch.float32)
    Xcal = torch.tensor(X[cal_idx], dtype=torch.long)

    model.train()
    for _ in range(3):
        opt.zero_grad()
        logits = model(Xfit)
        loss = loss_fn(logits, yfit)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        cal_scores = torch.sigmoid(model(Xcal)).numpy()
    threshold = metrics.threshold_at_fpr(cal_scores, y[cal_idx], metrics.TARGET_FPR)
    with torch.no_grad():
        test_scores = torch.sigmoid(model(Xte)).numpy()
    elapsed = time.time() - t0
    print(f"  transformer baseline trained in {elapsed:.1f}s", file=sys.stderr)

    metrics_out = metrics.at_threshold(test_scores, y[test_idx], threshold)
    return Result.measured(
        "transformer (URLTran-style)",
        len(fit_idx),
        len(test_idx),
        float(y[test_idx].mean()),
        metrics_out,
    )


def _split_fit_cal(rows: list[Row], train_idx: list[int]) -> tuple[list[int], list[int]]:
    rng_idx = list(train_idx)
    import random as _r

    _r.Random(1).shuffle(rng_idx)
    cut = int(len(rng_idx) * 0.25)
    return sorted(rng_idx[cut:]), sorted(rng_idx[:cut])
