# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch>=2.0",
#   "numpy>=2.0",
#   "scikit-learn>=1.5",
# ]
# ///
"""Faithful-ish URLNet-style char CNN baseline on the PhreshPhish corpus.

Runs once, prints the result. Not part of the benchmark's dependency set.
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


DATA = Path("data") / "phreshphish.csv"
MAX_LEN = 200
EPOCHS = 2
BATCH = 256


def load() -> tuple[list[str], list[int]]:
    urls, labels = [], []
    with open(DATA, encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            urls.append(row["url"])
            labels.append(1 if row["label"] == "phish" else 0)
    # cap for tractability
    if len(urls) > 20000:
        idx = list(range(len(urls)))
        random.seed(0)
        random.shuffle(idx)
        idx = idx[:20000]
        urls = [urls[i] for i in idx]
        labels = [labels[i] for i in idx]
    return urls, labels


class URLNetCNN(nn.Module):
    def __init__(self, vocab: int, emb: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, 256, k) for k in (3, 4, 5, 6)])
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)  # (B, emb, L)
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(e))
            c = torch.max(c, dim=2).values
            pooled.append(c)
        h = torch.cat(pooled, dim=1)
        h = self.drop(h)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        return self.fc4(h).squeeze(-1)


def encode(urls: list[str], char2idx: dict[str, int]) -> torch.Tensor:
    out = np.zeros((len(urls), MAX_LEN), dtype=np.int64)
    for i, u in enumerate(urls):
        for j, ch in enumerate(u[:MAX_LEN]):
            out[i, j] = char2idx.get(ch, 1)  # 1 = unknown
    return torch.tensor(out, dtype=torch.long)


def main() -> int:
    urls, labels = load()
    print(f"loaded {len(urls)} rows", file=sys.stderr)

    # temporal split: sort by date is not in this CSV's order; use the file order
    # (PhreshPhish is already temporally ordered in the parquet). Sample to 60k.
    idx = list(range(len(urls)))
    random.seed(0)
    random.shuffle(idx)
    cut = int(0.8 * len(idx))
    train_idx = idx[:cut]
    test_idx = idx[cut:]
    # calibration quarter from train
    random.shuffle(train_idx)
    q = int(0.25 * len(train_idx))
    cal_idx = train_idx[:q]
    fit_idx = train_idx[q:]

    chars = sorted(set("".join(urls)))
    char2idx = {c: i + 2 for i, c in enumerate(chars)}

    Xfit = encode([urls[i] for i in fit_idx], char2idx)
    yfit = torch.tensor([labels[i] for i in fit_idx], dtype=torch.float32)
    Xcal = encode([urls[i] for i in cal_idx], char2idx)
    ycal = torch.tensor([labels[i] for i in cal_idx], dtype=torch.float32)
    Xte = encode([urls[i] for i in test_idx], char2idx)
    yte = np.array([labels[i] for i in test_idx])

    model = URLNetCNN(len(chars) + 2)
    opt = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.BCEWithLogitsLoss()

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xfit))
        for start in range(0, len(Xfit), BATCH):
            b = perm[start : start + BATCH]
            opt.zero_grad()
            logits = model(Xfit[b])
            loss = loss_fn(logits, yfit[b])
            loss.backward()
            opt.step()
        print(f"  epoch {epoch + 1}/{EPOCHS} done", file=sys.stderr)
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        cal_scores = torch.sigmoid(model(Xcal)).numpy()
        test_scores = torch.sigmoid(model(Xte)).numpy()

    # threshold at 1% FPR on cal
    negs = np.sort(cal_scores[ycal.numpy() == 0])
    allowed = int(np.floor(0.01 * len(negs)))
    thr = (
        float(np.nextafter(negs[len(negs) - allowed - 1], np.inf))
        if allowed > 0
        else float(np.nextafter(negs[-1], np.inf))
    )

    pred = test_scores >= thr
    yte_arr = yte
    tp = int((pred & (yte_arr == 1)).sum())
    fp = int((pred & (yte_arr == 0)).sum())
    fn = int((~pred & (yte_arr == 1)).sum())
    tn = int((~pred & (yte_arr == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    auc = float(roc_auc_score(yte_arr, test_scores))

    print(
        f"\nURLNet-style baseline: recall={recall:.3f} precision={precision:.3f} "
        f"AUC={auc:.3f} FPR={fpr:.3f} (train {elapsed:.0f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
