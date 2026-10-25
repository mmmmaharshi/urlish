# urlish benchmark report

Target false-positive rate: 1%. Thresholds are picked on held-out
training data, never on the test set.

## Dataset

| collection | stratum | rows | positives | negatives |
|---|---|---:|---:|---:|
| majestic | head | 10,000 | 0 | 10,000 |
| openphish | - | 300 | 300 | 0 |
| phishing_army | - | 12,000 | 12,000 | 0 |
| tranco | head | 10,000 | 0 | 10,000 |
| tranco | tail | 50,000 | 0 | 50,000 |
| **main dataset** | | **72,300** | **12,300** | **60,000** |

Majestic contributes 10,000 negatives and is used only by the
collection-shift rows, so it cannot move the other numbers.

## Results

| split | n fit | n test | pos rate | recall@1%FPR | precision@1%FPR | ROC-AUC | realised FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| random rows (the usual baseline) | 43,380 | 14,460 | 17.0% | **63.5%** | 91.5% | 0.955 | 1.21% |
| domain-disjoint | 43,380 | 14,460 | 17.2% | **62.3%** | 94.3% | 0.958 | 0.79% |
| popularity shift (head train, tail test) | 14,880 | 52,460 | 4.7% | **69.0%** | 58.7% | 0.959 | 2.39% |
| collection shift, host-only | 12,756 | 10,300 | 2.9% | **42.0%** | 55.5% | 0.806 | 1.01% |
| collection shift, as deployed (paths kept) | 12,756 | 10,300 | 2.9% | **54.0%** | 61.6% | 0.914 | 1.01% |
| impersonation rules only (no ML) | 0 | 14,460 | 17.2% | **4.8%** | 85.1% | 0.507 | 0.18% |


## What the numbers say

The baseline reports 63.5% recall and 91.5%
precision at a 1% false-positive budget, AUC 0.955. Even the leaky
split is not 99%: this is host-level detection on live collections, not a toy.

Precision depends on how many positives the test set happens to contain, and that
ratio varies by an order of magnitude across rows (pos rate column). Compare rows
on recall@1%FPR and AUC; read precision only against its own pos rate.

**Domain leakage costs little.** Domain-disjoint evaluation loses
1.2 points of recall
(63.5% -> 62.3%) with AUC essentially unchanged
(0.955 -> 0.958). Phishing hosts are mostly single-use, so
there was less to memorise than the usual story assumes.

**Popularity shift spends the false-positive budget.** Training on ranks 1-10k and
testing on 10k-60k leaves AUC at 0.959 - but the threshold calibrated
for 1% false positives realises 2.39%, and recall lands at
69.0%. A shipped detector would block roughly 1 in
42 legitimate popular domains while every ranking
metric looked healthy.

**Collection shift is the largest honest drop.** With negatives taken from
two independent lists, this row changes the collector on both sides - Phishing
Army to OpenPhish, Tranco to Majestic - while holding the popularity band and the
host-only view fixed. AUC falls 0.955 -> 0.806 and
recall 63.5% -> 42.0%, with a realised FPR of
1.01% on negatives from a list it never saw.

4,992 rows were dropped from the training half because their
registrable domain appeared in the test half: the collections overlap heavily,
and a collection split that ignores that is measuring leakage, not transfer.

**Keeping the URL path recovers the ranking, and then some.** The same model,
scored on full URLs, reaches AUC 0.914 and recall
54.0% (95% CI 48.3%-59.6% on
300 positives), though the realised FPR is still
1.01%. Path length and lure words carry real signal, which is why
`featurize()` normalises away scheme and trailing slash rather than discarding the
path.

**Brand rules are precise but narrow.** No training data, 85.1%
precision at 0.18% FPR, but 4.8% recall. Most phishing
hosts do not impersonate a brand by string similarity; they are throwaway domains
on cheap TLDs. Rules and classifiers cover different halves of the problem, which
is the argument for shipping both.

## Operating point of the shipped detector

`urlish` scores at **0.8889**. The threshold is calibrated for a
1% budget on a domain-disjoint quarter of the training rows, using the same model
that is stored in the artifact, so the two cannot disagree. On the domain-disjoint
test row that pair realises 0.87% false positives at
63.9% recall; the 1% budget is spent on the calibration set,
and a different test distribution can spend more.
Scoring at 0.5 instead would realise a false-positive rate an order of magnitude
over budget, because a gradient-boosted probability is not a decision threshold.

## Limitations

- Single snapshot. No dates in these collections, so there is no temporal split. A
  temporal holdout is the missing experiment and is strictly harder than any row
  above.

- Labels are collection membership, not ground truth. Collections contain dead,
  parked and occasionally mislabelled hosts, and a popular domain is not a
  guarantee of safety.

- The collection-shift test has 300 positives and
  10,000 negatives, hence
  the interval above. Every other split is >= 14k rows.

- The training collection (`phishing_army`) is host-only, so path features are
  unseen at fit time. That is the realistic case for a collection consumer, not a
  contrived one, but it is a limitation of the data, not a finding about phishing.

- Negatives come from domain lists, so even with `www.` prefixes they carry at
  most one subdomain. Deep subdomain chains appear only in positives, and the model
  leans on that.

- Features are string-only: no DNS, no certificate, no page content. That is the
  client-side constraint and it caps what is achievable.

- Brands come from Tranco, which over-represents Western sites. Impersonation of
  regional brands is under-covered.


## Reproduce

```
uv sync
uv run urlish-benchmark
uv run pytest -q
```

