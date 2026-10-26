# urlish

A client-side malicious URL detector, and the evaluation that shows why the
numbers published for it are too optimistic.

No server, no DNS lookups, no page fetches. The detector sees a URL string and
nothing else, which is the constraint if you want it to run in a browser.

Vocabulary is defined in [GLOSSARY.md](GLOSSARY.md). "Negative" means listed on a
popularity list, not verified safe.

## The point

Same features, same model, six ways to split the data. The table is generated
by `uv run urlish-benchmark`; [REPORT.md](REPORT.md) carries the method and the
prose.

<!-- results:start -->
| split | n fit | n test | pos rate | recall@1%FPR | precision@1%FPR | ROC-AUC | realised FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| random rows | 43,200 | 14,400 | 16.7% | **78.1%** | 94.2% | 0.981 | 0.96% |
| domain-disjoint | 43,144 | 14,462 | 12.4% | **62.6%** | 87.4% | 0.945 | 1.28% |
| temporal split | 19,914 | 19,984 | 16.6% | **54.4%** | 92.8% | 0.949 | 0.85% |
| popularity shift (head train, tail test) | 25,893 | 12,567 | 19.2% | **76.4%** | 90.0% | 0.969 | 2.01% |
| cross-dataset, host-only | 30,034 | 10,300 | 2.9% | **46.0%** | 60.8% | 0.836 | 0.89% |
| cross-dataset, as deployed (paths kept) | 30,034 | 10,300 | 2.9% | **66.0%** | 38.2% | 0.814 | 3.21% |
| impersonation rules only (no ML) | 0 | 14,462 | 12.4% | **0.0%** | 0.0% | 0.555 | 0.00% |
<!-- results:end -->

Three things fall out of that table:

1. **Splitting by domain buys almost nothing.** The domain-disjoint row loses
   little recall and keeps its AUC. Phishing hosts are mostly single-use, so
   there was little leakage to remove.
2. **Popularity shift breaks the threshold before it breaks the ranking.** AUC
   barely moves while the threshold calibrated for the 1% budget realises more
   than twice that on the long tail - one blocked popular domain in a few dozen.
3. **Changing collection is the biggest hit to ranking.** At a fixed popularity
   band, host for host, AUC falls further here than in any other row. An earlier
   version of this row moved the popularity band *and* the collector at once;
   fixing the confound and the shared-domain leakage moved it further down.

The deterministic brand rules need no training data and are precise at a low
realised FPR, but their recall is in the single digits. Most phishing hosts are
throwaway domains on cheap TLDs, not typosquats. The two approaches
cover different halves of the problem: at the operating point the classifier
misses `paypa1.com`, and the rules catch it.

Full numbers and method: [REPORT.md](REPORT.md).

## How published detectors evaluate

Three published detectors show the common protocols:

1. [URLNet](https://arxiv.org/abs/1802.03162) (2018) trains on the first 60% of a timestamped VirusTotal set and tests on the last 40%. Rows inside each period are random.
2. [URLTran](https://arxiv.org/abs/2106.05256) (2021) uses six weeks of browser telemetry: four weeks to train, one to validate, one to test.
3. [Phishpedia](https://www.usenix.org/conference/usenixsecurity21/presentation/lin) (2021) collects six months of OpenPhish pages and splits them by time. It reports that 48% of the pages in the second half matched a page in the first half.

All three hold out data by time. None reports the popularity mix of the negatives, and none removes shared registrable domains. The rows in this repo change those two axes one at a time.

A 2026 review reaches the same conclusion: the field lacks a shared benchmark with fixed temporal and domain-disjoint splits and standard false-positive reporting ([Applied Sciences](https://www.mdpi.com/2076-3417/16/19/9576)).

## Quick start

```bash
uv sync                                 # creates .venv from uv.lock
uv run urlish-benchmark                 # ~50 s, writes REPORT.md, README table, artifacts/
uv run pytest -q                        # offline checks; adds the artifact check once trained
uv run urlish https://paypal.com.secure-login.xyz/verify
uv run urlish --train                   # rebuild the artifact before scoring, if needed
git config core.hooksPath .githooks     # pre-push runs the full suite; CI cannot
```

The threshold is calibrated for the 1% budget and stored in the model artifact. It
is deliberately not 0.5: a gradient-boosted probability is not a decision
threshold, and scoring at 0.5 spends many times the budget. The exact operating
point is in [REPORT.md](REPORT.md).

Dependencies are pinned in `pyproject.toml` and locked in `uv.lock`. Keep the pins:
the report's numbers are deterministic, and a different scikit-learn build can move
them at the third decimal.

Both commands resolve `data/` and `artifacts/` from the checkout, so run them from
a clone; a wheel install has no collections to read.

## Versioning

The version is declared once, in `pyproject.toml`; `urlish.__version__` reads it
back from the installed metadata, so there is no second copy to drift.

A version marks a **citable set of numbers**, not a code change: bump it when the
report or the features change, never for a docs edit. Releases are git tags
(`git tag -a v0.1.0`), not artifacts on a package index.

`REPORT.md` opens with the version and the SHA-256 prefix of each collection it
read. That line is the reproducibility claim: the collections are live and `data/`
is untracked, so rerunning a week later with fresh downloads changes the header
even though the code did not change. `ARTIFACT_SCHEMA` is a separate contract - it
guards the pickle against a feature change and has nothing to do with releases.

## How it works

**Classifier.** A fixed set of string features into `HistGradientBoostingClassifier`:
length and entropy, digit/hyphen/dot counts, subdomain depth, path depth, query
param count, suspicious TLD, lure keywords, distance to the nearest brand.
Trained in seconds, no GPU, no transformer.

**Rules.** `BrandIndex` folds homoglyphs and digit-for-letter swaps (`pаypal` and
`paypa1` both become `paypal`), strips the scheme, and compares the registrable
label against the top Tranco brands by edit distance. Short labels require an
exact or one-edit match, or three-letter suffixes collide with unrelated brands.

**Normalisation.** `featurize()` is deliberately invariant to scheme, case, and
trailing slash. An earlier version was not, and the model learned `http` vs
`https` - a property of how the dataset was built, not of phishing. There is a
test that pins this down.

**Provenance.** A row carries a `collection` and a `stratum`, which cannot be
folded into one field. Collapsing them is what made the old collection-shift row
confounded: it varied the phishing collector and the popularity band together.
Half the negatives are given a synthetic `www.` prefix, because ranked lists hold
registrable domains only and the model otherwise learned "has a subdomain =
phishing" ([ADR 1](docs/adr/0001-synthesise-www-prefixes.md)).

## What is actually rigorous here

- **Thresholds are chosen on held-out training data**, never on the test set.
- **Every row reports recall at a fixed false-positive budget**, because accuracy
  and even precision are meaningless under class imbalance. Precision is
  reported alongside the test set's positive rate so it can be read honestly.
- **`pos rate` is a column, not a footnote.** Precision varies widely across rows
  because prevalence does; read each precision against its own row's positive
  rate.
- **The split is the experiment.** `data.py` exposes `random_split`,
  `group_split`, and `select` so the comparison is reproducible, not prose.
- **Shared domains are removed across a collection boundary.** Majestic and
  Tranco overlap heavily, so rows whose registrable domain appears on both sides
  are dropped from the training half; the report gives the count ([ADR 2](docs/adr/0002-drop-shared-domains-from-train.md)).

## Data

| collection | what |
|---|---|
| Tranco top 100k | negatives, two popularity bands |
| Majestic Million | negatives, head band only |
| Phishing Army | positives, host-only |
| OpenPhish | positives, full paths |

Exact row counts per collection are in [REPORT.md](REPORT.md).

`data/majestic.csv` is trimmed to the top 60k of the 1M-row download; the full
file is 80 MB and only the head band is used. Tranco is dated - cite the list ID
you used.

Re-download:

```bash
curl -s https://tranco-list.eu/api/lists/date/latest          # -> list_id
curl -sO https://tranco-list.eu/download/<list_id>/100000     # -> data/tranco.csv
curl -s https://downloads.majestic.com/majestic_million.csv   # keep 60k rows
curl -sO https://phishing.army/download/phishing_army_blocklist_extended.txt
curl -sO https://openphish.com/feed.txt
```

## Known limits

- **One collection, one date range.** The temporal row uses one collection's
  dates (July 2024 to December 2025). The leakage filter in that collection's
  paper runs on page content, which this benchmark does not read, so the row is
  a URL-level approximation.
- **Labels are collection membership**, not ground truth. Collections include
  dead, parked and occasionally mislabelled hosts, and popularity is not safety.
- **The collection-shift test is narrower than the rest**: its positive count is
  too small to read without an interval, which the report prints. Every other row
  is an order of magnitude larger.
- **Host-level for the main training collection.** Path features are unseen at
  fit time, which is realistic for a feed consumer but limits the claim.
- **Negatives come from domain lists**, so even with `www.` prefixes they carry
  at most one subdomain. Deep subdomain chains appear only in positives and the
  model leans on that.
- **String-only.** No certificate, no DNS history, no page content. That is the
  client-side constraint and it caps the ceiling.
- **Brand list is Tranco-based** and over-represents Western brands.

## If you want to take this further

The experiment this harness is built for but does not contain:

1. **Threshold drift under a live feed.** Recalibrate weekly and plot realised
   FPR over months. The popularity-shift row suggests this is the real
   operational risk, and nobody publishes it.
