# urlish

A client-side malicious URL detector, and the evaluation that shows why most
published versions of it do not work.

No server, no DNS lookups, no page fetches. The detector sees a URL string and
nothing else, which is the constraint if you want it to run in a browser.

Vocabulary is defined in [GLOSSARY.md](GLOSSARY.md). "Negative" means listed on a
popularity list, not verified safe.

## The point

Same features, same model, six ways to split the data:

| split | recall@1%FPR | ROC-AUC | realised FPR |
|---|---:|---:|---:|
| random rows (the usual baseline) | 63.3% | 0.955 | 1.13% |
| domain-disjoint | 62.7% | 0.958 | 0.79% |
| popularity shift (head train, tail test) | 69.0% | 0.959 | 2.29% |
| collection shift, host-only | 28.0% | 0.810 | 0.95% |
| collection shift, full URLs | 48.3% | 0.930 | 0.95% |
| impersonation rules, no ML | 4.8% | 0.507 | 0.18% |

Three things fall out of that table:

1. **Splitting by domain buys almost nothing.** It costs 0.6 points of recall.
   Phishing hosts are mostly single-use, so there was little leakage to remove.
2. **Popularity shift breaks the threshold before it breaks the ranking.** AUC
   holds at 0.959 while a threshold calibrated for 1% false positives realises
   2.29% - one blocked popular domain in 44.
3. **Changing collection is the biggest hit to ranking.** At a fixed popularity
   band, host for host, AUC falls from 0.955 to 0.810. An earlier version of this
   row moved the popularity band *and* the collector at once and reported 0.855;
   fixing the confound and the shared-domain leakage moves it further down.

The deterministic brand rules need no training data and reach 85.1% precision at
0.18% FPR, but only 4.8% recall. Most phishing hosts are throwaway domains on
cheap TLDs, not typosquats. The two approaches cover different halves of the
problem: at the 1% operating point the classifier misses `paypa1.com`, and the
rules catch it.

Full numbers and method: [REPORT.md](REPORT.md).

## Quick start

```bash
uv sync                                 # creates .venv from uv.lock
uv run urlish-benchmark                 # ~50 s, writes REPORT.md and artifacts/
uv run pytest -q                        # offline checks + the artifact operating point
uv run urlish https://paypal.com.secure-login.xyz/verify
```

```
https://paypal.com.secure-login.xyz/verify
  classifier   phishing  (p=1.000, threshold 0.887)
  impersonates paypal (paypal.com) - brand-token, edit distance 0
```

The threshold is calibrated for a 1% budget and stored in the model artifact. It
is deliberately not 0.5; scoring at 0.5 would realise roughly 11% false positives.

Dependencies are pinned in `pyproject.toml` and locked in `uv.lock`. Keep the pins:
the report's numbers are deterministic, and a different scikit-learn build can move
them at the third decimal.

## How it works

**Classifier.** 25 string features into `HistGradientBoostingClassifier`:
length and entropy, digit/hyphen/dot counts, subdomain depth, path depth, query
param count, suspicious TLD, lure keywords, distance to the nearest brand.
Trained in seconds, no GPU, no transformer.

**Rules.** `BrandIndex` folds homoglyphs and digit-for-letter swaps (`pаypal` and
`paypa1` both become `paypal`), strips the scheme, and compares the registrable
label against the top 2k Tranco brands by edit distance. Short labels require an
exact or one-edit match, or three-letter suffixes collide with unrelated brands.

**Normalisation.** `featurize()` is deliberately invariant to scheme, case, and
trailing slash. An earlier version was not, and the model learned `http` vs
`https` - a property of how the dataset was built, not of phishing. There is a
self-check that pins this down.

**Provenance.** A row carries a `collection` and a `stratum`, which cannot be
folded into one field. Collapsing them is what made the old collection-shift row
confounded: it varied the phishing collector and the popularity band together.
Half the negatives are given a synthetic `www.` prefix, because ranked lists hold
registrable domains only and the model otherwise learned "has a subdomain =
phishing" ([ADR 1](docs/adr/0001-synthesise-www-prefixes.md)).

## What is actually rigorous here

- **Thresholds are chosen on held-out training data**, never on the test set.
- **Every row reports recall at a fixed false-positive rate**, because accuracy
  and even precision are meaningless under class imbalance. Precision is
  reported alongside the test set's positive rate so it can be read honestly.
- **`pos rate` is a column, not a footnote.** Precision varies 46.9% to 94.3%
  across rows largely because prevalence varies 2.9% to 17.2%.
- **The split is the experiment.** `data.py` exposes `random_split`,
  `group_split`, and `select` so the comparison is reproducible, not prose.
- **Shared domains are removed across a collection boundary.** Majestic and
  Tranco overlap heavily; 4,992 rows were dropped from the training half of the
  collection-shift test for exactly that reason ([ADR 2](docs/adr/0002-drop-shared-domains-from-train.md)).

## Data

| collection | what | rows used |
|---|---|---:|
| Tranco top 100k | negatives, two popularity bands | 60,000 |
| Majestic Million | negatives, head band only | 10,000 |
| Phishing Army | positives, host-only | 12,000 |
| OpenPhish | positives, full paths | 300 |

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

- **No temporal split.** These collections carry no dates. A temporal holdout is
  the missing experiment and is harder than anything above, because negatives
  always look "current".
- **Labels are collection membership**, not ground truth. Collections include
  dead, parked and occasionally mislabelled hosts, and popularity is not safety.
- **The collection-shift test is narrower than the rest**: 300 positives and
  10,000 negatives, so its recall carries roughly ±6 points. The other rows are
  >= 14k.
- **Host-level for the main training collection.** Path features are unseen at
  fit time, which is realistic for a feed consumer but limits the claim.
- **Negatives come from domain lists**, so even with `www.` prefixes they carry
  at most one subdomain. Deep subdomain chains appear only in positives and the
  model leans on that.
- **String-only.** No certificate, no DNS history, no page content. That is the
  client-side constraint and it caps the ceiling.
- **Brand list is Tranco-based** and over-represents Western brands.

## If you want to take this further

The two experiments this harness is built for but does not contain:

1. **Temporal split.** Needs dated collections (PhishTank with a key, URLhaus, or
   Certificate Transparency logs). This is the version a reviewer will ask for.
2. **Threshold drift under a live feed.** Recalibrate weekly and plot realised
   FPR over months. The 2.29% row suggests this is the real operational risk,
   and nobody publishes it.
