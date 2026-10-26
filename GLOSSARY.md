# urlish

Vocabulary for a benchmark that measures URL phishing detectors under
distribution shift. These are the honest names. The code and the report do not
all use them yet; where they differ, this file is the intended meaning.

## Labels

**Positive**:
A row whose URL appears in a phishing collection. Not "malicious" — the
collection is a claim about the URL, not a verdict on it.
_Avoid_: phishing sample, malicious URL, bad URL

**Negative**:
A row whose registrable domain appears in the Tranco snapshot. Not "benign" —
popularity is not a safety verdict, and popular sites get compromised.
_Avoid_: benign, safe, legitimate, good URL

**Label**:
Which collection a row is in. The benchmark measures agreement with collectors,
not with ground truth, and every claim inherits that limit.

## Sources

**Collection**:
One upstream list a row came from: Phishing Army, OpenPhish, Tranco, or
PhreshPhish.
_Avoid_: source, dataset, feed

**Collection date**:
The date a collection assigns to a row. It is the upstream's label, not a date
this repository measured, and every claim about time inherits that limit.

**Collector**:
The operator of a phishing collection. Tranco is not a collector, so the word
never applies to negatives.
_Avoid_: source

**Popularity stratum**:
Head (rank ≤ 10,000) or tail (10,001–60,000). A property of Tranco rows only;
meaningless for positives.

## Identity

**Host**:
The hostname a URL names, lowercased, trailing dot removed, ignoring port and
credentials.

**Registrable domain**:
eTLD+1 — the public suffix plus one label, so `login.secure.example.co.uk`
becomes `example.co.uk`. The grouping key for deduplication and for
domain-disjoint splitting.
_Avoid_: domain, root domain, base domain

**URL form**:
The surface string a detector receives. Two URL forms with the same host and
path — differing in scheme, case, or trailing slash — must produce identical
features.

## Splits

**Random split**:
Rows shuffled without regard to identity or collection. Reproduces the published
baseline.

**Domain-disjoint split**:
No registrable domain appears in both halves.

**Collection split**:
Train and test drawn from disjoint collections, with the popularity stratum held
fixed. Every row carries both facts; a split that varies both measures nothing
in particular.
_Avoid_: source split, cross-collector (when the stratum also moves)

**Temporal split**:
Train on rows dated before a cutoff, test on rows dated on or after it, with no
registrable domain shared across the boundary. The cutoff is a date, not a row
count, so the split is reproducible from the bytes.
**Fit set / calibration set / test set**:
Three disjoint parts of every experiment. The model is trained on the fit set,
the threshold is chosen on the calibration set, and nothing from the test set
influences either.

## Measurement

**Baseline**:
A published-style detector measured on the same splits, not a split strategy.
_Avoid_: SOTA, competitor

**Target FPR**:
The false-positive budget, 1%, spent on the calibration set.

**Realised FPR**:
FP/(FP+TN) on the test negatives. Every test negative is a Tranco host, so this
is a rate over popular domains, not over the web.
_Avoid_: FPR, false-positive rate

**Prevalence**:
The positive share of a test set — 0.6% to 17% across the rows here. It
dominates precision, so precision is unreadable without it.
_Avoid_: class balance, pos rate

**Lift**:
Precision divided by prevalence. The only precision-derived number comparable
across test sets of different prevalence.

**Shortcut**:
A feature that encodes how the dataset was assembled rather than what a URL is:
scheme, subdomain depth, a TLD mix particular to one collection.
_Avoid_: leak, artefact, bias

**Leakage**:
Statistical dependence between train and test that lets the model memorise
instead of generalise — chiefly a shared registrable domain.
_Avoid_: leak, contamination, overlap

## Impersonation

**Brand**:
The first label of a registrable domain in the top 2,000 Tranco rows, kept only
if at least four characters and not a generic word.

**Confusable**:
A registrable label that equals a brand only after folding homoglyphs and
digit-for-letter swaps (`paypa1` → `paypal`). The raw label is a different
string, which is what makes it useful to an attacker.
_Avoid_: homoglyph — too narrow, it also covers digit swaps

**Typosquat**:
A registrable label within one or two edits of a brand where folding does not
already make it exact. Labels shorter than eight characters allow one edit only.

**Brand token**:
A brand appearing as a token in a subdomain, path, or query while the
registrable domain belongs to somebody else (`paypal.com.secure-login.xyz`).

**Impersonation score**:
A hand-assigned value per match kind, used only to evaluate the rules alone. It
is not calibrated and carries no probability meaning.

## Provenance

**Data snapshot**:
The exact bytes of the four collection files a run read, identified by the first
eight hex characters of each file's SHA-256. The collections are live and `data/`
is untracked, so a digest is the only claim a report can make about its inputs.
_Avoid_: dataset version, data release

**Published version**:
The package version whose report is citable. Bumped when the numbers or the
features change, never to mark a docs edit. Distinct from `ARTIFACT_SCHEMA`,
which guards the pickle against a feature change.
_Avoid_: release number, semver
