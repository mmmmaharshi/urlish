# PhreshPhish: facts for the v0.2.0 migration

Source of record: arXiv 2507.10854 v2 (HTML, accessed 2026-10-26), the dataset
card at `huggingface.co/datasets/phreshphish/phreshphish`, and the parquet
footers read over HTTPS (one shard each, no full download). Every claim below
names its source. Items that could not be verified are marked "unverified".

## Short answer

PhreshPhish is usable for a URL-only benchmark, but only as a URL corpus. Its
date column is a collection date, the split is a 70/30 temporal cut followed by
an HTML-based leakage filter, and the negatives are mostly popular hosts. The
fetch is cheap: roughly 25 MB for the four columns we need, not 200 GB.

## 1. Schema and physical layout

Parquet schema (8 columns), from `test-000.parquet` footer:

| column | type (observed) |
|---|---|
| `sha256` | string |
| `url` | string |
| `label` | string (`phish` or the dataset's negative label) |
| `target` | string, nullable (brand name, or `other`, or null) |
| `date` | string |
| `lang` | string |
| `lang_score` | double |
| `html` | string |

Per-row compressed bytes, measured on `test-000.parquet` (1,000 rows, one row
group):

| column | bytes/row |
|---|---:|
| `html` | 55,262 |
| `sha256` | 35 |
| `url` | 34 |
| `lang_score` | 4.3 |
| `target` | 1.8 |
| `date` | 1.0 |
| `lang` | 0.6 |
| `label` | 0.2 |

Reading only `url`, `date`, `label`, `target` costs about 37 bytes per row.
For the full test split (168,060 rows) that is roughly 6 MB; for the train split
(498,255 rows) roughly 18 MB. Total fetch is on the order of 25 MB. The HTML
column is 99.9% of the bytes, so column projection is the whole game.

Shards: 56 train files plus 21 test files. Each train shard holds roughly 8,000
to 9,000 rows. The first shard of each split is a small sample (1,000 rows).
Source: HF tree API, `api/datasets/phreshphish/phreshphish/tree/main/data`.

## 2. Date semantics

The `date` column is the **date of collection** of the page. The paper calls it
"date of collection" and shows it in Figure 4, aggregated daily. It is not a
first-seen date and not a crawl date measured by us; it is the upstream
dataset's own label. Source: arXiv 2507.10854 v2, Section 4.1 and Figure 4.

Observed range in one train shard (`train-001.parquet`, 5,780 rows):
2024-07-02 to 2025-09-08. The card states collection ran July 2024 to
December 2025, and v1.0.1 added roughly 200,000 samples collected March to
December 2025. Source: dataset card changelog.

The card's v1.0.1 changelog says it "improved temporal consistency by
downsampling some earlier samples." That means the training half is not the raw
first 70% of the original collection; some early rows were removed so the split
stays clean. Consequence: the train set is earlier-heavy and the test set is
later-heavy, but the exact per-month counts after downsampling are not published.
Full per-month counts are unverified; measuring them requires reading the `date`
column of all 77 shards.

## 3. Split definition

From the paper, Section 4.1 and Algorithm 2 (Appendix B):

1. Sort all rows by date.
2. Train = the first floor((1 - p) * |D|) rows, test = the rest, with p = 0.3.
   The cut is stratified by class so both halves keep the same class proportion.
3. Leakage filter on the test half: traverse the test rows and drop any whose
   TF-IDF cosine distance to the closest training row is below a threshold. LSH
   makes the search scalable.

So the split is per-sample and time-ordered, not per-domain. The leakage filter
is HTML/URL-text based and runs on the full page content. A URL-only benchmark
cannot reproduce it exactly; the closest URL-level stand-in is the existing
domain-disjoint rule (no shared registrable domain). That difference must be
stated in the report.

## 4. Leakage and domain reuse

Phishing pages reuse registrable domains heavily. In one train shard, 2,596
phishing rows came from only 1,012 distinct registrable domains; the top five
domains alone held 755 rows (gitbook.io 235, webflow.io 221, google.com 136,
weeblysite.com 105, weebly.com 98). Source: `train-001.parquet` footer read.

This means the domain-disjoint split will drop a large share of rows, and the
shared-domain removal in ADR 0002 applies here too. The `target` column is the
impersonated brand; 60% of phishing rows in that shard had `target = other`,
and 3181 of 3184 negative rows had `target = null`. There are roughly 160
distinct targets per shard. Note a data-quality wrinkle: the same brand appears
as both `at&t` and `at&amp;t`.

## 5. Target brands

Major global brands appear (`facebook`, `meta`, `whatsapp`, `instagram`,
`netflix`, `usps`, `robinhood`, `booking`, `rakuten`, `naver`). About 40% of
phishing rows in the sampled shard name a specific brand; the rest are `other`.
Unverified: the exact brand distribution across all shards and whether regional
brands are present in meaningful numbers.

## 6. Negatives and popularity strata

In the sampled shard, 60.6% of negative hosts (1,931 of 3,184) are in the Tranco
top 100,000 by registrable domain: 1,286 in the head band (rank <= 10,000) and
525 in the tail band (rank 10,001 to 60,000). About 40% of negatives are
unranked and carry no stratum. Source: `data/tranco.csv` joined on registrable
domain.

Consequence: the popularity-shift row stays viable, but roughly four in ten
negative rows are excluded from it. The report must print that coverage, which
the spec requires.

## 7. License and terms

The dataset is released under Creative Commons Attribution 4.0 International and
"should only be used for anti-phishing research." Source: dataset card,
"License & Terms of Use".

CC BY 4.0 requires attribution and permits redistribution and derivatives with
attribution. The "anti-phishing research" sentence reads as a request, not an
added legal restriction; CC BY 4.0 does not carry use-field restrictions. A
derived CSV of `url`, `date`, `label`, `target` may be redistributed with
attribution, but this repo does not redistribute it: `data/` is untracked and
the fetch script is the reproducibility path. The README and report cite the
dataset and the paper regardless.

## 8. Citation

Dalton, T., Gowda, H., Rao, G., Pargi, S., Hadj Khodabakhshi, A., Rombs, J.,
Jou, S., and Marwah, M. "PhreshPhish: A Real-World, High-Quality, Large-Scale
Phishing Website Dataset and Benchmark." arXiv:2507.10854, 2025.
https://arxiv.org/abs/2507.10854

## 9. Traps

- The `datasets` library downloads full parquet shards, HTML included, even when
  you select columns. Do not use it for the fetch. Use pyarrow over an HTTP
  filesystem with column projection.
- The first shard of each split is a 1,000-row sample, not a normal shard.
- `target` is null for negative rows and `other` for a large share of positives.
- Brand names are inconsistent (`at&t` vs `at&amp;t`).
- The leakage filter is HTML-based; a URL-only study cannot reproduce it.
- v1.0.1 downsampled early rows; re-running on v1.0.0 would change the dates.

## 10. Three biggest risks for the migration

1. The popularity row loses roughly 40% of negatives to unranked hosts, so its
   test set is smaller and more popular than the rest.
2. The domain-disjoint row will drop a large share of phishing rows, because the
   same platform domains host hundreds of pages each.
3. The temporal row cannot reproduce the paper's HTML-based leakage filter, so
   it is a URL-level approximation and must be labelled as one.
