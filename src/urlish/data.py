"""Dataset assembly and the split strategies that decide the paper's numbers.

Two independent axes decide where a row came from: the collection it belongs to
(`phishing_army`, `openphish`, `tranco`, `majestic`) and, for negatives, the
popularity stratum (`head`, `tail`). Keeping them separate is the point: a split
that varies one axis while holding the other is the only way to attribute a drop
to either.

The split is the experiment. `random_split` reproduces the evaluation used by
most published URL classifiers; `group_split` removes leakage and `select`
builds the collection and stratum contrasts.
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .core import host_of, registrable_domain

STRATA = ("head", "tail")


@dataclass(frozen=True)
class Row:
    url: str
    y: int
    domain: str  # registrable domain
    collection: str  # upstream list the row came from
    stratum: str | None  # "head"/"tail" for negatives, None for positives


def _dedupe(rows: list[Row]) -> list[Row]:
    seen: set[tuple[str, int]] = set()
    out = []
    for row in rows:
        key = (row.url.lower(), row.y)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _require_rows[T](rows: list[T], path: str | Path, expected: str, skipped: int = 0) -> list[T]:
    """Reject an empty collection instead of letting it skew every later number."""
    if not rows:
        raise ValueError(f"{path}: no rows parsed; expected {expected}")
    if skipped:
        print(f"{path}: skipped {skipped} malformed line(s)", file=sys.stderr)
    return rows


def load_tranco(path: str | Path) -> list[tuple[int, str]]:
    """Tranco CSV is `rank,domain` with no header."""
    rows = []
    skipped = 0
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[0].isdigit():
                rows.append((int(parts[0]), parts[1].strip().lower()))
            else:
                skipped += 1
    return _require_rows(rows, path, "rank,domain", skipped)


def load_majestic(path: str | Path) -> list[tuple[int, str]]:
    """Majestic Million CSV: GlobalRank in column 0, Domain in column 2."""
    rows = []
    skipped = 0
    with open(path, encoding="utf-8-sig") as fh:
        next(fh, None)  # header
        for line in fh:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) >= 3 and parts[0].isdigit():
                rows.append((int(parts[0]), parts[2].strip().lower()))
            else:
                skipped += 1
    return _require_rows(rows, path, "GlobalRank,Domain CSV with a header", skipped)


def load_url_lines(path: str | Path) -> list[str]:
    """One URL per line, comments and blanks skipped."""
    out = []
    skipped = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            low = line.lower()
            if low.startswith(("http://", "https://")):
                out.append(line)
            elif "." in low and " " not in low:
                out.append("http://" + line)
            else:
                skipped += 1
    return _require_rows(out, path, "one URL per line", skipped)


def negative_rows(
    ranked: list[tuple[int, str]],
    collection: str,
    *,
    head_cut: int = 10_000,
    tail_cut: int = 60_000,
    cap: int = 60_000,
    strata: tuple[str, ...] = STRATA,
) -> list[Row]:
    """Sample negative rows from a ranked domain list, one band per stratum."""
    rows: list[Row] = []
    bands = {
        "head": [(r, d) for r, d in ranked if r <= head_cut],
        "tail": [(r, d) for r, d in ranked if head_cut < r <= tail_cut],
    }
    for stratum in STRATA:
        band = bands[stratum]
        if stratum not in strata or not band:
            continue
        take = min(cap, len(band))
        step = max(1, len(band) // take)
        for i, (_, domain) in enumerate(band[::step][:take]):
            # Ranked lists hold registrable domains only, so a plain `domain`
            # row has zero subdomains and the model learns "has a subdomain =
            # phishing". Half the rows get a www prefix to break that shortcut.
            host = f"www.{domain}" if i % 2 else domain
            rows.append(
                Row(
                    f"http://{host}/",
                    0,
                    registrable_domain(domain),
                    collection,
                    stratum,
                )
            )
    return rows


def build_dataset(
    positives: dict[str, list[str]],
    negatives: dict[str, list[tuple[int, str]]],
    *,
    head_cut: int = 10_000,
    tail_cut: int = 60_000,
    pos_cap: int = 12_000,
    neg_cap: int = 60_000,
) -> list[Row]:
    """Positives from named phishing collections, negatives from ranked lists."""
    rows: list[Row] = []

    for collection, urls in positives.items():
        step = max(1, len(urls) // pos_cap)
        for url in urls[::step][:pos_cap]:
            rows.append(Row(url, 1, registrable_domain(host_of(url)), collection, None))

    for collection, ranked in negatives.items():
        rows.extend(
            negative_rows(
                ranked,
                collection,
                head_cut=head_cut,
                tail_cut=tail_cut,
                cap=neg_cap,
            )
        )

    return _dedupe(rows)


def select(
    rows: list[Row],
    *,
    collections: set[str] | None = None,
    stratum: str | None = None,
    label: int | None = None,
) -> list[int]:
    """Indices matching every condition given; None means "any"."""
    out = []
    for i, row in enumerate(rows):
        if collections is not None and row.collection not in collections:
            continue
        if stratum is not None and row.stratum != stratum:
            continue
        if label is not None and row.y != label:
            continue
        out.append(i)
    return out


def remove_domains(idxs: list[int], rows: list[Row], domains: set[str]) -> list[int]:
    """Drop rows whose registrable domain appears in `domains`.

    A collection split is only honest if the two halves share no domain, and
    popular-domain lists overlap heavily with each other.
    """
    return [i for i in idxs if rows[i].domain not in domains]


def _by_label(rows: list[Row]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row.y].append(i)
    return groups


def random_split(rows: list[Row], frac: float, seed: int = 0) -> tuple[list[int], list[int]]:
    """Stratified random rows. This is the leaky baseline."""
    rng = random.Random(seed)
    test: list[int] = []
    for idxs in _by_label(rows).values():
        idxs = idxs[:]
        rng.shuffle(idxs)
        test.extend(idxs[: int(len(idxs) * frac)])
    test_set = set(test)
    train = [i for i in range(len(rows)) if i not in test_set]
    return train, sorted(test)


def group_split(rows: list[Row], frac: float, seed: int = 0) -> tuple[list[int], list[int]]:
    """No registrable domain appears in both halves."""
    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row.domain].append(i)
    keys = list(groups)
    rng.shuffle(keys)
    target = int(len(rows) * frac)
    test: list[int] = []
    for key in keys:
        if len(test) >= target:
            break
        test.extend(groups[key])
    test_set = set(test)
    train = [i for i in range(len(rows)) if i not in test_set]
    return train, sorted(test)


def describe(rows: list[Row]) -> dict[tuple[str, str], dict[str, int]]:
    out: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"n": 0, "pos": 0, "neg": 0})
    for row in rows:
        bucket = out[(row.collection, row.stratum or "-")]
        bucket["n"] += 1
        bucket["pos" if row.y == 1 else "neg"] += 1
    return dict(out)
