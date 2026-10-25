"""Run six evaluations of the same features and write REPORT.md.

The point is not the model. It is the gap between the first row and the others.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import ROOT, artifact, report
from .core import host_of
from .data import (
    Counts,
    Row,
    build_dataset,
    describe,
    digest,
    group_split,
    load_majestic,
    load_tranco,
    load_url_lines,
    negative_rows,
    random_split,
    remove_domains,
    select,
)
from .features import BrandIndex, analyze, featurize
from .metrics import TARGET_FPR, at_threshold, threshold_at_fpr
from .report import Result, table

# The benchmark runs from a checkout: data/ and artifacts/ are repo-relative, so
# a wheel install has nothing to read. That is the documented contract.
SEED = 0
HEAD_CUT = 10_000
TAIL_CUT = 60_000
DATA = ROOT / "data"
README = ROOT / "README.md"
RESULTS_START = "<!-- results:start -->"
RESULTS_END = "<!-- results:end -->"


def impersonation_score(url: str, brands: BrandIndex) -> float:
    """Deterministic lookalike detector, scored without any training."""
    matches, _ = analyze(url, brands)
    if not matches:
        return 0.0
    best = min(matches, key=lambda m: m.distance)
    if best.kind == "brand-token":
        return 0.99
    return {0: 0.97, 1: 0.90, 2: 0.75}.get(best.distance, 0.5)


def new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.07,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED,
        class_weight="balanced",
    )


def split_indices(
    rows: list[Row],
    idxs: list[int],
    fn: Callable[[list[Row]], tuple[list[int], list[int]]],
) -> tuple[list[int], list[int]]:
    """Run a split over `idxs` and map the halves back to positions in `rows`."""
    sub = [rows[i] for i in idxs]
    first, second = fn(sub)
    return [idxs[i] for i in first], [idxs[i] for i in second]


def calibration_split(
    rows: list[Row], train_idx: list[int], grouped: bool
) -> tuple[list[int], list[int]]:
    """Threshold is chosen on held-out training data, never on the test set."""
    split = group_split if grouped else random_split
    return split_indices(rows, train_idx, lambda sub: split(sub, 0.25, SEED + 1))


def fit_pair(
    X: np.ndarray, y: np.ndarray, fit_idx: list[int], cal_idx: list[int]
) -> tuple[HistGradientBoostingClassifier, float]:
    """Train on the fit half and take the operating point from the calibration half."""
    model = new_model()
    model.fit(X[fit_idx], y[fit_idx])
    threshold = threshold_at_fpr(model.predict_proba(X[cal_idx])[:, 1], y[cal_idx], TARGET_FPR)
    return model, threshold


def run(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    fit_idx: list[int],
    cal_idx: list[int],
    test_idx: list[int],
) -> Result:
    model, threshold = fit_pair(X, y, fit_idx, cal_idx)
    return Result.measured(
        name,
        len(fit_idx),
        len(test_idx),
        float(y[test_idx].mean()),
        at_threshold(model.predict_proba(X[test_idx])[:, 1], y[test_idx], threshold),
    )


def load_dataset() -> tuple[list[Row], BrandIndex, dict[str, str]]:
    """Assemble the rows and the brand index exactly as the report does.

    The digest of each file is taken here, where the bytes are read, so the
    report cannot name a collection file that this run did not load.
    """
    print("loading collections...", file=sys.stderr)
    files = {
        "tranco": DATA / "tranco.csv",
        "majestic": DATA / "majestic.csv",
        "phishing_army": DATA / "army.txt",
        "openphish": DATA / "openphish.txt",
    }
    snapshot = {name: digest(path) for name, path in files.items()}
    tranco = load_tranco(files["tranco"])
    majestic = load_majestic(files["majestic"])
    positives = {
        "phishing_army": load_url_lines(files["phishing_army"]),
        "openphish": load_url_lines(files["openphish"]),
    }

    base_rows = build_dataset(
        positives,
        {"tranco": tranco},
        head_cut=HEAD_CUT,
        tail_cut=TAIL_CUT,
        pos_cap=12_000,
        neg_cap=60_000,
    )
    # Majestic is only used as a second collection of negatives for the
    # collection-shift test, so it never enters the main dataset and cannot move
    # the other rows.
    majestic_rows = negative_rows(
        majestic,
        "majestic",
        head_cut=HEAD_CUT,
        tail_cut=TAIL_CUT,
        cap=60_000,
        strata=("head",),
    )
    print("building brand index...", file=sys.stderr)
    brands = BrandIndex.from_tranco(files["tranco"], top=2000)
    return base_rows + majestic_rows, brands, snapshot


@dataclass(frozen=True)
class Experiment:
    """Everything the report renders and the artifact needs."""

    summary: dict[tuple[str, str], Counts]
    base: Result
    domain: Result
    popularity: Result
    collection_host: Result
    collection_full: Result
    rules: Result
    dropped: int
    n_base: int
    n_pos: int
    n_neg: int
    majestic: int
    shipped: HistGradientBoostingClassifier
    shipped_threshold: float
    trained_on: int
    calibrated_on: int

    @property
    def results(self) -> list[Result]:
        """The six rows in report order."""
        return [
            self.base,
            self.domain,
            self.popularity,
            self.collection_host,
            self.collection_full,
            self.rules,
        ]


def featurize_rows(rows: list[Row], brands: BrandIndex) -> tuple[np.ndarray, np.ndarray]:
    """Full-URL and host-only views of the same rows."""
    t0 = time.time()
    X = np.asarray([featurize(r.url, brands) for r in rows], dtype=np.float32)
    host_urls = [f"http://{host_of(r.url)}/" for r in rows]
    Xh = np.asarray([featurize(u, brands) for u in host_urls], dtype=np.float32)
    print(
        f"featurised {len(rows):,} rows twice in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    return X, Xh


def run_experiments(
    rows: list[Row], y: np.ndarray, X: np.ndarray, Xh: np.ndarray, brands: BrandIndex
) -> Experiment:
    base_idx = select(rows, collections={"phishing_army", "openphish", "tranco"})
    pos_idx = select(rows, label=1)
    head_idx = select(rows, collections={"tranco"}, stratum="head", label=0)
    tail_idx = select(rows, collections={"tranco"}, stratum="tail", label=0)
    army_idx = select(rows, collections={"phishing_army"})
    openphish_idx = select(rows, collections={"openphish"})
    majestic_idx = select(rows, collections={"majestic"}, stratum="head", label=0)

    def evaluated(
        name: str, features: np.ndarray, train: list[int], test: list[int], grouped: bool = True
    ) -> Result:
        """Calibrate on part of the training half, then score the test half."""
        fit, cal = calibration_split(rows, train, grouped)
        return run(name, features, y, fit, cal, test)

    def negative_shift() -> tuple[list[int], list[int]]:
        """Train against popular domains, test against the long tail."""
        train_p, test_p = split_indices(rows, pos_idx, lambda sub: group_split(sub, 0.2, SEED))
        return train_p + head_idx, test_p + tail_idx

    def collection_shift() -> tuple[list[int], list[int]]:
        """Vary the collection, hold the popularity stratum fixed.

        Positives move from one phishing collector to another, negatives from one
        popularity list to another, both in the head band. Domains present on
        both sides are dropped from the training half only: the test half holds
        the scarce positives, so discarding there would throw away the
        measurement.
        """
        train = army_idx + head_idx
        test = openphish_idx + majestic_idx
        shared = {rows[i].domain for i in train} & {rows[i].domain for i in test}
        return remove_domains(train, rows, shared), test

    r_train, r_test = split_indices(rows, base_idx, lambda sub: random_split(sub, 0.2, SEED))
    d_train, d_test = split_indices(rows, base_idx, lambda sub: group_split(sub, 0.2, SEED))
    s_train, s_test = negative_shift()
    c_train, c_test = collection_shift()

    print("  random rows...", file=sys.stderr)
    base = evaluated("random rows", X, r_train, r_test, grouped=False)
    print("  domain-disjoint...", file=sys.stderr)
    domain = evaluated("domain-disjoint", X, d_train, d_test)
    print("  popularity shift (head train, tail test)...", file=sys.stderr)
    popularity = evaluated("popularity shift (head train, tail test)", X, s_train, s_test)
    print("  collection shift, host-only...", file=sys.stderr)
    collection_host = evaluated("collection shift, host-only", Xh, c_train, c_test)
    print("  collection shift, as deployed (paths kept)...", file=sys.stderr)
    collection_full = evaluated("collection shift, as deployed (paths kept)", X, c_train, c_test)

    print("  impersonation rules...", file=sys.stderr)
    imp = np.asarray([impersonation_score(r.url, brands) for r in rows], dtype=float)
    thr_imp = threshold_at_fpr(imp[head_idx], y[head_idx], TARGET_FPR)
    rules = Result.measured(
        "impersonation rules only (no ML)",
        0,
        len(d_test),
        float(y[d_test].mean()),
        at_threshold(imp[d_test], y[d_test], thr_imp),
    )

    print("fitting the shipped model...", file=sys.stderr)
    # Same fit/calibration pair as the domain-disjoint row, so the artifact's
    # operating point is that row: measured on test rows the pair never saw.
    d_fit, d_cal = calibration_split(rows, d_train, True)
    shipped, shipped_threshold = fit_pair(X, y, d_fit, d_cal)

    base_pos = int(y[base_idx].sum())
    return Experiment(
        summary=describe(rows),
        base=base,
        domain=domain,
        popularity=popularity,
        collection_host=collection_host,
        collection_full=collection_full,
        rules=rules,
        dropped=len(army_idx) + len(head_idx) - len(c_train),
        n_base=len(base_idx),
        n_pos=base_pos,
        n_neg=len(base_idx) - base_pos,
        majestic=len(majestic_idx),
        shipped=shipped,
        shipped_threshold=shipped_threshold,
        trained_on=len(d_fit),
        calibrated_on=len(d_cal),
    )


def update_readme(results: str) -> None:
    """Generate the README table, so it cannot drift from the report."""
    text = README.read_text(encoding="utf-8")
    start = text.find(RESULTS_START)
    end = text.find(RESULTS_END)
    if start < 0 or end < start:
        raise ValueError(f"{README}: missing {RESULTS_START} / {RESULTS_END} markers")
    block = f"{RESULTS_START}\n{results}{RESULTS_END}"
    README.write_text(text[:start] + block + text[end + len(RESULTS_END) :], encoding="utf-8")


def main() -> int:
    rows, brands, snapshot = load_dataset()
    y = np.array([r.y for r in rows], dtype=int)
    X, Xh = featurize_rows(rows, brands)
    exp = run_experiments(rows, y, X, Xh, brands)

    # Artifact first: a failure here must not leave docs advertising a model
    # that was not written.
    artifact.write(
        exp.shipped,
        brands,
        exp.shipped_threshold,
        exp.trained_on,
        exp.calibrated_on,
    )

    rendered = report.render(exp, snapshot)
    (ROOT / "REPORT.md").write_text(rendered, encoding="utf-8")
    update_readme(table(exp.results))
    print("\n" + rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
