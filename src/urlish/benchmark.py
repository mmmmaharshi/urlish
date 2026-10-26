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
from .core import host_of, registrable_domain
from .data import (
    Counts,
    Row,
    dedupe,
    describe,
    digest,
    group_split,
    load_majestic,
    load_phreshphish,
    load_tranco,
    load_url_lines,
    negative_rows,
    random_split,
    remove_domains,
    select,
    time_split,
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
        "phreshphish": DATA / "phreshphish.csv",
        "tranco": DATA / "tranco.csv",
        "majestic": DATA / "majestic.csv",
        "openphish": DATA / "openphish.txt",
    }
    snapshot = {name: digest(path) for name, path in files.items()}
    tranco = load_tranco(files["tranco"])
    tranco_rank = {domain.lower(): rank for rank, domain in tranco}
    majestic = load_majestic(files["majestic"])
    openphish_urls = load_url_lines(files["openphish"])

    raw = load_phreshphish(files["phreshphish"])
    pos_all = [r for r in raw if r.y == 1]
    neg_all = [r for r in raw if r.y == 0]
    pos_step = max(1, len(pos_all) // 12_000)
    neg_step = max(1, len(neg_all) // 60_000)
    sampled_pos = pos_all[::pos_step][:12_000]
    sampled_neg = neg_all[::neg_step][:60_000]

    def stratum_for(domain: str) -> str | None:
        rank = tranco_rank.get(domain.lower())
        if rank is None:
            return None
        if rank <= HEAD_CUT:
            return "head"
        if rank <= TAIL_CUT:
            return "tail"
        return None

    sampled_neg = [
        Row(r.url, r.y, r.domain, r.collection, stratum_for(r.domain), r.date) for r in sampled_neg
    ]
    ranked = sum(1 for r in sampled_neg if r.stratum is not None)
    print(
        f"phreshphish: {len(sampled_pos):,} positives, {len(sampled_neg):,} negatives "
        f"({ranked:,} ranked in Tranco)",
        file=sys.stderr,
    )
    base_rows = dedupe(sampled_pos + sampled_neg)
    # shortcut: stride sampling over file order keeps every month roughly represented
    # but is not stratified by date; replace with per-month sampling if the
    # temporal row looks lopsided.

    openphish_step = max(1, len(openphish_urls) // 12_000)
    openphish_rows = [
        Row(url, 1, registrable_domain(host_of(url)), "openphish", None)
        for url in openphish_urls[::openphish_step][:12_000]
    ]

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
    return base_rows + openphish_rows + majestic_rows, brands, snapshot


@dataclass(frozen=True)
class Experiment:
    """Everything the report renders and the artifact needs."""

    summary: dict[tuple[str, str], Counts]
    base: Result
    domain: Result
    temporal: Result
    popularity: Result
    cutoff: str
    date_range: tuple[str, str]
    ranked_neg: int
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
            self.temporal,
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
    base_idx = select(rows, collections={"phreshphish"})
    pos_idx = select(rows, label=1, collections={"phreshphish"})
    head_idx = select(rows, collections={"phreshphish"}, stratum="head", label=0)
    tail_idx = select(rows, collections={"phreshphish"}, stratum="tail", label=0)
    openphish_idx = select(rows, collections={"openphish"})
    majestic_idx = select(rows, collections={"majestic"}, stratum="head", label=0)

    phish_dates = sorted(r.date for r in rows if r.collection == "phreshphish" and r.date)
    # 80th percentile row's month, as a date: the cutoff is the first day of that month.
    idx = int(0.8 * len(phish_dates))
    cutoff = phish_dates[idx][:7] + "-01"
    date_range = (phish_dates[0], phish_dates[-1])
    print(f"  temporal cutoff: {cutoff}", file=sys.stderr)

    def evaluated(
        name: str, features: np.ndarray, train: list[int], test: list[int], grouped: bool = True
    ) -> Result:
        fit, cal = calibration_split(rows, train, grouped)
        return run(name, features, y, fit, cal, test)

    def negative_shift() -> tuple[list[int], list[int]]:
        train_p, test_p = split_indices(rows, pos_idx, lambda sub: group_split(sub, 0.2, SEED))
        return train_p + head_idx, test_p + tail_idx

    def cross_dataset() -> tuple[list[int], list[int]]:
        train = base_idx
        test = openphish_idx + majestic_idx
        shared = {rows[i].domain for i in train} & {rows[i].domain for i in test}
        return remove_domains(train, rows, shared), test

    r_train, r_test = split_indices(rows, base_idx, lambda sub: random_split(sub, 0.2, SEED))
    d_train, d_test = split_indices(rows, base_idx, lambda sub: group_split(sub, 0.2, SEED))
    t_train, t_test = split_indices(rows, base_idx, lambda sub: time_split(sub, cutoff))
    s_train, s_test = negative_shift()
    x_train, x_test = cross_dataset()

    print("  random rows...", file=sys.stderr)
    base = evaluated("random rows", X, r_train, r_test, grouped=False)
    print("  domain-disjoint...", file=sys.stderr)
    domain = evaluated("domain-disjoint", X, d_train, d_test)
    print("  temporal split...", file=sys.stderr)
    temporal = evaluated("temporal split", X, t_train, t_test)
    print("  popularity shift (head train, tail test)...", file=sys.stderr)
    popularity = evaluated("popularity shift (head train, tail test)", X, s_train, s_test)
    print("  cross-dataset, host-only...", file=sys.stderr)
    collection_host = evaluated("cross-dataset, host-only", Xh, x_train, x_test)
    print("  cross-dataset, as deployed (paths kept)...", file=sys.stderr)
    collection_full = evaluated("cross-dataset, as deployed (paths kept)", X, x_train, x_test)

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
    d_fit, d_cal = calibration_split(rows, d_train, True)
    shipped, shipped_threshold = fit_pair(X, y, d_fit, d_cal)

    base_pos = int(y[base_idx].sum())
    cross_dropped = len(base_idx) - len(x_train)
    ranked_neg = sum(1 for i in base_idx if rows[i].y == 0 and rows[i].stratum is not None)
    return Experiment(
        summary=describe(rows),
        base=base,
        domain=domain,
        temporal=temporal,
        popularity=popularity,
        collection_host=collection_host,
        collection_full=collection_full,
        rules=rules,
        dropped=cross_dropped,
        n_base=len(base_idx),
        n_pos=base_pos,
        n_neg=len(base_idx) - base_pos,
        majestic=len(majestic_idx),
        cutoff=cutoff,
        date_range=date_range,
        ranked_neg=ranked_neg,
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
