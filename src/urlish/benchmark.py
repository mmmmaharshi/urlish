"""Run six evaluations of the same features and write REPORT.md.

The point is not the model. It is the gap between the first row and the others.
"""

from __future__ import annotations

import math
import pickle
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .core import host_of
from .data import (
    Row,
    build_dataset,
    describe,
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
from .metrics import at_threshold, threshold_at_fpr

# Installed editable from src/, so the repo root is two levels above this file.
ROOT = Path(__file__).resolve().parents[2]

SEED = 0
TARGET_FPR = 0.01
HEAD_CUT = 10_000
TAIL_CUT = 60_000
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval - honest error bars on the 300-URL test set."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


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


def calibration_split(
    rows: list[Row], train_idx: list[int], grouped: bool
) -> tuple[list[int], list[int]]:
    """Threshold is chosen on held-out training data, never on the test set."""
    sub = [rows[i] for i in train_idx]
    split = group_split if grouped else random_split
    fit, cal = split(sub, 0.25, SEED + 1)
    return [train_idx[i] for i in fit], [train_idx[i] for i in cal]


def run(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    fit_idx: list[int],
    cal_idx: list[int],
    test_idx: list[int],
) -> dict[str, Any]:
    model = new_model()
    model.fit(X[fit_idx], y[fit_idx])
    threshold = threshold_at_fpr(model.predict_proba(X[cal_idx])[:, 1], y[cal_idx], TARGET_FPR)
    result: dict[str, Any] = at_threshold(
        model.predict_proba(X[test_idx])[:, 1], y[test_idx], threshold
    )
    result.update(
        name=name,
        n_train=len(fit_idx),
        n_test=len(test_idx),
        threshold=threshold,
        pos_rate=float(y[test_idx].mean()),
    )
    return result


def table(results: list[dict[str, Any]]) -> str:
    head = (
        "| split | n fit | n test | pos rate | recall@1%FPR "
        "| precision@1%FPR | ROC-AUC | realised FPR |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = "".join(
        f"| {r['name']} | {r['n_train']:,} | {r['n_test']:,} | {r['pos_rate']:.1%} | "
        f"**{r['recall']:.1%}** | {r['precision']:.1%} | {r['auc']:.3f} | {r['fpr']:.2%} |\n"
        for r in results
    )
    return head + body


def main() -> int:
    ARTIFACTS.mkdir(exist_ok=True)

    print("loading collections...", file=sys.stderr)
    tranco = load_tranco(DATA / "tranco.csv")
    majestic = load_majestic(DATA / "majestic.csv")
    positives = {
        "phishing_army": load_url_lines(DATA / "army.txt"),
        "openphish": load_url_lines(DATA / "openphish.txt"),
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
    rows = base_rows + majestic_rows
    y = np.array([r.y for r in rows], dtype=int)

    print("building brand index...", file=sys.stderr)
    brands = BrandIndex.from_tranco(DATA / "tranco.csv", top=2000)

    t0 = time.time()
    X = np.asarray([featurize(r.url, brands) for r in rows], dtype=np.float32)
    host_urls = [f"http://{host_of(r.url)}/" for r in rows]
    Xh = np.asarray([featurize(u, brands) for u in host_urls], dtype=np.float32)
    print(
        f"featurised {len(rows):,} rows twice in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )

    base_idx = select(rows, collections={"phishing_army", "openphish", "tranco"})
    pos_idx = select(rows, label=1)
    head_idx = select(rows, collections={"tranco"}, stratum="head", label=0)
    tail_idx = select(rows, collections={"tranco"}, stratum="tail", label=0)
    army_idx = select(rows, collections={"phishing_army"})
    openphish_idx = select(rows, collections={"openphish"})
    majestic_idx = select(rows, collections={"majestic"}, stratum="head", label=0)

    def split_base(
        fn: Callable[[list[Row]], tuple[list[int], list[int]]],
    ) -> tuple[list[int], list[int]]:
        """Run a split over the main dataset and map indices back to `rows`."""
        sub = [rows[i] for i in base_idx]
        train, test = fn(sub)
        return [base_idx[i] for i in train], [base_idx[i] for i in test]

    def negative_shift() -> tuple[list[int], list[int]]:
        """Train against popular domains, test against the long tail."""
        sub = [rows[i] for i in pos_idx]
        train_p, test_p = group_split(sub, 0.2, SEED)
        return [pos_idx[i] for i in train_p] + head_idx, [pos_idx[i] for i in test_p] + tail_idx

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

    print("  random rows (the usual baseline)...", file=sys.stderr)
    r_train, r_test = split_base(lambda s: random_split(s, 0.2, SEED))
    r_fit, r_cal = calibration_split(rows, r_train, False)
    results = [run("random rows (the usual baseline)", X, y, r_fit, r_cal, r_test)]

    print("  domain-disjoint...", file=sys.stderr)
    d_train, d_test = split_base(lambda s: group_split(s, 0.2, SEED))
    d_fit, d_cal = calibration_split(rows, d_train, True)
    results.append(run("domain-disjoint", X, y, d_fit, d_cal, d_test))

    print("  popularity shift (head train, tail test)...", file=sys.stderr)
    s_train, s_test = negative_shift()
    s_fit, s_cal = calibration_split(rows, s_train, True)
    results.append(run("popularity shift (head train, tail test)", X, y, s_fit, s_cal, s_test))

    print("  collection shift, host-only...", file=sys.stderr)
    c_train, c_test = collection_shift()
    c_fit, c_cal = calibration_split(rows, c_train, True)
    results.append(run("collection shift, host-only", Xh, y, c_fit, c_cal, c_test))
    print("  collection shift, as deployed (paths kept)...", file=sys.stderr)
    results.append(run("collection shift, as deployed (paths kept)", X, y, c_fit, c_cal, c_test))

    print("  impersonation rules...", file=sys.stderr)
    imp = np.asarray([impersonation_score(r.url, brands) for r in rows], dtype=float)
    thr_imp = threshold_at_fpr(imp[head_idx], y[head_idx], TARGET_FPR)
    imp_res: dict[str, Any] = at_threshold(imp[d_test], y[d_test], thr_imp)
    imp_res.update(
        name="impersonation rules only (no ML)",
        n_train=0,
        n_test=len(d_test),
        threshold=thr_imp,
        pos_rate=float(y[d_test].mean()),
    )
    results.append(imp_res)

    by = {r["name"]: r for r in results}
    base = by["random rows (the usual baseline)"]
    dom = by["domain-disjoint"]
    shift = by["popularity shift (head train, tail test)"]
    cross_h = by["collection shift, host-only"]
    cross_full = by["collection shift, as deployed (paths kept)"]
    rules = by["impersonation rules only (no ML)"]
    lo, hi = wilson(cross_full["tp"], cross_full["tp"] + cross_full["fn"])
    dropped = len(army_idx) + len(head_idx) - len(c_train)

    print("calibrating the shipped model...", file=sys.stderr)
    ship_fit, ship_cal = calibration_split(rows, base_idx, True)
    probe = new_model()
    probe.fit(X[ship_fit], y[ship_fit])
    shipped_threshold = threshold_at_fpr(
        probe.predict_proba(X[ship_cal])[:, 1], y[ship_cal], TARGET_FPR
    )

    summary = describe(rows)
    lines = [
        "# urlish benchmark report\n",
        f"Target false-positive rate: {TARGET_FPR:.0%}. Thresholds are picked on held-out\n"
        "training data, never on the test set.\n",
        "## Dataset\n",
        "| collection | stratum | rows | positives | negatives |\n|---|---|---:|---:|---:|",
    ]
    for (collection, stratum), s in sorted(summary.items()):
        lines.append(f"| {collection} | {stratum} | {s['n']:,} | {s['pos']:,} | {s['neg']:,} |")
    in_base = len(base_idx)
    lines += [
        f"| **main dataset** | | **{in_base:,}** | **{int(y[base_idx].sum()):,}** | "
        f"**{int((1 - y[base_idx]).sum()):,}** |\n",
        f"Majestic contributes {len(majestic_idx):,} negatives and is used only by the\n"
        "collection-shift rows, so it cannot move the other numbers.\n",
        "## Results\n",
        table(results),
        "\n## What the numbers say\n",
        f"The baseline reports {base['recall']:.1%} recall and {base['precision']:.1%}\n"
        f"precision at a 1% false-positive budget, AUC {base['auc']:.3f}. Even the leaky\n"
        "split is not 99%: this is host-level detection on live collections, not a toy.\n",
        "Precision depends on how many positives the test set happens to contain, and that\n"
        "ratio varies by an order of magnitude across rows (pos rate column). Compare rows\n"
        "on recall@1%FPR and AUC; read precision only against its own pos rate.\n",
        f"**Domain leakage costs little.** Domain-disjoint evaluation loses\n"
        f"{100 * (base['recall'] - dom['recall']):.1f} points of recall\n"
        f"({base['recall']:.1%} -> {dom['recall']:.1%}) with AUC essentially unchanged\n"
        f"({base['auc']:.3f} -> {dom['auc']:.3f}). Phishing hosts are mostly single-use, so\n"
        "there was less to memorise than the usual story assumes.\n",
        f"**Popularity shift spends the false-positive budget.** Training on ranks 1-10k and\n"
        f"testing on 10k-60k leaves AUC at {shift['auc']:.3f} - but the threshold calibrated\n"
        f"for 1% false positives realises {shift['fpr']:.2%}, and recall lands at\n"
        f"{shift['recall']:.1%}. A shipped detector would block roughly 1 in\n"
        f"{1 / max(shift['fpr'], 1e-9):.0f} legitimate popular domains while every ranking\n"
        "metric looked healthy.\n",
        f"**Collection shift is the largest honest drop.** With negatives taken from\n"
        f"two independent lists, this row changes the collector on both sides - Phishing\n"
        f"Army to OpenPhish, Tranco to Majestic - while holding the popularity band and the\n"
        f"host-only view fixed. AUC falls {base['auc']:.3f} -> {cross_h['auc']:.3f} and\n"
        f"recall {base['recall']:.1%} -> {cross_h['recall']:.1%}, with a realised FPR of\n"
        f"{cross_h['fpr']:.2%} on negatives from a list it never saw.\n",
        f"{dropped:,} rows were dropped from the training half because their\n"
        "registrable domain appeared in the test half: the collections overlap heavily,\n"
        "and a collection split that ignores that is measuring leakage, not transfer.\n",
        f"**Keeping the URL path recovers the ranking, and then some.** The same model,\n"
        f"scored on full URLs, reaches AUC {cross_full['auc']:.3f} and recall\n"
        f"{cross_full['recall']:.1%} (95% CI {lo:.1%}-{hi:.1%} on\n"
        f"{cross_full['tp'] + cross_full['fn']} positives), though the realised FPR is still\n"
        f"{cross_full['fpr']:.2%}. Path length and lure words carry real signal, which is why\n"
        "`featurize()` normalises away scheme and trailing slash rather than discarding the\n"
        "path.\n",
        f"**Brand rules are precise but narrow.** No training data, {rules['precision']:.1%}\n"
        f"precision at {rules['fpr']:.2%} FPR, but {rules['recall']:.1%} recall. Most phishing\n"
        "hosts do not impersonate a brand by string similarity; they are throwaway domains\n"
        "on cheap TLDs. Rules and classifiers cover different halves of the problem, which\n"
        "is the argument for shipping both.\n",
        "## Operating point of the shipped detector\n",
        f"`predict.py` scores at **{shipped_threshold:.4f}**, calibrated for a 1% budget on a\n"
        "domain-disjoint quarter of the training rows and stored in the model artifact.\n"
        "Scoring at 0.5 instead would realise a false-positive rate an order of magnitude\n"
        "over budget, because a gradient-boosted probability is not a decision threshold.\n",
        "## Limitations\n",
        "- Single snapshot. No dates in these collections, so there is no temporal split. A\n"
        "  temporal holdout is the missing experiment and is strictly harder than any row\n"
        "  above.\n",
        "- Labels are collection membership, not ground truth. Collections contain dead,\n"
        "  parked and occasionally mislabelled hosts, and a popular domain is not a\n"
        "  guarantee of safety.\n",
        f"- The collection-shift test has {cross_full['tp'] + cross_full['fn']} positives and\n"
        f"  {cross_full['n_test'] - cross_full['tp'] - cross_full['fn']:,} negatives, hence\n"
        "  the interval above. Every other split is >= 14k rows.\n",
        "- The training collection (`phishing_army`) is host-only, so path features are\n"
        "  unseen at fit time. That is the realistic case for a collection consumer, not a\n"
        "  contrived one, but it is a limitation of the data, not a finding about phishing.\n",
        "- Negatives come from domain lists, so even with `www.` prefixes they carry at\n"
        "  most one subdomain. Deep subdomain chains appear only in positives, and the model\n"
        "  leans on that.\n",
        "- Features are string-only: no DNS, no certificate, no page content. That is the\n"
        "  client-side constraint and it caps what is achievable.\n",
        "- Brands come from Tranco, which over-represents Western sites. Impersonation of\n"
        "  regional brands is under-covered.\n",
        "\n## Reproduce\n",
        "```\nuv sync\nuv run urlish-benchmark\nuv run python tests/test_selfcheck.py\n```\n",
    ]

    report = "\n".join(lines) + "\n"
    (ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    print("\n" + report)

    print("fitting final model...", file=sys.stderr)
    final = new_model()
    final.fit(X[base_idx], y[base_idx])
    with open(ARTIFACTS / "model.pkl", "wb") as fh:
        pickle.dump(
            {
                "model": final,
                "brands": brands,
                "threshold": shipped_threshold,
                "trained_on": in_base,
            },
            fh,
        )
    print(f"artifacts written to {ARTIFACTS}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
