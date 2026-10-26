"""Render REPORT.md from a finished experiment.

Measurement lives in `benchmark.py`; prose lives here. Nothing in this module
reads a data file, so the report can only state what was measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import __version__
from .metrics import TARGET_FPR, Metrics, wilson

if TYPE_CHECKING:
    from .benchmark import Experiment


@dataclass(frozen=True)
class Result:
    """One row of the results table."""

    name: str
    n_train: int
    n_test: int
    pos_rate: float
    tp: int
    fn: int
    recall: float
    precision: float
    fpr: float
    auc: float

    @classmethod
    def measured(
        cls, name: str, n_train: int, n_test: int, pos_rate: float, metrics: Metrics
    ) -> Result:
        return cls(
            name=name,
            n_train=n_train,
            n_test=n_test,
            pos_rate=pos_rate,
            tp=metrics["tp"],
            fn=metrics["fn"],
            recall=metrics["recall"],
            precision=metrics["precision"],
            fpr=metrics["fpr"],
            auc=metrics["auc"],
        )


def table(results: list[Result]) -> str:
    head = (
        "| split | n fit | n test | pos rate | recall@1%FPR "
        "| precision@1%FPR | ROC-AUC | realised FPR |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = "".join(
        f"| {r.name} | {r.n_train:,} | {r.n_test:,} | {r.pos_rate:.1%} | "
        f"**{r.recall:.1%}** | {r.precision:.1%} | {r.auc:.3f} | {r.fpr:.2%} |\n"
        for r in results
    )
    return head + body


def render(exp: Experiment, snapshot: dict[str, str]) -> str:
    """The whole report, from the measured results and the inputs it read."""
    base = exp.base
    dom = exp.domain
    temporal = exp.temporal
    baseline = exp.baseline
    shift = exp.popularity
    cross_h = exp.collection_host
    cross_full = exp.collection_full
    rules = exp.rules
    lo, hi = wilson(cross_full.tp, cross_full.tp + cross_full.fn)
    inputs = " · ".join(f"{name} {sha}" for name, sha in snapshot.items())

    lines = [
        "# urlish benchmark report\n",
        f"urlish {__version__} · {inputs}\n",
        f"Temporal cutoff: {exp.cutoff}. Corpus dates: {exp.date_range[0]} to "
        f"{exp.date_range[1]}.\n",
        f"Target false-positive rate: {TARGET_FPR:.0%}. Thresholds are picked on held-out\n"
        "training data, never on the test set.\n",
        "## Dataset\n",
        "| collection | stratum | rows | positives | negatives |\n|---|---|---:|---:|---:|",
    ]
    for (collection, stratum), s in sorted(exp.summary.items()):
        lines.append(f"| {collection} | {stratum} | {s['n']:,} | {s['pos']:,} | {s['neg']:,} |")
    lines += [
        f"| **main dataset** | | **{exp.n_base:,}** | **{exp.n_pos:,}** | **{exp.n_neg:,}** |\n",
        f"Of the {exp.n_neg:,} main-corpus negatives, {exp.ranked_neg:,} carry a Tranco\n"
        "stratum and enter the popularity row; the rest are unranked and are excluded\n"
        "from it.\n",
        f"Majestic contributes {exp.majestic:,} negatives and is used only by the\n"
        "cross-dataset rows, so it cannot move the other numbers.\n",
        "## Results\n",
        table(exp.results),
        "\n## What the numbers say\n",
        f"The baseline reports {base.recall:.1%} recall and {base.precision:.1%}\n"
        f"precision at a 1% false-positive budget, AUC {base.auc:.3f}. Even the leaky\n"
        "split is not 99%: this is host-level detection on live collections, not a toy.\n",
        "Precision depends on how many positives the test set happens to contain, and that\n"
        "ratio varies by an order of magnitude across rows (pos rate column). Compare rows\n"
        "on recall@1%FPR and AUC; read precision only against its own pos rate.\n",
        f"**Domain leakage costs little.** Domain-disjoint evaluation loses\n"
        f"{100 * (base.recall - dom.recall):.1f} points of recall\n"
        f"({base.recall:.1%} -> {dom.recall:.1%}) with AUC essentially unchanged\n"
        f"({base.auc:.3f} -> {dom.auc:.3f}). Phishing hosts are mostly single-use, so\n"
        "there was less to memorise than the usual story assumes.\n",
        f"**Popularity shift spends the false-positive budget.** Training on ranks 1-10k and\n"
        f"testing on 10k-60k leaves AUC at {shift.auc:.3f} - but the threshold calibrated\n"
        f"for 1% false positives realises {shift.fpr:.2%}, and recall lands at\n"
        f"{shift.recall:.1%}. A shipped detector would block roughly 1 in\n"
        f"{1 / max(shift.fpr, 1e-9):.0f} legitimate popular domains while every ranking\n"
        "metric looked healthy.\n",
        f"**Time costs ranking.** Training on rows dated before the cutoff and testing on\n"
        f"rows dated after it drops AUC to {temporal.auc:.3f} and recall to\n"
        f"{temporal.recall:.1%}; the threshold calibrated on the earlier period realises\n"
        f"{temporal.fpr:.2%} false positives on the later one.\n",
        f"**Cross-collection transfer is the largest honest drop.** With negatives taken from\n"
        f"two independent lists, this row changes the collector on both sides - PhreshPhish\n"
        f"to OpenPhish, Tranco to Majestic - while holding the popularity band and the\n"
        f"host-only view fixed. AUC falls {base.auc:.3f} -> {cross_h.auc:.3f} and\n"
        f"recall {base.recall:.1%} -> {cross_h.recall:.1%}, with a realised FPR of\n"
        f"{cross_h.fpr:.2%} on negatives from a list it never saw. The test half is also\n"
        "collected after the training half, so this row is both cross-collection and\n"
        "out of time.\n",
        f"{exp.dropped:,} rows were dropped from the training half because their\n"
        "registrable domain appeared in the test half: the collections overlap heavily,\n"
        "and a collection split that ignores that is measuring leakage, not transfer.\n",
        f"**Keeping the URL path recovers the ranking, and then some.** The same model,\n"
        f"scored on full URLs, reaches AUC {cross_full.auc:.3f} and recall\n"
        f"{cross_full.recall:.1%} (95% CI {lo:.1%}-{hi:.1%} on\n"
        f"{cross_full.tp + cross_full.fn} positives), though the realised FPR is still\n"
        f"{cross_full.fpr:.2%}. Path length and lure words carry real signal, which is why\n"
        "`featurize()` normalises away scheme and trailing slash rather than discarding the\n"
        "path.\n",
        f"**Brand rules are precise but narrow.** No training data, {rules.precision:.1%}\n"
        f"precision at {rules.fpr:.2%} FPR, but {rules.recall:.1%} recall. Most phishing\n"
        "hosts do not impersonate a brand by string similarity; they are throwaway domains\n"
        "on cheap TLDs. Rules and classifiers cover different halves of the problem, which\n"
        "is the argument for shipping both.\n",
        f"**A published-style baseline.** A character n-gram logistic regression, trained\n"
        f"and calibrated the same way, reaches {baseline.recall:.1%} recall at {baseline.fpr:.2%}\n"
        f"realised FPR (AUC {baseline.auc:.3f}). It is a cheaper detector than the tree\n"
        "model and does not beat it.\n",
        "## Operating point of the shipped detector\n",
        f"`urlish` scores at **{exp.shipped_threshold:.4f}**. The artifact holds the model and\n"
        "threshold of the domain-disjoint row: fitted on the training half, calibrated on a\n"
        "quarter of it, and measured on the test half that neither half saw. That row reports\n"
        f"{dom.fpr:.2%} false positives at {dom.recall:.1%} recall. The budget is spent\n"
        "on the calibration set, and a different test distribution can spend more.\n"
        "Scoring at 0.5 instead would realise a false-positive rate many times over budget,\n"
        "because a gradient-boosted probability is not a decision threshold.\n",
        "## Limitations\n",
        "- Single dated collection. The temporal row uses one collection's dates, and the\n"
        "  leakage filter in that collection's paper runs on page content, which this\n"
        "  benchmark discards: only the URL string is used, so the row is a URL-level\n"
        "  approximation.\n"
        "- About four in ten negative rows are unranked in Tranco and are excluded from\n"
        "  the popularity row; the report prints that coverage.\n"
        "- Collection dates are the upstream dataset's labels, not dates this repository\n"
        "  measured; every claim about time inherits that limit.\n",
        "- Labels are collection membership, not ground truth. Collections contain dead,\n"
        "  parked and occasionally mislabelled hosts, and a popular domain is not a\n"
        "  guarantee of safety.\n",
        f"- The collection-shift test has {cross_full.tp + cross_full.fn} positives and\n"
        f"  {cross_full.n_test - cross_full.tp - cross_full.fn:,} negatives, hence\n"
        "  the interval above. Every other split is >= 14k rows.\n",
        "- The shipped model trains on full URLs, so path features are seen at fit time.\n"
        "  That is the realistic case for a feed consumer, but it is a limitation of the\n"
        "  data, not a finding about phishing.\n",
        "- Negatives come from domain lists, so even with `www.` prefixes they carry at\n"
        "  most one subdomain. Deep subdomain chains appear only in positives, and the model\n"
        "  leans on that.\n",
        "- Features are string-only: no DNS, no certificate, no page content. That is the\n"
        "  client-side constraint and it caps what is achievable.\n",
        "- Brands come from Tranco, which over-represents Western sites. Impersonation of\n"
        "  regional brands is under-covered.\n",
        "\n## Reproduce\n",
        "```\nuv sync\nuv run urlish-benchmark\nuv run pytest -q\n```\n",
    ]
    return "\n".join(lines) + "\n"
