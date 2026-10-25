"""The model artifact: one owner for its path, shape, writer, and checks.

A silently stale model scores confidently and wrongly, so the bundle carries the
schema and feature names it was built with and `validate()` refuses a mismatch.
"""

from __future__ import annotations

import pickle
import sys
from typing import TypedDict

from sklearn.ensemble import HistGradientBoostingClassifier

from . import ROOT
from .features import ARTIFACT_SCHEMA, FEATURE_NAMES, BrandIndex

PATH = ROOT / "artifacts" / "model.pkl"


class Bundle(TypedDict):
    schema: int
    feature_names: list[str]
    model: HistGradientBoostingClassifier
    brands: BrandIndex
    threshold: float
    trained_on: int
    calibrated_on: int


def validate(bundle: Bundle) -> None:
    """Reject an artifact whose features do not match this code."""
    if bundle.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError(f"artifact schema {bundle.get('schema')!r}, expected {ARTIFACT_SCHEMA}")
    if bundle.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("artifact was built with different features")
    if bundle["model"].n_features_in_ != len(FEATURE_NAMES):
        raise ValueError("artifact model has a different feature count")


def load() -> Bundle:
    if not PATH.exists():
        print("no model yet - run `uv run urlish-benchmark` first", file=sys.stderr)
        raise SystemExit(1)
    with open(PATH, "rb") as fh:
        bundle = pickle.load(fh)
    try:
        validate(bundle)
    except ValueError as exc:
        print(f"{PATH}: {exc}; rerun `uv run urlish-benchmark`", file=sys.stderr)
        raise SystemExit(2) from exc
    return bundle


def write(
    model: HistGradientBoostingClassifier,
    brands: BrandIndex,
    threshold: float,
    trained_on: int,
    calibrated_on: int,
) -> None:
    """Write the bundle, stamping the contract so a reader can check it."""
    bundle: Bundle = {
        "schema": ARTIFACT_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "model": model,
        "brands": brands,
        "threshold": threshold,
        "trained_on": trained_on,
        "calibrated_on": calibrated_on,
    }
    PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PATH, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"artifacts written to {PATH.parent}", file=sys.stderr)
