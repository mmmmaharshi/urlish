"""The operating point in the artifact must belong to the model in the artifact.

Run after `uv run urlish-benchmark`; needs data/ and artifacts/model.pkl.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from urlish import artifact
from urlish.benchmark import (
    DATA,
    SEED,
    TARGET_FPR,
    calibration_split,
    load_dataset,
    split_indices,
)
from urlish.data import group_split, select
from urlish.features import FEATURE_NAMES, featurize
from urlish.metrics import at_threshold, threshold_at_fpr

pytestmark = pytest.mark.skipif(
    not (DATA / "tranco.csv").exists() or not artifact.PATH.exists(),
    reason="needs data/ and artifacts/model.pkl; run `uv run urlish-benchmark` first",
)


def shipped_split(rows, brands):
    """Re-derive the split the benchmark used for the artifact."""
    base_idx = select(rows, collections={"phreshphish"})
    train, test = split_indices(rows, base_idx, lambda sub: group_split(sub, 0.2, SEED))
    fit, cal = calibration_split(rows, train, True)
    return fit, cal, test


def score(rows, brands, idx):
    X = np.asarray([featurize(rows[i].url, brands) for i in idx], dtype=np.float32)
    y = np.array([rows[i].y for i in idx], dtype=int)
    return X, y


def test_threshold_is_calibrated_on_the_shipped_model() -> None:
    bundle = artifact.load()
    rows, brands, _ = load_dataset()
    fit, cal, test = shipped_split(rows, brands)

    X_cal, y_cal = score(rows, brands, cal)
    scores = bundle["model"].predict_proba(X_cal)[:, 1]
    realised = at_threshold(scores, y_cal, bundle["threshold"])
    assert realised["fpr"] <= TARGET_FPR, realised

    # Recalibrating the stored model on the same held-out quarter must return
    # the stored number. A threshold taken from a probe model does not.
    assert bundle["threshold"] == pytest.approx(threshold_at_fpr(scores, y_cal, TARGET_FPR))
    assert bundle["trained_on"] == len(fit)
    assert bundle["calibrated_on"] == len(cal)

    # The reported operating point is measured on rows neither half saw. The
    # budget is spent on the calibration set, so a different distribution may
    # spend more; the report documents that rather than asserting it here.
    X_test, y_test = score(rows, brands, test)
    held_out = at_threshold(
        bundle["model"].predict_proba(X_test)[:, 1], y_test, bundle["threshold"]
    )
    assert held_out["fpr"] <= 0.05, held_out


def test_test_half_is_domain_disjoint_from_the_shipped_model() -> None:
    rows, brands, _ = load_dataset()
    fit, cal, test = shipped_split(rows, brands)
    seen = {rows[i].domain for i in fit + cal}
    assert not seen & {rows[i].domain for i in test}


def test_rejects_a_bundle_built_for_other_features() -> None:
    bundle = artifact.load()
    mutations: list[dict[str, object]] = [
        {"schema": 99},
        {"feature_names": bundle["feature_names"][:-1]},
        {"feature_names": list(reversed(bundle["feature_names"]))},
        {"feature_names": None},
        {"model": SimpleNamespace(n_features_in_=len(FEATURE_NAMES) - 1)},
    ]
    for mutation in mutations:
        bad = {**bundle, **mutation}
        with pytest.raises(ValueError):
            artifact.validate(bad)


def test_load_exits_with_the_rerun_hint(tmp_path, monkeypatch, capsys) -> None:
    bundle = artifact.load()
    bad = tmp_path / "model.pkl"
    with open(bad, "wb") as fh:
        pickle.dump({**bundle, "schema": 99}, fh)
    monkeypatch.setattr(artifact, "PATH", bad)

    with pytest.raises(SystemExit) as exit_info:
        artifact.load()
    assert exit_info.value.code == 2
    assert "rerun `uv run urlish-benchmark`" in capsys.readouterr().err
