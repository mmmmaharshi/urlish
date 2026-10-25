"""The operating point in the artifact must belong to the model in the artifact.

Run after `uv run urlish-benchmark`; needs data/ and artifacts/model.pkl.
"""

from __future__ import annotations

import numpy as np
import pytest

from urlish.benchmark import DATA, TARGET_FPR, calibration_split, load_dataset
from urlish.cli import ARTIFACTS, load, validate
from urlish.data import select
from urlish.features import featurize
from urlish.metrics import at_threshold, threshold_at_fpr

pytestmark = pytest.mark.skipif(
    not (DATA / "tranco.csv").exists() or not ARTIFACTS.exists(),
    reason="needs data/ and artifacts/model.pkl; run `uv run urlish-benchmark` first",
)


def test_threshold_is_calibrated_on_the_shipped_model() -> None:
    bundle = load()
    rows, brands = load_dataset()
    base_idx = select(rows, collections={"phishing_army", "openphish", "tranco"})
    _, cal = calibration_split(rows, base_idx, True)
    X = np.asarray([featurize(rows[i].url, brands) for i in cal], dtype=np.float32)
    y = np.array([rows[i].y for i in cal], dtype=int)

    scores = bundle["model"].predict_proba(X)[:, 1]
    realised = at_threshold(scores, y, bundle["threshold"])
    assert realised["fpr"] <= TARGET_FPR, realised

    # Recalibrating the stored model on the same held-out quarter must return
    # the stored number. A threshold taken from a probe model does not.
    assert bundle["threshold"] == pytest.approx(threshold_at_fpr(scores, y, TARGET_FPR))


def test_rejects_a_bundle_built_for_other_features() -> None:
    bundle = load()
    mutations: list[dict[str, object]] = [
        {"schema": 99},
        {"feature_names": bundle["feature_names"][:-1]},
        {"feature_names": list(reversed(bundle["feature_names"]))},
        {"feature_names": None},
    ]
    for mutation in mutations:
        bad = {**bundle, **mutation}
        with pytest.raises(ValueError):
            validate(bad)
