"""Score URLs from the command line.

urlish https://paypa1.com/login
urlish --train          # rebuild the model artifact first
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

from .features import analyze, featurize

# Installed editable from src/, so the repo root is two levels above this file.
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "model.pkl"


def load() -> dict[str, Any]:
    if not ARTIFACTS.exists():
        print("no model yet - run `uv run urlish-benchmark` first", file=sys.stderr)
        raise SystemExit(1)
    with open(ARTIFACTS, "rb") as fh:
        return pickle.load(fh)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--train" in argv:
        from .benchmark import main as benchmark_main

        return benchmark_main()

    urls = [a for a in argv if not a.startswith("--")]
    if not urls:
        print(__doc__)
        return 2

    bundle = load()
    model, brands = bundle["model"], bundle["brands"]
    threshold = float(bundle["threshold"])

    for url in urls:
        prob = float(model.predict_proba([featurize(url, brands)])[0][1])
        matches, _ = analyze(url, brands)
        print(f"\n{url}")
        print(
            f"  classifier   {'phishing' if prob >= threshold else 'not phishing'}"
            f"  (p={prob:.3f}, threshold {threshold:.3f})"
        )
        if matches:
            for m in matches:
                print(
                    f"  impersonates {m.brand} ({m.primary}) - {m.kind}, edit distance {m.distance}"
                )
        else:
            print("  impersonation  no brand lookalike signal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
