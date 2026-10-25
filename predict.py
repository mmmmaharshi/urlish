"""Score URLs from the command line.

python predict.py https://paypa1.com/login
python predict.py --train          # rebuild artifacts/model.pkl first
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from urlish.features import analyze, featurize  # noqa: E402

ARTIFACTS = ROOT / "artifacts" / "model.pkl"


def load():
    if not ARTIFACTS.exists():
        print("no model yet - run `python benchmark.py` first", file=sys.stderr)
        raise SystemExit(1)
    with open(ARTIFACTS, "rb") as fh:
        return pickle.load(fh)


def main(argv: list[str]) -> int:
    if "--train" in argv:
        return subprocess.call([sys.executable, str(ROOT / "benchmark.py")])

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
                    f"  impersonates {m.brand} ({m.primary}) - {m.kind}, "
                    f"edit distance {m.distance}"
                )
        else:
            print("  impersonation  no brand lookalike signal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
