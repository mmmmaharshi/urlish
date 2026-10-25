"""Score URLs from the command line.

urlish https://paypa1.com/login
urlish --train          # rebuild the model artifact first
"""

from __future__ import annotations

import argparse

from .artifact import load
from .features import analyze, featurize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="urlish", description=__doc__)
    parser.add_argument("--train", action="store_true", help="rebuild the model artifact first")
    parser.add_argument("urls", nargs="*", help="URLs to score")
    args = parser.parse_args(argv)

    if args.train:
        from .benchmark import main as benchmark_main

        return benchmark_main()
    if not args.urls:
        parser.print_help()
        return 2

    bundle = load()
    model, brands = bundle["model"], bundle["brands"]
    threshold = bundle["threshold"]

    for url in args.urls:
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
