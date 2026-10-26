# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pyarrow>=18",
#   "fsspec>=2024.0",
#   "aiohttp>=3.9",
#   "requests>=2.31",
# ]
# ///
"""Fetch the PhreshPhish corpus as a small CSV of URL, date, label, target.

Only four parquet columns are read, so the download is tens of megabytes rather
than hundreds of gigabytes. Output is untracked (data/ is gitignored); the
report records its SHA-256 prefix for reproducibility.
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from pathlib import Path

import fsspec
import pyarrow.parquet as pq

REPO = "phreshphish/phreshphish"
TREE_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/main/data"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"
COLUMNS = ("url", "date", "label", "target")
OUT = Path("data") / "phreshphish.csv"


def list_shards() -> list[str]:
    with urllib.request.urlopen(TREE_URL, timeout=60) as resp:
        entries = json.load(resp)
    paths = sorted(e["path"] for e in entries if e["path"].endswith(".parquet"))
    if not paths:
        raise RuntimeError("no parquet shards found")
    return paths


def main() -> int:
    shards = list_shards()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    bytes_read = 0
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        fs = fsspec.filesystem("http")
        for shard in shards:
            url = f"{BASE_URL}/{shard}"
            with fs.open(url, "rb") as remote:
                table = pq.read_table(remote, columns=list(COLUMNS))
            missing = [c for c in COLUMNS if c not in table.column_names]
            if missing:
                raise RuntimeError(f"{shard}: missing columns {missing}")
            n = table.num_rows
            for i in range(n):
                writer.writerow([table.column(c)[i].as_py() for c in COLUMNS])
            rows += n
            bytes_read += table.nbytes
            print(f"{shard}: {n:,} rows", file=sys.stderr)
    print(
        f"wrote {rows:,} rows to {OUT} ({bytes_read:,} uncompressed bytes of columns)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
EOF
