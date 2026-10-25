"""A banned label word fails the build, not the next review.

`GLOSSARY.md` lists avoided synonyms for every term. Only the ones with no
legitimate use here are checked mechanically; homographs such as "feed",
"source", "dataset" and "domain" stay a review call, because a word check cannot
tell a URL from a collection.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from urlish import ROOT

BANNED = (
    "benign",
    "phishing sample",
    "data release",
    "dataset version",
    "semver",
    "release number",
)
DOCS = ("README.md", "REPORT.md", "AGENTS.md", "CODING_STANDARDS.md")


def banned_hits(text: str) -> list[str]:
    return [word for word in BANNED if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE)]


def test_the_check_can_fail() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "sample.md"
        sample.write_text("Every benign URL was fine.\n", encoding="utf-8")
        assert banned_hits(sample.read_text(encoding="utf-8")) == ["benign"]


def test_every_checked_word_is_a_documented_avoided_synonym() -> None:
    glossary = (ROOT / "GLOSSARY.md").read_text(encoding="utf-8")
    for word in BANNED:
        assert word in glossary, f"{word!r} is not a documented avoided synonym"


def test_docs_use_the_glossary_terms() -> None:
    paths = [ROOT / name for name in DOCS] + sorted((ROOT / "docs").rglob("*.md"))
    for path in paths:
        hits = banned_hits(path.read_text(encoding="utf-8"))
        assert not hits, f"{path.name} uses avoided synonyms: {', '.join(hits)}"
