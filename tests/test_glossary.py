"""A banned label word fails the build, not the next review.

`GLOSSARY.md` lists avoided synonyms for every term. Only the ones with no
legitimate use here are checked mechanically; homographs such as "feed",
"source", "dataset" and "domain" stay a review call, because a word check cannot
tell a URL from a collection. A new synonym is enforced by adding it here.
"""

from __future__ import annotations

import re

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
    """Whole words only, plurals included, and immune to line wrapping."""
    flat = " ".join(text.split())
    return [w for w in BANNED if re.search(rf"\b{re.escape(w)}s?\b", flat, re.IGNORECASE)]


def test_the_check_can_fail() -> None:
    assert banned_hits("Every benign URL was fine.") == ["benign"]
    assert banned_hits("Two benign\nhosts.") == ["benign"], "line wrapping must not hide a hit"
    assert banned_hits("Data\nrelease notes.") == ["data release"], "wrapped phrase, plural"


def test_every_checked_word_is_a_documented_avoided_synonym() -> None:
    glossary = (ROOT / "GLOSSARY.md").read_text(encoding="utf-8")
    avoided = " ".join(re.findall(r"_Avoid_:(.*)", glossary)).lower()
    for word in BANNED:
        assert word in avoided, f"{word!r} is not on an _Avoid_: line in GLOSSARY.md"


def test_docs_use_the_glossary_terms() -> None:
    paths = [ROOT / name for name in DOCS] + sorted((ROOT / "docs").rglob("*.md"))
    for path in paths:
        hits = banned_hits(path.read_text(encoding="utf-8"))
        assert not hits, (
            f"{path.name} uses avoided synonyms: {', '.join(hits)}"
            " (generated docs: rerun `uv run urlish-benchmark`)"
        )
