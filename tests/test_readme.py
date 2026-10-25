"""The generated docs cannot drift from the run that produced them."""

from __future__ import annotations

import tomllib

from urlish import ROOT, __version__
from urlish.benchmark import README, RESULTS_END, RESULTS_START


def test_report_header_names_the_version_and_the_data() -> None:
    text = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    header = next(line for line in text.splitlines() if line.startswith("urlish "))
    assert __version__ in header, f"stale report header, rerun urlish-benchmark: {header}"
    assert "tranco" in header, f"report does not name the data snapshot: {header}"


def test_installed_version_matches_pyproject() -> None:
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == declared["project"]["version"], "run `uv sync` after bumping the version"


def test_readme_table_is_the_report_table() -> None:
    readme = README.read_text(encoding="utf-8")
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    block = readme.split(RESULTS_START)[1].split(RESULTS_END)[0].strip()
    assert block, "the results block is empty"
    assert block in report, "README table differs from REPORT.md; rerun urlish-benchmark"
