"""The README results table is generated, so it cannot drift from the report."""

from __future__ import annotations

from urlish import ROOT
from urlish.benchmark import README, RESULTS_END, RESULTS_START


def test_readme_table_is_the_report_table() -> None:
    readme = README.read_text(encoding="utf-8")
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    block = readme.split(RESULTS_START)[1].split(RESULTS_END)[0].strip()
    assert block, "the results block is empty"
    assert block in report, "README table differs from REPORT.md; rerun urlish-benchmark"
