from __future__ import annotations

import sys
from pathlib import Path

from atlas.cli import main


def test_fixture_compare_cli_requires_explicit_mode_and_writes_report(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    suite_path = Path(__file__).parents[1] / "examples" / "benchmarks" / "smoke.json"
    output_db = tmp_path / "evaluations.sqlite"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "atlas",
            "compare",
            "--suite",
            str(suite_path),
            "--baseline-id",
            "baseline",
            "--candidate-id",
            "candidate",
            "--trials",
            "2",
            "--fixture-only",
            "--store",
            str(output_db),
        ],
    )
    assert main() == 0
    output = capsys.readouterr().out
    assert '"paired_sample_count": 4' in output
    assert '"illustrative": true' in output
    assert output_db.exists()
