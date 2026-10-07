"""A run's `kind` comes from where it was filed below the results root, and nowhere else."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from andrey_bench.aggregate import pool


def _run(run_dir: Path) -> None:
    (run_dir / "records").mkdir(parents=True)
    pd.DataFrame({"status": ["ok"]}).to_parquet(run_dir / "records" / "part.parquet")


def test_a_campaign_directory_above_the_root_does_not_make_a_probe_a_campaign(tmp_path):
    root = tmp_path / "campaign" / "results"
    _run(root / "batch" / "probes" / "reach")
    _run(root / "batch" / "campaign" / "main")

    table = pool(root)

    assert dict(zip(table.run_dir, table.kind)) == {
        "batch/probes/reach": "probe",
        "batch/campaign/main": "campaign",
    }
