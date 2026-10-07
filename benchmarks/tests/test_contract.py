"""Test benchmark dataset identity, record schema, and generated artifacts.

Relative identity tests cannot detect a change that moves every digest. These tests pin canonical
values so global identity changes must be deliberate.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

from andrey_bench import contracts
from andrey_bench.contracts import RUN_ENV_FIELDS, RunRecord
from andrey_bench.datasets import DatasetKey

# The canonical sparse slice at d=20 from the smoke benchmark.
_CANONICAL = DatasetKey(
    topology="er",
    functional="linear",
    noise="gaussian",
    standardize="standardized",
    density=2.0,
    d=20,
    n=200,
    seed=0,
)


def test_the_canonical_dataset_keeps_its_identity():
    """Pin the dataset digest and filename used by existing records."""
    assert _CANONICAL.digest == "d73a6b2a3227"
    assert _CANONICAL.filename == "ds_d73a6b2a3227.npz"


def test_the_row_schema_is_what_it_says():
    """`runs.parquet` columns come from `RunRecord`, so the schema changes when this list does."""
    assert tuple(sorted(f.name for f in dataclasses.fields(RunRecord))) == (
        "algorithm", "backend", "cap_mem_mb", "cap_wall_s", "child_cpu_s", "child_peak_rss_mb",
        "cores", "cpu_s", "d", "data_seed", "dataset_id", "density", "device", "device_id",
        "dtype", "elapsed_at_kill_s", "error_text", "exit_code", "functional", "ges_min_work",
        "gpu_energy_j", "gpu_peak_alloc_mb", "hc_min_work", "latents", "machine_id", "mem_route",
        "metrics",
        "mode", "n", "name",
        "noise", "num_workers", "oom_source", "output_hash", "output_type", "package",
        "package_version", "params", "peak_rss_mb", "r2_sortability", "repeat", "run_id",
        "scheduler", "setup_s",
        "standardize", "status", "structure_hash", "threads", "topology", "tree_peak_mem_mb",
        "varsortability", "wall_s", "warmup", "warmup_s",
    )  # fmt: skip


def test_the_environment_legs_are_what_they_say():
    """A field leaving or joining silently re-partitions every `run_id`."""
    assert RUN_ENV_FIELDS == (
        "backend", "mode", "dtype", "device", "cap_mem_mb",
        "cap_wall_s", "warmup", "machine_id", "threads", "num_workers",
    )  # fmt: skip


def test_the_campaign_a_user_runs_produces_what_the_readme_promises(tmp_path):
    """Run the README entry point end to end without stubs.

    Every other test reaches `run_batch` directly, so nothing covers argument parsing, the
    output layout, or completion.
    """
    bench = __import__("pathlib").Path(__file__).resolve().parents[1]
    out = tmp_path / "run"
    done = subprocess.run(
        # fmt: off
        [
            sys.executable,
            str(bench / "benchmark.py"),
            "--out",
            str(out),
            "--ladder",
            "10",
            "--seeds",
            "1",
            "--repeats",
            "1",
        ],
        # fmt: on
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, done.stderr[-3000:]

    for promised in ("runs.parquet", "provenance.json", "run_meta.json", "data/registry.json"):
        assert (out / promised).exists(), f"README promises {promised}; the run did not write it"

    import pandas as pd

    rows = pd.read_parquet(out / "runs.parquet")
    assert len(rows) == 4, f"one fit per rostered solution; got {len(rows)}"
    assert set(rows["status"]) == {"ok"}, dict(rows["status"].value_counts())

    # The empty graph is the floor: it scores the truth's edge count, so nothing should be worse.
    floor = rows.loc[rows["name"] == "baseline.empty", "shd"].iloc[0]
    worse = rows.loc[(rows["name"] != "baseline.empty") & (rows["shd"] > floor), "name"]
    assert worse.empty, f"scored worse than the empty graph: {list(worse)}"

    registry = json.loads((out / "data" / "registry.json").read_text())
    assert registry["n_datasets"] == 1


def test_an_unset_conditional_field_leaves_the_digest_as_it_was():
    """Keep existing run IDs when a conditional environment field is unset."""
    env = dict.fromkeys(contracts.RUN_ENV_FIELDS, 1)
    assert contracts.environment_id(env) == "da71311cf2e2"
    assert contracts.environment_id({**env, "ges_min_work": None}) == "da71311cf2e2"


def test_the_graph_survives_the_run_that_measured_it(tmp_path):
    """Scoring reduces a graph to numbers; the graph itself must survive the run."""
    import numpy as np

    from andrey import GraphStructure
    from andrey_bench import rundir

    marks = np.zeros((5, 5), dtype=np.int8)
    marks[0, 1], marks[1, 0] = 1, 2  # 0 -> 1
    marks[2, 3], marks[3, 2] = 1, 2  # 2 -> 3
    original = GraphStructure.from_numpy(marks, kind="cpdag")

    run_id = "abc123:andrey.ges.numpy.serial:0:def456"
    rundir.write_structure(tmp_path, run_id, original)
    restored = rundir.read_structure(tmp_path, run_id)

    from andrey_bench.scoring import structure_hash

    assert structure_hash(restored) == structure_hash(original)
    assert rundir.read_structure(tmp_path, "never:written:0:xxx") is None


def test_a_run_missing_a_unit_does_not_report_itself_complete(tmp_path):
    """A partial run must not report itself complete."""
    from andrey_bench import rundir

    (tmp_path / rundir.RECORDS_DIRNAME).mkdir()
    rundir.write_status(tmp_path, units_expected=3)
    for i in range(2):
        (tmp_path / rundir.RECORDS_DIRNAME / f"part{i}.parquet").touch()

    rundir.finish_run(tmp_path)
    status = json.loads((tmp_path / rundir.STATUS_NAME).read_text())
    assert status["units_done"] == 2
    assert status["complete"] is False

    (tmp_path / rundir.RECORDS_DIRNAME / "part2.parquet").touch()
    rundir.finish_run(tmp_path)
    status = json.loads((tmp_path / rundir.STATUS_NAME).read_text())
    assert status["units_done"] == 3 and status["complete"] is True
