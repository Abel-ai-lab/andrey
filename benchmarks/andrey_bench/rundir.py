"""Persist completed benchmark units and resume interrupted runs.

Each part file holds all repeats for one dataset/solution pair. Its filename includes the
environment ID to prevent reuse across environments; its presence marks completion. Filenames
do not detect code changes, so freeze the benchmark source during a campaign.
"""

from __future__ import annotations

import difflib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from andrey_bench import provenance
from andrey_bench.contracts import RunRecord

RECORDS_DIRNAME = "records"
RUN_META_NAME = "run_meta.json"
PROVENANCE_NAME = "provenance.json"
STATUS_NAME = "STATUS.json"
STRUCTURES_DIRNAME = "structures"


def structure_path(out: str | Path, run_id: str) -> Path:
    """Where the graph for one measurement is kept."""
    return Path(out) / STRUCTURES_DIRNAME / f"{run_id.replace(':', '__')}.npz"


def write_structure(out: str | Path, run_id: str, structure) -> None:
    """Write the graph a solution returned to ``structures/``, normalized, as its edge set."""
    path = structure_path(out, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        edges=np.asarray(structure.to_edges()),
        kind=np.asarray(structure.kind),
        n_nodes=np.asarray(structure.n_nodes),
    )


def read_structure(out: str | Path, run_id: str):
    """Rebuild the graph written by :func:`write_structure`, or None if it was never written."""
    from andrey import GraphStructure

    path = structure_path(out, run_id)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return GraphStructure.from_edges(z["edges"], n_nodes=int(z["n_nodes"]), kind=str(z["kind"]))


def part_path(records_dir: str | Path, dataset_id: str, name: str, environment_id: str) -> Path:
    """Return the part-file path for one benchmark unit."""
    return Path(records_dir) / f"{dataset_id}__{name}__{environment_id}.parquet"


def write_part(path: str | Path, records: Sequence[RunRecord]) -> None:
    """Atomically save a completed unit by writing a temporary file, then renaming it."""
    from andrey_bench.integration import to_dataframe

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    to_dataframe(records).to_parquet(tmp, index=False)
    os.replace(tmp, path)


def assemble(records_dir: str | Path) -> pd.DataFrame:
    """Assemble all completed part files into the run table."""
    from andrey_bench.integration import to_dataframe

    parts = sorted(Path(records_dir).glob("*.parquet"))
    if not parts:
        return to_dataframe([])
    table = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    canonical = [c for c in to_dataframe([]).columns if c in table.columns]
    return table[canonical + sorted(set(table.columns) - set(canonical))]


def start_run(out: str | Path, run_meta: dict, *, resume: bool) -> Path:
    """Prepare incremental record storage and validate resume compatibility.

    Existing measurements require ``resume=True`` and matching run metadata.
    """
    out = Path(out)
    records_dir = out / RECORDS_DIRNAME
    records_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out / RUN_META_NAME
    written = json.dumps(run_meta, indent=2, sort_keys=True, default=str)

    if any(records_dir.glob("*.parquet")):
        recorded = meta_path.read_text() if meta_path.exists() else ""
        if not resume:
            raise SystemExit(
                f"{records_dir} already holds measurements. Pass --resume to continue that run, "
                "or give --out a fresh directory."
            )
        if recorded != written:
            diff = difflib.unified_diff(
                recorded.splitlines(), written.splitlines(), "recorded", "requested", lineterm=""
            )
            raise SystemExit(
                "--resume continues the same run, and this is not it:\n" + "\n".join(diff)
            )

    meta_path.write_text(written)
    # A resume writes into the parent's --out, so a directory can hold records from two jobs.
    prior = []
    status_path = Path(out) / STATUS_NAME
    if status_path.exists():
        try:
            prior = json.loads(status_path.read_text()).get("sittings", []) or []
        except json.JSONDecodeError:
            prior = []
    node = os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME")
    job = os.environ.get("SLURM_JOB_ID")
    write_status(
        out,
        complete=False,
        driver=run_meta.get("driver"),
        node=node,
        slurm_job=job,
        units_expected=_units_expected(run_meta),
        sittings=[*prior, {"slurm_job": job, "node": node}],
    )
    return records_dir


def _units_expected(run_meta: dict) -> int | None:
    """How many ``(dataset, solution)`` parts a finished ladder should hold, if the meta says."""
    try:
        return len(run_meta["ladder"]) * int(run_meta["seeds"]) * len(run_meta["solution list"])
    except (KeyError, TypeError, ValueError):
        return None


def finish_run(out: str | Path) -> None:
    """Record how many units arrived, and mark the run complete only if all of them did."""
    out = Path(out)
    done = len(list((out / RECORDS_DIRNAME).glob("*.parquet")))
    expected = None
    status_path = out / STATUS_NAME
    if status_path.exists():
        try:
            expected = json.loads(status_path.read_text()).get("units_expected")
        except json.JSONDecodeError:
            pass
    # Unknown expectation is not evidence of a shortfall: run_meta may not describe a ladder.
    write_status(out, units_done=done, complete=expected is None or done >= expected)


def write_status(out: str | Path, **fields) -> None:
    """Record how far a run got. Merged into any existing ``STATUS.json``."""
    path = Path(out) / STATUS_NAME
    prior = {}
    if path.exists():
        try:
            prior = json.loads(path.read_text())
        except json.JSONDecodeError:
            prior = {}
    merged = {**prior, **{k: v for k, v in fields.items() if v is not None}}
    tmp = path.with_name(f".{STATUS_NAME}.tmp")
    tmp.write_text(json.dumps(merged, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def provenance_once(
    out: str | Path, *, timestamp: str, extra_venvs: Mapping[str, str | Path] | None = None
) -> dict:
    """Return the run's provenance, capturing it only if it does not exist.

    ``extra_venvs`` names any further environments this run's fits ran in; see
    :func:`~andrey_bench.provenance.capture`. A resumed run reads the existing file, so the
    environments recorded are the ones the *first* sitting was told about.
    """
    path = Path(out) / PROVENANCE_NAME
    if path.exists():
        return json.loads(path.read_text())
    return provenance.capture(out, timestamp=timestamp, extra_venvs=extra_venvs)


__all__ = [
    "part_path",
    "write_part",
    "assemble",
    "start_run",
    "write_status",
    "provenance_once",
    "structure_path",
    "write_structure",
    "read_structure",
]
