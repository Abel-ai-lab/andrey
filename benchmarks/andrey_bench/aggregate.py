#!/usr/bin/env python
"""Pool benchmark records with their run metadata and provenance.

Reads every ``records/`` below the root, including interrupted runs whose ``runs.parquet`` has
not been assembled. Joins ``provenance.json``, ``run_meta.json``, and ``STATUS.json`` onto rows so
comparisons can distinguish builds, settings, and machines. An editable Andrey version of
``0.0.0`` alone does not identify a build.

Environment stamps describe the shared interpreter. ``extra_environments`` names additional
interpreters; matching package labels supply per-solution BLAS and Python stamps.

    python -m andrey_bench.aggregate <results root> --out pooled.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

#: Run metadata copied to every row as ``column: (file, dotted path)``. A setting that changes a
#: measured number and is not already a record field belongs here.
STAMPS: dict[str, tuple[str, str]] = {
    "git_sha": ("provenance", "git_sha"),
    "cpu_model": ("provenance", "cpu.model"),
    "cpu_cores": ("provenance", "cpu.logical_cores"),
    # Shared-environment linkage; pool() replaces it for matching extra-environment packages.
    "blas": ("provenance", "blas.blas.name"),
    "captured_at": ("provenance", "timestamp"),
    "driver": ("run_meta", "driver"),
    "ges_min_work": ("run_meta", "ges_min_work"),
    "hc_min_work": ("run_meta", "hc_min_work"),
    "seeds": ("run_meta", "seeds"),
    "repeats": ("run_meta", "repeats"),
    # From STATUS.json, so a reader can tell a finished run from one that stopped early without
    # reconstructing it from the scheduler.
    "run_complete": ("status", "complete"),
    "run_units_done": ("status", "units_done"),
    "run_units_expected": ("status", "units_expected"),
    "superseded_by": ("status", "superseded_by"),
    "caveat": ("status", "caveat"),
}


def _dig(blob: dict, path: str):
    """Follow a dotted path into nested dicts; ``None`` if any step is missing."""
    cur = blob
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def pool(root: str | Path) -> pd.DataFrame:
    """Pool ``records/*.parquet`` below ``root`` and attach each run's metadata."""
    root = Path(root)
    frames = []
    for records in sorted(root.rglob("records")):
        if not records.is_dir():
            continue
        run_dir = records.parent
        parts = sorted(records.glob("*.parquet"))
        if not parts:
            continue
        df = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
        if df.empty:
            continue
        sidecars = {
            "provenance": _read(run_dir / "provenance.json"),
            "run_meta": _read(run_dir / "run_meta.json"),
            "status": _read(run_dir / "STATUS.json"),
        }
        for column, (which, path) in STAMPS.items():
            stamped = _dig(sidecars[which], path)
            # Preserve per-solution values and nulls: a parallel threshold may apply only to
            # Andrey, with competitors hashing it as unset. Use run metadata only when the
            # column is absent or entirely null, as in records predating the field.
            if column in df.columns and df[column].notna().any():
                continue
            df[column] = stamped
        # More than one means a resume: two jobs' records in one directory.
        df["run_sittings"] = len(_dig(sidecars["status"], "sittings") or []) or None
        # Environments beyond the shared one, when a solution's package could not live in it.
        environments = _dig(sidecars["provenance"], "environments") or {}
        df["extra_environments"] = ",".join(sorted(environments))
        # Extra environments are registered by package name. Use each matching package's own
        # BLAS and Python identity so grouping does not assign it the shared environment's linkage.
        for label, entry in environments.items():
            mine = df.package == label
            if mine.any():
                df.loc[mine, "blas"] = _dig(entry, "blas.blas.name")
                df.loc[mine, "env_python"] = _dig(entry, "versions.python")
        # Keep runs distinguishable even when all their metadata agrees.
        df["run_dir"] = str(run_dir.relative_to(root))
        # Campaign placement records the submitter's choice to measure on a held node; seed count
        # cannot establish isolation. Match a path component below root, including nested batches;
        # a ``campaign`` directory above root does not classify the run.
        df["kind"] = "campaign" if "campaign" in run_dir.relative_to(root).parts else "probe"
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)

    # Records written before a field existed lack the column. Fill so consumers see one schema.
    import dataclasses

    from andrey_bench.contracts import RunRecord

    for field in dataclasses.fields(RunRecord):
        if field.name not in table.columns:
            table[field.name] = pd.NA
    return table


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pool benchmark runs with their provenance")
    p.add_argument("root", help="results root; every runs.parquet beneath it is pooled")
    p.add_argument("--out", help="write the pooled table here as parquet")
    args = p.parse_args(argv)

    table = pool(args.root)
    if table.empty:
        print(f"no runs under {args.root}")
        return 1

    print(f"{len(table)} rows from {table.run_dir.nunique()} run directories")
    cols = ["run_dir", "kind", "driver", "git_sha", "cpu_model", "ges_min_work", "machine_id"]
    summary = table.groupby("run_dir", dropna=False)[cols[1:]].first()
    print(summary.to_string())

    # Report incomplete and superseded runs before comparing their measurements.
    incomplete = table[table.run_complete == False]  # noqa: E712 - None must not match
    if len(incomplete):
        runs = sorted(incomplete.run_dir.unique())
        print(f"\n! {len(runs)} run(s) stopped early and are pooled anyway: {', '.join(runs)}")
        print("  their rows are real; the run simply holds fewer than it planned")
    superseded = table[table.superseded_by.notna()]
    if len(superseded):
        print(f"\n! superseded runs present: {sorted(superseded.run_dir.unique())}")
        print("  exclude them unless you are looking at why they were replaced")

    # Per-run notes on how the rows should be read.
    for run_dir, note in sorted(
        {
            (r.run_dir, r.caveat)
            for r in table.itertuples()
            if isinstance(getattr(r, "caveat", None), str)
        }
    ):
        print(f"\n! {run_dir}\n  {note}")

    for column in ("git_sha", "cpu_model", "ges_min_work", "hc_min_work"):
        distinct = table[column].dropna().unique()
        if len(distinct) > 1:
            print(
                f"\n! rows in this table span {len(distinct)} values of {column}: {list(distinct)}"
            )
            print("  group by it before comparing; a median across it is meaningless")

    if args.out:
        table.to_parquet(args.out, index=False)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
