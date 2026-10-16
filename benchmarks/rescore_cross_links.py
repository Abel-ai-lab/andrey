#!/usr/bin/env python
"""Score every finished fit of a time-series run by cross-link F1 of its window causal graph.

Reads a run folder that ``pcmci_cpu.py`` or ``lingam_ts.py`` wrote: ``runs.parquet``, the graph each
fit returned (``structures/``), and the datasets with their truths (``data/``, or
``$ANDREY_BENCH_DATA`` when the run shared a store). Fits nothing. Writes
``<run>/cross_links.parquet``, one row per scored record: its ``run_id`` and the columns of
:func:`andrey_bench.scoring.cross_link_scores`. A record whose truth has unobserved variables is
left out; LPCMCI's estimate on such data is compared with tigramite's only.

    $ANDREY_BENCH_PYTHON benchmarks/rescore_cross_links.py outputs/pcmci outputs/pcmciplus
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from andrey_bench import datasets, rundir, scoring
from andrey_bench.integration import _truth

OUTPUT = "cross_links.parquet"


def rescore(run: Path) -> pd.DataFrame:
    """Return the cross-link scores of every finished record in ``run`` and write them beside it."""
    table = pd.read_parquet(run / "runs.parquet")
    store = datasets.store_dir(run)
    registry = datasets.load_registry(store)["keys"]
    by_file = {entry["file"]: entry for entry in registry.values()}
    truths: dict[str, object] = {}
    rows = []
    for record in table[table["status"] == "ok"].itertuples():
        entry = by_file[f"ds_{record.dataset_id}.npz"]
        if entry["latents"]:
            continue
        if record.dataset_id not in truths:
            with np.load(store / entry["file"]) as z:
                truths[record.dataset_id] = _truth(z["graph"], entry["graph_kind"], 0)
        estimate = rundir.read_structure(run, record.run_id)
        if estimate is None:
            raise SystemExit(f"{run}: no stored graph for {record.run_id}")
        rows.append(
            {
                "run_id": record.run_id,
                **scoring.cross_link_scores(estimate, truths[record.dataset_id]),
            }
        )
    scored = pd.DataFrame(rows)
    scored.to_parquet(run / OUTPUT, index=False)
    return scored


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+", type=Path, help="run folders to rescore")
    args = p.parse_args(argv)
    for run in args.runs:
        scored = rescore(run)
        print(f"{run}: {len(scored)} records scored -> {run / OUTPUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
