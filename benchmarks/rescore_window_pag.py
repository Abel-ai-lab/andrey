#!/usr/bin/env python
"""Score every finished LPCMCI-lane fit against the true PAG over the observed series.

Reads a run folder that ``pcmci_cpu.py --method lpcmci`` wrote: ``runs.parquet``, the graph each
fit returned (``structures/``), and the datasets with their truths (``data/``, or
``$ANDREY_BENCH_DATA`` when the run shared a store). Fits nothing. Writes
``<run>/window_pag.parquet``, one row per scored record: its ``run_id`` and the columns of
:func:`andrey_bench.scoring.window_scores` against the true time-series PAG
(:func:`andrey_bench.window_pag.true_window_pag`). Only records whose truth has unobserved series
are scored; the others already carry these columns.

    $ANDREY_BENCH_PYTHON benchmarks/rescore_window_pag.py outputs/lpcmci
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from andrey_bench import datasets, rundir, scoring
from andrey_bench.integration import _truth

OUTPUT = "window_pag.parquet"


def rescore(run: Path) -> pd.DataFrame:
    """Return the true-PAG scores of every finished record in ``run`` and write them beside it."""
    table = pd.read_parquet(run / "runs.parquet")
    store = datasets.store_dir(run)
    registry = datasets.load_registry(store)["keys"]
    by_file = {entry["file"]: entry for entry in registry.values()}
    rows = []
    ok = table[table["status"] == "ok"].sort_values("dataset_id")  # one true PAG per dataset
    truth, truth_id = None, None
    for record in ok.itertuples():
        entry = by_file[f"ds_{record.dataset_id}.npz"]
        if not entry["latents"] or entry["graph_kind"] != datasets.WINDOW_KIND:
            continue
        if record.dataset_id != truth_id:
            with np.load(store / entry["file"]) as z:
                truth = _truth(z["graph"], entry["graph_kind"], entry["latents"])
            truth_id = record.dataset_id
        estimate = rundir.read_structure(run, record.run_id)
        if estimate is None:
            raise SystemExit(f"{run}: no stored graph for {record.run_id}")
        rows.append({"run_id": record.run_id, **scoring.window_scores(estimate, truth)})
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
