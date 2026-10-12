#!/usr/bin/env python
"""Benchmark tabu search on CPU: Andrey against bnlearn, with hill climbing beside it.

Runs Andrey's and bnlearn's `tabu`, with each package's `hc`, the search tabu search extends, and
the empty graph as a quality baseline. The data is linear-Gaussian on Erdos-Renyi graphs at average
degree 2 with `n = 10 * d` rows, the R lane's canonical slice. Every solution is serial and runs
with its package's tabu settings, bnlearn's defaults.

Tabu search exists to reach a better score than hill climbing, so each fit also gets `bic`: the BIC
of the graph it returned on the data it was fitted on, in Andrey's deviance units (lower is better),
with any DAG of a CPDAG standing for its class. Andrey's tabu and hc return CPDAGs and bnlearn's
return DAGs: compare a DAG row's `mec_shd` with a CPDAG row's `shd`.

Before running, load the R modules and export `R_LIBS_USER` as `benchmarks/start_r.sh --env`
prints.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/tabu_cpu.py \\
        --out outputs/tabu-cpu --cap-mem-mb 32000
"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from andrey_bench import datasets, memcap, rsession, rundir
from andrey_bench.adapters.andrey_hc import andrey_hc_adapters
from andrey_bench.adapters.andrey_tabu import andrey_tabu_adapters
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.bnlearn_hc import BnlearnHC
from andrey_bench.adapters.bnlearn_tabu import BnlearnTabu
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores

#: The solutions, tabu pairs first.
SOLUTIONS = [
    *andrey_tabu_adapters(),
    BnlearnTabu(),
    *andrey_hc_adapters(),
    BnlearnHC(),
    BaselineEmpty(),
]

#: The solution whose `bic` the summary subtracts: what tabu search improves on.
REFERENCE = "andrey.hc.numpy.serial"

#: The linear-Gaussian Erdos-Renyi slice at the canonical density.
REGIME = "linear_gauss_er"

#: Smallest run that touches every solution: one tiny dataset.
SMOKE = {"ladder": [5], "seeds": 1, "repeats": 1, "cap_wall_s": 300.0}

#: Untimed fits before each measurement. R starts and loads bnlearn in `setup()`, outside timing.
WARMUP = 1


def regime_slice() -> dict:
    """Return the :class:`DatasetKey` coordinates of :data:`REGIME`."""
    topology, functional, noise, standardize = datasets.REGIMES[REGIME]
    return {
        "topology": topology,
        "functional": functional,
        "noise": noise,
        "standardize": standardize,
        "density": datasets.CANONICAL_DENSITY,
    }


def ladder_keys(ladder: list[int], seeds: int) -> list[DatasetKey]:
    """Return one dataset of ``10 * d`` rows per ``(d, seed)``."""
    return [
        DatasetKey(**regime_slice(), d=d, n=10 * d, seed=seed)
        for d in ladder
        for seed in range(seeds)
    ]


def _log(msg: str = "") -> None:
    """Print and flush batch output immediately."""
    print(msg, flush=True)


def graph_bic(data: np.ndarray, estimate) -> float:
    """The BIC of `estimate` on `data` in deviance units, as Andrey's searches report it.

    A CPDAG is scored through one of its DAGs; every DAG of the class has the same BIC.
    """
    from andrey.core import ARROW, TAIL
    from andrey.core.orient import pdag2dag
    from andrey.core.score import BICScore

    marks = estimate.to_numpy()
    dag = pdag2dag(marks) if estimate.kind == "cpdag" else marks
    d = dag.shape[0]
    # Endpoint marks: `i -> j` has the tail at `i` and the arrowhead at `j`.
    parents = [[i for i in range(d) if dag[i, j] == TAIL and dag[j, i] == ARROW] for j in range(d)]
    score = BICScore(np.asarray(data, dtype=np.float64))
    return float(sum(score.score(j, parents[j]) for j in range(d)))


def add_bic(
    out: Path, records_dir: Path, keys: Mapping[str, DatasetKey], rescore: bool = False
) -> None:
    """Add the `bic` column to every part file that lacks it.

    `keys` maps each ``dataset_id`` of the run to its :class:`DatasetKey`. Each fit's stored
    estimate is read back and scored with :func:`graph_bic` on its dataset; a fit with no estimate
    has ``NaN``. The rewrite is atomic, and a part that already has the column is left alone, so a
    resumed run scores only the units it measured; `rescore` replaces every part's column instead.
    """
    store = datasets.store_dir(out)
    # The last dataset read, kept so each dataset is loaded once for all of its solutions.
    loaded: tuple[str, np.ndarray] | None = None
    for part in sorted(records_dir.glob("*.parquet")):
        frame = pd.read_parquet(part)
        if not rescore and "bic" in frame.columns:
            continue
        frame = frame.drop(columns=["bic"], errors="ignore")
        values = []
        for record in frame.itertuples():
            estimate = rundir.read_structure(out, record.run_id) if record.status == "ok" else None
            if estimate is None:
                values.append(float("nan"))
                continue
            if loaded is None or loaded[0] != record.dataset_id:
                if record.dataset_id not in keys:
                    raise SystemExit(
                        f"{part.name}: dataset {record.dataset_id} is not in this run's ladder; "
                        "pass the first run's flags"
                    )
                loaded = (record.dataset_id, datasets.load(store, keys[record.dataset_id])[0])
            values.append(graph_bic(loaded[1], estimate))
        frame["bic"] = np.asarray(values, dtype=float)
        scratch = part.with_name(f".{part.name}.tmp")
        frame.to_parquet(scratch, index=False)
        os.replace(scratch, part)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tabu search, Andrey against bnlearn, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--ladder",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100],
        help="d sizes, each with n = 10 * d rows (default: 10 20 50 100)",
    )
    p.add_argument("--seeds", type=int, default=5, help="datasets per rung (default: 5)")
    p.add_argument("--repeats", type=int, default=3, help="timed repeats per fit (default: 3)")
    p.add_argument(
        "--warmup", type=int, default=WARMUP, help="untimed fits per repeat (default: 1)"
    )
    p.add_argument("--cap-wall-s", type=float, default=1800.0, help="wall cap per repeat, seconds")
    p.add_argument("--cap-mem-mb", type=float, default=None, help="memory cap per repeat, MB")
    p.add_argument("--mem-route", choices=memcap.ROUTES, default=None, help="cap enforcement route")
    p.add_argument(
        "--cores",
        type=int,
        default=available_cores(),
        help="core budget per fit (default: every core this job is allowed)",
    )
    p.add_argument(
        "--machine-id",
        default=socket.gethostname(),
        help="machine id stamped on every record and in run_id (default: this host's name)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="run d=5, one seed, one repeat, and a 300 s cap, replacing those flags",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue the run already in --out: skip units whose records are on disk. Requires "
        "the same run_meta.json, so pass the flags the first run was given",
    )
    p.add_argument(
        "--rescore",
        action="store_true",
        help="fit nothing: recompute the bic column of the run already in --out and rewrite "
        "runs.parquet. Takes the first run's flags, as --resume does",
    )
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)

    out = Path(args.out)
    keys = {key.digest: key for key in ladder_keys(args.ladder, args.seeds)}

    if args.rescore:
        # Fits nothing, so the run's STATUS.json and sittings stay as the fitting runs left them.
        records_dir = out / rundir.RECORDS_DIRNAME
        if not any(records_dir.glob("*.parquet")):
            raise SystemExit(f"--rescore: {records_dir} holds no records")
        add_bic(out, records_dir, keys, rescore=True)
        table = rundir.assemble(records_dir)
        table.to_parquet(out / "runs.parquet", index=False)
        _log(f"rescored    : {len(table)} rows  -> {out / 'runs.parquet'}")
        _summarize(table)
        return 0

    # Starts R before the first fit, so a missing module or library fails once.
    r_environment = rsession.check_environment("bnlearn")
    # Written before the first fit, so an interrupted benchmark still says what it was asked for.
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in SOLUTIONS],
        "params": {a.name: dict(a.params()) for a in SOLUTIONS},
        "r environment": r_environment,
        "slice": regime_slice(),
        "ladder": args.ladder,
        "seeds": args.seeds,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cap_wall_s": args.cap_wall_s,
        "cap_mem_mb": args.cap_mem_mb,
        "mem_route": memcap.resolve_route(args.cap_mem_mb, args.mem_route),
        "cores": args.cores,
        "machine_id": args.machine_id,
    }
    # Reads run_meta.json before writing it, so --resume has something to check against.
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())

    _log(f"=== Tabu search, Andrey and bnlearn -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:12s}: {value}")
    _log(f"{'git_sha':12s}: {prov['git_sha']}")
    _log("")

    records: list = []
    errors: list[str] = []
    skipped = 0
    start = time.perf_counter()

    for d in args.ladder:
        # One rung at a time, so a rung's data is materialized only when the ladder reaches it.
        result = run_batch(
            datasets.store_dir(out),
            SOLUTIONS,
            ladder_keys([d], args.seeds),
            machine_id=args.machine_id,
            cap_wall_s=args.cap_wall_s,
            cap_mem_mb=args.cap_mem_mb,
            mem_route=args.mem_route,
            repeats=args.repeats,
            warmup=args.warmup,
            cores=args.cores,
            records_dir=records_dir,
            run_dir=out,
        )
        records.extend(result.records)
        errors.extend(result.errors)
        skipped += result.skipped

        # What each solution cost at this rung, or what stopped it.
        ran = []
        for adapter in SOLUTIONS:
            mine = [r for r in result.records if r.name == adapter.name]
            walls = [r.wall_s for r in mine if r.status == "ok" and r.wall_s is not None]
            statuses = sorted({r.status for r in mine})
            cost = (
                f"{statistics.median(walls):.3g}s" if walls else "/".join(statuses) or "no-records"
            )
            ran.append(f"{adapter.name}={cost}")
        _log(f"[d={d}] {'  '.join(ran)} | elapsed {time.perf_counter() - start:.0f}s")
        add_bic(out, records_dir, keys)

    # From the parts, not from memory: a resume that measured nothing still writes the table for
    # everything measured before it.
    add_bic(out, records_dir, keys)
    table = rundir.assemble(records_dir)
    table.to_parquet(out / "runs.parquet", index=False)
    rundir.finish_run(out)

    _log("")
    _log(f"elapsed     : {time.perf_counter() - start:.1f}s")
    _log(f"measured    : {len(records)} records, {skipped} units skipped")
    _log(f"table       : {len(table)} rows  -> {out / 'runs.parquet'}")
    _log(f"errors      : {len(errors)}")
    for err in errors:
        _log(f"  ! {err}")
    _summarize(table)
    return 1 if errors else 0


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """One row per solution and rung: finished fits, median wall time, graph error, and BIC.

    ``graph_shd`` is the median of ``shd`` for a CPDAG row and of ``mec_shd`` for a DAG row, the
    pair that compares across the two output types. ``bic_vs_hc`` is the median, over the datasets
    both finished, of the solution's ``bic`` minus :data:`REFERENCE`'s on the same dataset: below
    zero is a better score than Andrey's hill climbing. A timeout has no scores and is not counted
    as any graph; ``ok`` and ``fits`` show how many fits each median covers.
    """
    ok = table[table["status"] == "ok"].copy()
    ok["graph_shd"] = np.where(ok["output_type"] == "dag", ok["mec_shd"], ok["shd"])
    # One BIC per (solution, dataset): repeats refit the same data with the same deterministic search.
    per_dataset = ok.groupby(["name", "dataset_id"])["bic"].first()
    reference = per_dataset.xs(REFERENCE, level="name") if REFERENCE in ok["name"].values else None
    rows = []
    for (name, d), fits in table.groupby(["name", "d"], sort=True):
        done = ok[(ok["name"] == name) & (ok["d"] == d)]
        gap = float("nan")
        if reference is not None and not done.empty:
            mine = per_dataset.xs(name, level="name").reindex(done["dataset_id"].unique())
            diff = (mine - reference.reindex(mine.index)).dropna()
            gap = float(diff.median()) if len(diff) else float("nan")
        rows.append(
            {
                "name": name,
                "d": int(d),
                "ok": len(done),
                "fits": len(fits),
                "wall_s": done["wall_s"].median(),
                "graph_shd": done["graph_shd"].median(),
                "bic_vs_hc": gap,
            }
        )
    return pd.DataFrame(rows)


def _summarize(table: pd.DataFrame) -> None:
    """Print :func:`summary`, one line per solution and rung."""
    if table.empty:
        return
    _log("")
    _log(f"{'solution':28s} {'d':>4s} {'ok/all':>7s} {'wall_s':>8s} {'shd':>6s} {'bic-hc':>9s}")
    for row in summary(table).itertuples():
        _log(
            f"{row.name:28s} {row.d:4d} {row.ok:3d}/{row.fits:<3d} {row.wall_s:8.3g} "
            f"{row.graph_shd:6.3g} {row.bic_vs_hc:9.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
