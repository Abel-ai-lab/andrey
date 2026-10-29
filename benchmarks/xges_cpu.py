#!/usr/bin/env python
"""Benchmark XGES on CPU: Andrey's against the xges package, with Andrey GES.

Runs XGES with and without the extended search from both packages, Andrey GES, and the empty graph
on linear-Gaussian data over Erdos-Renyi graphs at two average degrees. XGES gains most over GES on
denser graphs, so the default densities are 2 and 4. ``n`` is ``10 * d`` unless ``--n`` fixes it.

Every arm searches under one penalty, ``--lambda-value``. The package's ``alpha`` is Andrey's
``penalty_discount``: the two scores differ by a factor of ``-2`` and by terms that cancel in every move.
No arm caps the parents of a node: XGES has no cap, and the GES arm lifts the benchmark adapter's
cap of 4, as the public ``andrey.ges`` does.

Each record carries ``shd`` against the true CPDAG, and the driver adds two columns after each
rung: ``bic``, the BIC of the returned graph under one scorer (Andrey's ``BICScore`` at the
campaign's penalty, deviance units, lower is better), and ``bic_gap``, ``bic`` minus the true
graph's BIC on the same data. Every graph is rescored from ``structures/``, so the packages compare
on one scale rather than each on its own.

Threads differ by design. Andrey's XGES runs on one BLAS thread whatever its budget; Andrey GES uses
the budget as BLAS threads and the package as numba threads.

**The xges arms run in their own interpreter** (``benchmarks/start_xges.sh``), named by
``$ANDREY_BENCH_XGES_PYTHON``; a run that includes them fails at launch without it.

    ANDREY_BENCH_XGES_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/xges_cpu.py \\
        --out outputs/xges-cpu --cap-wall-s 3600 --cap-mem-mb 32000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from xges_ges import xges_venv

from andrey.core.score import BICScore
from andrey.search.ges import _recompute_objective
from andrey_bench import datasets, memcap, rundir
from andrey_bench.adapters.andrey_ges import AndreyGES, _andrey_version
from andrey_bench.adapters.andrey_xges import AndreyXGES
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.xges_ges import PYTHON_ENV_VAR, xges_adapter
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores, check_interpreters

#: Linear-Gaussian on Erdos-Renyi graphs, standardized: what GES and XGES assume. The average
#: degree comes from ``--densities``.
SLICE = {
    "topology": "er",
    "functional": "linear",
    "noise": "gaussian",
    "standardize": "standardized",
}

#: Smallest run that touches every solution at both densities.
SMOKE = {"ladder": [10], "seeds": 1, "cap_wall_s": 300.0}

#: Untimed fits before each timed one. The package's numba scorer compiles on its first call, and
#: without a warm-up that compilation is published as its speed.
WARMUP = 1

#: The columns :func:`add_bic` adds to each record.
BIC_COLUMNS = ("bic", "bic_gap")


def solutions(lambda_value: float) -> list:
    """The six arms, every one at the BIC weight `lambda_value`.

    Built rather than declared because the penalty comes from the command line and the package arms
    read their version from the interpreter that runs them. Each search setting has its own name,
    because ``params`` are excluded from ``run_id``.
    """
    return [
        AndreyXGES(name="andrey.xges.numpy", lambda_value=lambda_value),
        AndreyXGES(
            name="andrey.xges.numpy.no_extended_search",
            extended_search=False,
            lambda_value=lambda_value,
        ),
        xges_adapter(alpha=lambda_value, name="xges.xges"),
        xges_adapter(
            alpha=lambda_value, name="xges.xges.no_extended_search", extended_search=False
        ),
        AndreyGES(
            name="andrey.ges.numpy.serial.uncapped",
            backend="numpy",
            mode="serial",
            package_version=_andrey_version(),
            lambda_value=lambda_value,
            max_parents=None,
        ),
        BaselineEmpty(),
    ]


def ladder_keys(
    ladder: list[int], densities: list[float], seeds: int, n: int | None = None
) -> list[DatasetKey]:
    """One dataset per ``(d, density, seed)`` on :data:`SLICE`, of `n` rows, else ``10 * d``."""
    return [
        DatasetKey(**SLICE, density=density, d=d, n=10 * d if n is None else n, seed=seed)
        for d in ladder
        for density in densities
        for seed in range(seeds)
    ]


def total_bic(marks: np.ndarray, score: BICScore) -> float:
    """BIC of the graph with endpoint `marks` under `score`; ``NaN`` if it has no DAG extension.

    A CPDAG's members share one BIC, so any consistent extension gives the total.
    """
    marks = np.asarray(marks)
    try:
        return _recompute_objective(marks, score, marks.shape[0])
    except ValueError:
        return math.nan


def _log(msg: str = "") -> None:
    """Print and flush batch output immediately."""
    print(msg, flush=True)


def add_bic(
    out: Path,
    records_dir: Path,
    keys: Mapping[str, DatasetKey],
    lambda_value: float,
    rescore: bool = False,
) -> None:
    """Add :data:`BIC_COLUMNS` to every part file that lacks them.

    `keys` maps each ``dataset_id`` of the run to its :class:`DatasetKey`. Each fit's stored graph is
    read back and scored with one ``BICScore`` per dataset at `lambda_value`, which also scores the
    true graph. A fit with no graph has ``NaN`` in both columns, and so does a graph with no DAG
    extension. The rewrite is atomic, and a part that already has the columns is left alone, so a
    resumed run scores only the units it measured; `rescore` replaces every part's columns instead.
    """
    store = datasets.store_dir(out)
    # The last dataset's scorer and true BIC: a dataset's parts sort together.
    cached: tuple[str, BICScore, float] | None = None
    for part in sorted(records_dir.glob("*.parquet")):
        frame = pd.read_parquet(part)
        if not rescore and set(BIC_COLUMNS) <= set(frame.columns):
            continue
        frame = frame.drop(columns=[c for c in BIC_COLUMNS if c in frame.columns])
        rows = []
        for record in frame.itertuples():
            estimate = rundir.read_structure(out, record.run_id) if record.status == "ok" else None
            if estimate is None:
                rows.append((math.nan, math.nan))
                continue
            if cached is None or cached[0] != record.dataset_id:
                if record.dataset_id not in keys:
                    raise SystemExit(
                        f"{part.name}: dataset {record.dataset_id} is not in this run's ladder; "
                        "pass the first run's flags"
                    )
                data, marks = datasets.load(store, keys[record.dataset_id])
                score = BICScore(data, penalty_discount=lambda_value)
                cached = (record.dataset_id, score, total_bic(marks, score))
            _, score, truth_bic = cached
            bic = total_bic(estimate.to_numpy(), score)
            if math.isnan(bic):
                _log(f"  ! {record.name} @ {record.dataset_id}: no DAG extension, bic is NaN")
            rows.append((bic, bic - truth_bic))
        scores = pd.DataFrame(rows, index=frame.index, columns=list(BIC_COLUMNS), dtype=float)
        scratch = part.with_name(f".{part.name}.tmp")
        pd.concat([frame, scores], axis=1).to_parquet(scratch, index=False)
        os.replace(scratch, part)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="XGES, Andrey against the xges package, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--ladder",
        type=int,
        nargs="+",
        default=[25, 50, 100, 200, 500, 1000],
        help="d sizes, run in order (default: 25 50 100 200 500 1000)",
    )
    p.add_argument(
        "--densities",
        type=float,
        nargs="+",
        default=[2.0, 4.0],
        help="average degrees of the ER graphs (default: 2 4)",
    )
    p.add_argument(
        "--seeds", type=int, default=5, help="datasets per rung and density (default: 5)"
    )
    p.add_argument("--n", type=int, default=None, help="rows per dataset (default: 10 * d)")
    p.add_argument(
        "--lambda-value",
        type=float,
        default=1.0,
        help="BIC penalty weight of every arm, the package's alpha; the scale of the bic column "
        "(default: 1.0)",
    )
    p.add_argument("--repeats", type=int, default=1, help="timed repeats per fit (default: 1)")
    p.add_argument(
        "--warmup",
        type=int,
        default=WARMUP,
        help="untimed fits per repeat; the wall cap covers them too (default: 1)",
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
        "--solutions",
        nargs="+",
        default=None,
        help="run only these solution names (default: every solution)",
    )
    p.add_argument(
        "--machine-id",
        default=socket.gethostname(),
        help="machine id stamped on every record and in run_id (default: this host's name)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="run d=10, one seed, and a 300 s cap, replacing those flags",
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
        help="fit nothing: recompute the bic columns of the run already in --out and rewrite "
        "runs.parquet. Takes the first run's flags, as --resume does",
    )
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)

    for flag, value, least in (
        ("--seeds", args.seeds, 1),
        ("--repeats", args.repeats, 1),
        ("--warmup", args.warmup, 0),
    ):
        if value < least:
            p.error(f"{flag} must be at least {least}, got {value}")
    if args.n is not None and args.n < 1:
        p.error(f"--n must be at least 1, got {args.n}")
    if not (math.isfinite(args.lambda_value) and args.lambda_value >= 0):
        p.error(f"--lambda-value must be finite and non-negative, got {args.lambda_value}")
    if any(not density > 0 for density in args.densities):
        p.error(f"--densities must be positive, got {args.densities}")

    field = solutions(args.lambda_value)
    if args.solutions:
        known = {a.name for a in field}
        unknown = sorted(set(args.solutions) - known)
        # A typo would otherwise run fewer solutions, and the missing rows would read as failures.
        if unknown:
            p.error(f"unknown solutions {unknown}; the solutions are {sorted(known)}")
        field = [a for a in field if a.name in args.solutions]

    out = Path(args.out)
    keys = {k.digest: k for k in ladder_keys(args.ladder, args.densities, args.seeds, args.n)}

    run_meta = {
        "driver": Path(__file__).name,
        # One entry per (density, solution), so the expected unit count covers both densities.
        "solution list": [a.name for _ in args.densities for a in field],
        "params": {a.name: dict(a.params()) for a in field},
        "xges_python": {PYTHON_ENV_VAR: os.environ.get(PYTHON_ENV_VAR, "")},
        "slice": SLICE,
        "densities": args.densities,
        "ladder": args.ladder,
        "n": args.n,
        "seeds": args.seeds,
        "lambda_value": args.lambda_value,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cap_wall_s": args.cap_wall_s,
        "cap_mem_mb": args.cap_mem_mb,
        "mem_route": memcap.resolve_route(args.cap_mem_mb, args.mem_route),
        "cores": args.cores,
        "machine_id": args.machine_id,
    }
    if args.rescore:
        # Fits nothing, so the run's STATUS.json and sittings stay as the fitting runs left them.
        records_dir = out / rundir.RECORDS_DIRNAME
        if not any(records_dir.glob("*.parquet")):
            raise SystemExit(f"--rescore: {records_dir} holds no records")
        recorded = json.loads((out / rundir.RUN_META_NAME).read_text()).get("lambda_value")
        if recorded != args.lambda_value:
            raise SystemExit(
                f"--rescore: the run searched at --lambda-value {recorded}, not "
                f"{args.lambda_value}; its bic must be scored at the same weight"
            )
        add_bic(out, records_dir, keys, args.lambda_value, rescore=True)
        table = rundir.assemble(records_dir)
        table.to_parquet(out / "runs.parquet", index=False)
        _log(f"rescored    : {len(table)} rows  -> {out / 'runs.parquet'}")
        _summarize(table)
        return 0

    # Every interpreter before any rung runs, and the second environment only when it is used.
    check_interpreters(field)
    venv = xges_venv() if any(getattr(a, "python_env", None) for a in field) else None

    # Reads run_meta.json before writing it, so --resume has something to check against.
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    prov = rundir.provenance_once(
        out,
        timestamp=datetime.now(timezone.utc).isoformat(),
        extra_venvs={"xges": venv} if venv else None,
    )

    _log(f"=== XGES, Andrey and the xges package, CPU -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:14s}: {value}")
    _log(f"{'git_sha':14s}: {prov['git_sha']}")
    xges_env = prov.get("environments", {}).get("xges")
    if xges_env:
        _log(f"{'xges':14s}: {xges_env['versions'].get('packages', {}).get('xges')}")
    _log("")

    records: list = []
    errors: list[str] = []
    skipped = 0
    start = time.perf_counter()

    for d in args.ladder:
        # One rung at a time, so a rung's data is materialized only when the ladder reaches it.
        result = run_batch(
            datasets.store_dir(out),
            field,
            ladder_keys([d], args.densities, args.seeds, args.n),
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
        for density in args.densities:
            ran = []
            for adapter in field:
                mine = [
                    r for r in result.records if r.name == adapter.name and r.density == density
                ]
                walls = [r.wall_s for r in mine if r.status == "ok" and r.wall_s is not None]
                statuses = sorted({r.status for r in mine})
                cost = (
                    f"{statistics.median(walls):.3g}s"
                    if walls
                    else "/".join(statuses) or "no-records"
                )
                ran.append(f"{adapter.name}={cost}")
            _log(
                f"[d={d} deg={density:g}] {'  '.join(ran)} | "
                f"elapsed {time.perf_counter() - start:.0f}s"
            )
        add_bic(out, records_dir, keys, args.lambda_value)

    # From the parts, not from memory: a resume that measured nothing still writes the table for
    # everything measured before it.
    add_bic(out, records_dir, keys, args.lambda_value)
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

    # An errored fit is a record, not an entry in `errors`, and an earlier sitting's is only in the
    # table. Timeouts and OOMs measure a cap and are expected at the top of the ladder.
    failed = table[table.status == "error"]
    if len(failed):
        _log(f"failed fits : {len(failed)}")
        for row in failed.head(5).itertuples():
            _log(f"  ! {row.name} @ d={row.d} seed={row.data_seed}: {row.error_text}")
    return 1 if errors or len(failed) else 0


def _summarize(table: pd.DataFrame) -> None:
    """Print, per density, solution, and rung, the fits that finished and their medians.

    ``shd`` is against the true CPDAG; ``bic_gap`` is the returned graph's BIC minus the true
    graph's, so a negative gap is a graph that scores better than the truth on its data.
    """
    if table.empty:
        return
    _log("")
    _log(
        f"{'deg':>4s} {'solution':34s} {'d':>5s} {'ok/all':>7s} {'wall_s':>9s} {'shd':>7s} "
        f"{'bic_gap':>10s}"
    )
    for (density, name, d), rows in table.groupby(["density", "name", "d"], sort=True):
        ok = rows[rows["status"] == "ok"]
        _log(
            f"{density:4g} {name:34s} {d:5d} {len(ok):3d}/{len(rows):<3d} "
            f"{ok['wall_s'].median():9.3g} {ok['shd'].median():7.1f} {ok['bic_gap'].median():10.4g}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
