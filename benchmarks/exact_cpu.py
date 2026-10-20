#!/usr/bin/env python
"""Benchmark exact search on CPU: Andrey against causal-learn.

Runs Andrey's `exact_search` and causal-learn's `bic_exact_search`, both with A*, and the empty
graph as a quality baseline. Both find a DAG with the lowest BIC, so on the same data they should
reach the same score; they may return different DAGs of the optimal class. The data is the R lane's
slice, linear-Gaussian on Erdos-Renyi graphs at average degree 2 with `n = 10 * d` rows, on the
small graphs exact search can finish: its cost at least doubles with each added variable.

Each fit also gets `bic`: the BIC of the DAG it returned on the data it was fitted on, in Andrey's
deviance units (lower is better). Compare the two packages' `bic` on each dataset, and the graphs by
`mec_shd`.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/exact_cpu.py \\
        --out outputs/exact-cpu --cap-mem-mb 32000
"""

from __future__ import annotations

import argparse
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tabu_cpu import add_bic, ladder_keys, regime_slice

from andrey_bench import datasets, memcap, rundir
from andrey_bench.adapters.andrey_exact_search import AndreyExactSearch
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.causal_learn_exact_search import CausalLearnExactSearch
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores

#: The solutions: the exact-search pair, then the empty graph.
SOLUTIONS = [AndreyExactSearch(), CausalLearnExactSearch(), BaselineEmpty()]

#: The solution whose `bic` the summary subtracts.
REFERENCE = "causal-learn.exact_search"

#: Smallest run that touches every solution: one tiny dataset.
SMOKE = {"ladder": [5], "seeds": 1, "repeats": 1, "cap_wall_s": 300.0}


def _log(msg: str = "") -> None:
    """Print and flush batch output immediately."""
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Exact search, Andrey against causal-learn, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--ladder",
        type=int,
        nargs="+",
        default=[5, 10, 15],
        help="d sizes, each with n = 10 * d rows (default: 5 10 15)",
    )
    p.add_argument("--seeds", type=int, default=5, help="datasets per rung (default: 5)")
    p.add_argument("--repeats", type=int, default=1, help="timed repeats per fit (default: 1)")
    p.add_argument("--warmup", type=int, default=1, help="untimed fits per repeat (default: 1)")
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
        default=None,
        help="run only these solution names, comma-separated (default: every solution)",
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
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)

    field = SOLUTIONS
    if args.solutions is not None:
        chosen = set(args.solutions.split(","))
        known = {a.name for a in SOLUTIONS}
        unknown = sorted(chosen - known)
        if unknown:
            p.error(f"unknown solutions {unknown}; the solutions are {sorted(known)}")
        field = [a for a in SOLUTIONS if a.name in chosen]

    out = Path(args.out)
    keys = {key.digest: key for key in ladder_keys(args.ladder, args.seeds)}
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in field],
        "params": {a.name: dict(a.params()) for a in field},
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
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())

    _log(f"=== Exact search, Andrey and causal-learn -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:12s}: {value}")
    _log(f"{'git_sha':12s}: {prov['git_sha']}")
    _log("")

    errors: list[str] = []
    start = time.perf_counter()
    for d in args.ladder:
        result = run_batch(
            datasets.store_dir(out),
            field,
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
        errors.extend(result.errors)
        ran = []
        for adapter in field:
            mine = [r for r in result.records if r.name == adapter.name]
            walls = [r.wall_s for r in mine if r.status == "ok" and r.wall_s is not None]
            statuses = sorted({r.status for r in mine})
            cost = (
                f"{statistics.median(walls):.3g}s" if walls else "/".join(statuses) or "no-records"
            )
            ran.append(f"{adapter.name}={cost}")
        _log(f"[d={d}] {'  '.join(ran)} | elapsed {time.perf_counter() - start:.0f}s")
        add_bic(out, records_dir, keys)

    add_bic(out, records_dir, keys)
    table = rundir.assemble(records_dir)
    table.to_parquet(out / "runs.parquet", index=False)
    rundir.finish_run(out)

    _log("")
    _log(f"elapsed     : {time.perf_counter() - start:.1f}s")
    _log(f"table       : {len(table)} rows  -> {out / 'runs.parquet'}")
    _log(f"errors      : {len(errors)}")
    for err in errors:
        _log(f"  ! {err}")
    _summarize(table)
    return 1 if errors else 0


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """One row per solution and rung: finished fits, median wall time, graph error, and BIC gap.

    ``graph_shd`` is the median ``mec_shd`` of a DAG row and ``shd`` of a CPDAG row (the empty
    graph). ``bic_vs_reference`` is the largest absolute difference, over the datasets both
    finished, between the solution's ``bic`` and :data:`REFERENCE`'s: zero when both reach the
    same optimum on every dataset.
    """
    ok = table[table["status"] == "ok"].copy()
    ok["graph_shd"] = np.where(ok["output_type"] == "dag", ok["mec_shd"], ok["shd"])
    per_dataset = ok.groupby(["name", "dataset_id"])["bic"].first()
    reference = per_dataset.xs(REFERENCE, level="name") if REFERENCE in ok["name"].values else None
    rows = []
    for (name, d), fits in table.groupby(["name", "d"], sort=True):
        done = ok[(ok["name"] == name) & (ok["d"] == d)]
        gap = float("nan")
        if reference is not None and not done.empty:
            mine = per_dataset.xs(name, level="name").reindex(done["dataset_id"].unique())
            diff = (mine - reference.reindex(mine.index)).dropna()
            gap = float(diff.abs().max()) if len(diff) else float("nan")
        rows.append(
            {
                "name": name,
                "d": int(d),
                "ok": len(done),
                "fits": len(fits),
                "wall_s": done["wall_s"].median(),
                "graph_shd": done["graph_shd"].median(),
                "bic_vs_reference": gap,
            }
        )
    return pd.DataFrame(rows)


def _summarize(table: pd.DataFrame) -> None:
    """Print :func:`summary`, one line per solution and rung."""
    if table.empty:
        return
    _log("")
    _log(f"{'solution':28s} {'d':>4s} {'ok/all':>7s} {'wall_s':>8s} {'shd':>6s} {'|bic-ref|':>10s}")
    for row in summary(table).itertuples():
        _log(
            f"{row.name:28s} {row.d:4d} {row.ok:3d}/{row.fits:<3d} {row.wall_s:8.3g} "
            f"{row.graph_shd:6.3g} {row.bic_vs_reference:10.3g}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
