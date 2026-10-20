#!/usr/bin/env python
"""Benchmark CD-NOD on CPU: Andrey against causal-learn, on data pooled from several domains.

Runs on the ``domain_shift_er`` regime (:mod:`andrey_bench.regime_models`): a linear-Gaussian
Erdos-Renyi DAG at mean degree 2, with ``n`` rows from four domains in equal blocks, where one
variable in three, at least one, shifts its mean with the domain. Both packages run Fisher-Z at
``alpha = 0.05`` with the same skeleton search and collider rule, listed in ``params``, and both
rebuild the domain index from the rows. Every estimate is scored against the CPDAG CD-NOD can
identify, where the domain index orients the edges it reaches (``shd``, ``skeleton_f1``,
``arrowhead_f1``); the empty graph is the quality baseline. Every solution is serial.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/cdnod_cpu.py \\
        --out outputs/cdnod-cpu --ladder 10 20 50 --cap-mem-mb 32000
"""

from __future__ import annotations

import argparse
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from andrey_bench import datasets, memcap, pairing, rundir
from andrey_bench.adapters.andrey_cdnod import AndreyCDNOD
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.causal_learn_cdnod import CausalLearnCDNOD
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores

#: Both packages' CD-NOD and the empty-graph baseline.
SOLUTIONS = [AndreyCDNOD(), CausalLearnCDNOD(), BaselineEmpty()]

REGIME = "domain_shift_er"

#: Smallest run that touches every solution: one small dataset.
SMOKE = {"ladder": [10], "seeds": 1, "n": 1000, "repeats": 1, "cap_wall_s": 300.0}

#: Untimed fits before each measurement. The per-repeat wall cap covers warm-up and fit together.
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


def ladder_keys(ladder: list[int], seeds: int, n: int) -> list[DatasetKey]:
    """Return one dataset of `n` rows per ``(d, seed)``."""
    return [
        DatasetKey(**regime_slice(), d=d, n=n, seed=seed) for d in ladder for seed in range(seeds)
    ]


def _log(msg: str = "") -> None:
    """Print and flush batch output immediately."""
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CD-NOD, Andrey against causal-learn, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--ladder", type=int, nargs="+", default=[10, 20, 50], help="d sizes (default: 10 20 50)"
    )
    p.add_argument("--seeds", type=int, default=5, help="datasets per rung (default: 5)")
    p.add_argument(
        "--n", type=int, default=2000, help="rows per dataset, all domains (default: 2000)"
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
        "--machine-id",
        default=socket.gethostname(),
        help="machine id stamped on every record and in run_id (default: this host's name)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="run d=10, one seed, 1000 rows, one repeat, and a 300 s cap, replacing those flags",
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

    out = Path(args.out)
    # Written before the first fit, so an interrupted benchmark still says what it was asked for.
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in SOLUTIONS],
        "params": {a.name: dict(a.params()) for a in SOLUTIONS},
        "slice": regime_slice(),
        "ladder": args.ladder,
        "n": args.n,
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

    _log(f"=== CD-NOD, Andrey and causal-learn -> {out} ===")
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
            ladder_keys([d], args.seeds, args.n),
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

    # From the parts, not from memory: a resume that measured nothing still writes the table for
    # everything measured before it.
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
    summarize(table)
    return 1 if errors else 0


def summarize(table: pd.DataFrame) -> None:
    """Print, per solution and rung, the finished fits with their medians and mean F1 scores.

    ``same`` counts the datasets where causal-learn returned Andrey's graph, of those both
    finished.
    """
    if table.empty:
        return
    _log("")
    _log(
        f"{'solution':24s} {'d':>4s} {'ok/all':>7s} {'wall_s':>8s} {'shd':>6s} "
        f"{'f1 skel':>7s} {'f1 arrow':>8s} {'same':>6s}"
    )
    for (name, d), rows in table.groupby(["name", "d"], sort=True):
        ok = rows[rows["status"] == "ok"]
        same = ""
        if name == CausalLearnCDNOD.name:
            andrey = table[(table["name"] == AndreyCDNOD.name) & (table["d"] == d)]
            same = "{}/{}".format(*pairing.same_graph(andrey, rows))
        _log(
            f"{name:24s} {d:4d} {len(ok):3d}/{len(rows):<3d} {ok['wall_s'].median():8.3g} "
            f"{ok['shd'].median():6.3g} {ok['skeleton_f1'].mean():7.3f} "
            f"{ok['arrowhead_f1'].mean():8.3f} {same:>6s}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
