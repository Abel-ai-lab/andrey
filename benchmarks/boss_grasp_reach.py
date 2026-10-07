#!/usr/bin/env python
"""Benchmark BOSS beyond GRaSP's measured range.

Runs both BOSS implementations and the empty-graph baseline. GRaSP censors first on both sides, and
a censored solution still spends the full wall cap on every seed and repeat. ``boss_grasp_cpu.py``
covers sizes where BOSS and GRaSP both finish. The drivers share their data slice, penalties, seeds,
and default warm-up, so their results can be pooled.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/boss_grasp_reach.py \\
        --out outputs/boss-grasp-hi --ladder 800 1600 --cap-mem-mb 64000
"""

from __future__ import annotations

import argparse
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from andrey_bench import datasets, memcap, rundir
from andrey_bench.adapters.andrey_boss import AndreyBOSS
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.causal_learn_boss import CausalLearnBOSS
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores

#: Both BOSS implementations and the empty baseline; omit both GRaSP sides to preserve pairing.
SOLUTIONS = [
    AndreyBOSS(),
    CausalLearnBOSS(),
    BaselineEmpty(),
]

#: Match ``boss_grasp_cpu.py`` so results from both drivers can be pooled.
SLICE = {
    "topology": "er",
    "functional": "linear",
    "noise": "gaussian",
    "standardize": "standardized",
    "density": datasets.CANONICAL_DENSITY,
}

#: Untimed fits before each measurement. Use ``--warmup 0`` when measuring maximum reach: the wall
#: cap covers warm-up too, so there warm-up only spends it.
WARMUP = 1


def ladder_keys(ladder: list[int], seeds: int) -> list[DatasetKey]:
    """Return one :data:`SLICE` dataset per ``(d, seed)`` at ``n = 10 * d``."""
    return [DatasetKey(**SLICE, d=d, n=10 * d, seed=seed) for d in ladder for seed in range(seeds)]


def _log(msg: str = "") -> None:
    """Print and flush batch output immediately."""
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="BOSS reach benchmark above GRaSP's ceiling")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument("--ladder", type=int, nargs="+", required=True, help="d sizes, run in order")
    p.add_argument("--seeds", type=int, default=3, help="datasets per rung (default: 3)")
    p.add_argument("--repeats", type=int, default=5, help="timed repeats per fit (default: 5)")
    p.add_argument(
        "--warmup",
        type=int,
        default=WARMUP,
        help="untimed fits per repeat; 0 on a reach run, where warm-up only spends the cap",
    )
    p.add_argument("--cap-wall-s", type=float, default=1800.0, help="wall cap per repeat, seconds")
    p.add_argument("--cap-mem-mb", type=float, default=None, help="memory cap per repeat, MB")
    p.add_argument("--mem-route", choices=memcap.ROUTES, default=None, help="cap enforcement route")
    p.add_argument(
        "--cores",
        type=int,
        default=available_cores(),
        help="core budget per fit, split into BLAS threads and pool workers by the solution's mode "
        "so every one gets the same machine (default: every core this job is allowed)",
    )
    p.add_argument(
        "--machine-id",
        default=socket.gethostname(),
        help="machine id stamped on every record and in run_id (default: this host's name)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue the run already in --out: skip units whose records are on disk. Requires the "
        "same run_meta.json, so pass the flags the first run was given",
    )
    args = p.parse_args(argv)

    out = Path(args.out)

    # Record the request before fitting so interrupted runs retain metadata.
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in SOLUTIONS],
        "penalty": {a.name: a.params().get("lambda_value") for a in SOLUTIONS},
        "search seed": {a.name: a.params().get("seed") for a in SOLUTIONS},
        "slice": SLICE,
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
    # Read `run_meta.json` before writing so `--resume` can validate it.
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())

    _log(f"=== BOSS/GRaSP reach, CPU -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:14s}: {value}")
    _log(f"{'git_sha':14s}: {prov['git_sha']}")
    _log("")

    records: list = []
    errors: list[str] = []
    skipped = 0
    start = time.perf_counter()

    for d in args.ladder:
        # Materialize data one rung at a time.
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

        # Report each solution's median cost or stopping status.
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

    # Rebuild from saved parts so resume includes records from earlier invocations.
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
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
