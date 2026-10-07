#!/usr/bin/env python
"""Benchmark FCI implementations on data with hidden confounders.

Runs Andrey and causal-learn FCI on identical data with Fisher-Z at ``alpha = 0.05``. The empty
graph provides a quality baseline. ``--include-majority`` also measures Andrey's optional majority
collider rule as a separate solution, using the same data and resource limits.

The ``latent_gauss_er`` regime uses an Erdos-Renyi linear-Gaussian graph with ``d`` nodes. One node
in ten is hidden; ``d`` includes hidden nodes. Each rung has ``d - latents`` data columns. Every
estimate is scored against the PAG of the observed nodes. The projection runs untimed in the parent,
once per dataset.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/fci_latent.py \\
        --out outputs/fci-latent --ladder 20 50 100 --cap-mem-mb 32000
"""

from __future__ import annotations

import argparse
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from andrey_bench import datasets, memcap, rundir
from andrey_bench.adapters.andrey_fci import AndreyFCI
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.causal_learn_fci import CausalLearnFCI
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores

#: Both FCI implementations, plus an empty-graph quality baseline.
SOLUTIONS = [AndreyFCI(), CausalLearnFCI(), BaselineEmpty()]

#: Each key carries this regime's hidden-node count.
REGIME = "latent_gauss_er"

#: Coordinates of ``latent_gauss_er``; these and the hidden-node count identify each dataset.
SLICE = {
    "topology": "er",
    "functional": "linear",
    "noise": "gaussian",
    "standardize": "standardized",
    "density": datasets.CANONICAL_DENSITY,
}

#: Untimed fits before each measurement. The per-repeat wall cap covers warm-up and fit together.
WARMUP = 1


def ladder_keys(ladder: list[int], seeds: int) -> list[DatasetKey]:
    """Return one :data:`SLICE` dataset per ``(d, seed)`` at ``n = 10 * d``.

    One node in ten is hidden.
    """
    return [
        DatasetKey(**SLICE, d=d, n=10 * d, seed=seed, latents=datasets.latents_for(REGIME, d))
        for d in ladder
        for seed in range(seeds)
    ]


def _log(msg: str = "") -> None:
    """Print and flush batch output immediately."""
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="FCI comparison on data with hidden confounders")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument("--ladder", type=int, nargs="+", required=True, help="d sizes, run in order")
    p.add_argument("--seeds", type=int, default=3, help="datasets per rung (default: 3)")
    p.add_argument("--repeats", type=int, default=5, help="timed repeats per fit (default: 5)")
    p.add_argument(
        "--include-majority",
        action="store_true",
        help="also measure Andrey FCI with collider_rule=majority as a separate solution",
    )
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
        help="continue the run already in --out: skip units whose records are on disk. Requires "
        "the same run_meta.json, so pass the flags the first run was given",
    )
    args = p.parse_args(argv)
    solutions = list(SOLUTIONS)
    if args.include_majority:
        solutions.insert(1, AndreyFCI(collider_rule="majority"))

    out = Path(args.out)

    # Written before the first fit, so an interrupted benchmark still says what it was asked for.
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in solutions],
        "regime": REGIME,
        "slice": SLICE,
        "ladder": args.ladder,
        "latents": [datasets.latents_for(REGIME, d) for d in args.ladder],
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

    _log(f"=== FCI comparison, hidden confounders -> {out} ===")
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
            solutions,
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
        for adapter in solutions:
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
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
