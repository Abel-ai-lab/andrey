#!/usr/bin/env python
"""A comparison of GES speed and accuracy across graph sizes on CPU.

Four solutions run on identical data: serial and parallel Andrey GES, causal-learn, and an
empty-graph baseline. The comparison measures fit time, graph accuracy, and the largest graph
each solution finishes within the resource limits.

``--cap-mem-mb`` sets a memory cap so the largest completed graph is measured under a declared
limit rather than whatever memory happens to be available on the host.

``--ges-min-work`` sets a threshold on the estimated number of candidate evaluations.
``andrey.ges.numpy.parallel`` uses worker processes only above that threshold; otherwise it
evaluates candidates in the calling process, like ``andrey.ges.numpy.serial``. Without the flag
the fits run under the default in ``src/andrey/search/_parallel_ges.py``, since the runner drops
an inherited ``ANDREY_GES_PARALLEL_MIN_WORK``. ``0`` sends every pass to the workers. The chosen
threshold belongs beside the timing results because it determines which passes run in parallel.
``andrey calibrate`` measures a suitable value for the target host.

``--cap-wall-s`` covers imports, warm-up, and the timed fit. Startup can dominate at small graph
sizes, so a cap based only on expected fit time can terminate a solution before timing begins.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/ges_cpu.py \\
        --out outputs/ges-cpu --ladder 20 50 100 200 --cap-mem-mb 32000
"""

from __future__ import annotations

import argparse
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from andrey_bench import datasets, memcap, rundir
from andrey_bench.adapters.andrey_ges import andrey_ges_adapters
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.causal_learn_ges import CausalLearnGES
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores

#: ``andrey_ges_adapters()`` supplies serial and parallel GES with the same ``numpy`` backend and
#: score settings. Every solution runs on every dataset.
SOLUTIONS = [*andrey_ges_adapters(), CausalLearnGES(), BaselineEmpty()]

#: The solutions the work gate reaches: parallel-mode Andrey only. A serial fit is pinned to one
#: worker and never consults the gate, and no competitor reads it, so declaring it on them would
#: claim a knob they do not have.
GATED = [a.name for a in SOLUTIONS if getattr(a, "mode", None) == "parallel"]


def groups(min_work: int | None):
    """Solutions split by the environment each declares, as ``(solutions, extra_env)`` pairs."""
    if min_work is None:
        return [(SOLUTIONS, None)]
    gated = [a for a in SOLUTIONS if a.name in GATED]
    rest = [a for a in SOLUTIONS if a.name not in GATED]
    return [(rest, None), (gated, {"ANDREY_GES_PARALLEL_MIN_WORK": str(min_work)})]


#: The canonical sparse slice: linear-Gaussian on Erdos-Renyi at average degree 2, the regime where
#: GES is identifiable and every solution above can run.
SLICE = {
    "topology": "er",
    "functional": "linear",
    "noise": "gaussian",
    "standardize": "standardized",
    "density": datasets.CANONICAL_DENSITY,
}

#: Untimed fits before each timed fit, excluding first-call setup from the reported fit time.
WARMUP = 1


def ladder_keys(ladder: list[int], seeds: int) -> list[DatasetKey]:
    """One dataset per ``(d, seed)`` on :data:`SLICE`, at ``n = 10·d`` and seeds ``0 .. seeds-1``."""
    return [DatasetKey(**SLICE, d=d, n=10 * d, seed=seed) for d in ladder for seed in range(seeds)]


def _log(msg: str = "") -> None:
    """Print and flush immediately — a batch job's stdout is only useful unbuffered."""
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GES comparison, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument("--ladder", type=int, nargs="+", required=True, help="d sizes, run in order")
    p.add_argument("--seeds", type=int, default=3, help="datasets per rung (default: 3)")
    p.add_argument("--repeats", type=int, default=5, help="timed repeats per fit (default: 5)")
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
        "--ges-min-work",
        type=int,
        default=None,
        help="parallel work gate (estimated candidate evals) for andrey's parallel mode; omitted "
        "leaves the shipped default in force. Recorded in run_meta.json, so --resume refuses a "
        "continuation under a different value rather than pooling two gates in one table",
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

    # Written before the first fit, so an interrupted benchmark still says what it was asked for.
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in SOLUTIONS],
        "ges_min_work": args.ges_min_work,
        "gated solutions": GATED if args.ges_min_work is not None else [],
        "slice": SLICE,
        "ladder": args.ladder,
        "seeds": args.seeds,
        "repeats": args.repeats,
        "warmup": WARMUP,
        "cap_wall_s": args.cap_wall_s,
        "cap_mem_mb": args.cap_mem_mb,
        "mem_route": memcap.resolve_route(args.cap_mem_mb, args.mem_route),
        "cores": args.cores,
        "machine_id": args.machine_id,
    }
    # Reads run_meta.json before writing it, so --resume has something to check against.
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())

    _log(f"=== GES comparison, CPU -> {out} ===")
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
        rung: list = []
        for group, extra_env in groups(args.ges_min_work):
            result = run_batch(
                datasets.store_dir(out),
                group,
                ladder_keys([d], args.seeds),
                machine_id=args.machine_id,
                cap_wall_s=args.cap_wall_s,
                cap_mem_mb=args.cap_mem_mb,
                mem_route=args.mem_route,
                repeats=args.repeats,
                warmup=WARMUP,
                cores=args.cores,
                records_dir=records_dir,
                run_dir=out,
                extra_env=extra_env,
            )
            rung.extend(result.records)
            errors.extend(result.errors)
            skipped += result.skipped
        records.extend(rung)

        # What each solution cost at this rung, or what stopped it.
        ran = []
        for adapter in SOLUTIONS:
            mine = [r for r in rung if r.name == adapter.name]
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
