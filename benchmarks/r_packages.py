#!/usr/bin/env python
"""Benchmark Andrey against pcalg and bnlearn across graph sizes.

Compare speed, accuracy, and the largest graph each search finishes on the same data.

Each job runs one lane. Their graph-size limits differ by an order of magnitude, so a shared
ladder would miss larger GES sizes or exhaust its time limit on FCI timeouts:

    --lane score  Andrey GES (3 builds) and HC (2), pcalg ges, bnlearn hc and tabu
    --lane pc     Andrey PC, pcalg pc, bnlearn pc.stable
    --lane fci    Andrey FCI, pcalg fci and rfci

The score lane pairs `pcalg.ges` with Andrey GES and bnlearn hill climbers with Andrey HC.
Every lane measures `baseline.empty`.

Before running R solutions, load the R modules and export `R_LIBS_USER` using the commands from
`benchmarks/start_r.sh --env`. The driver starts R and records its environment before fitting;
an invalid environment fails once, rather than per unit.

`bnlearn.hc` and `bnlearn.tabu` return DAGs; GES, HC and PC return CPDAGs. Their `shd` units differ:
compare a DAG row's `mec_shd` with a CPDAG row's `shd`.

Use `--seed-start` for one job per seed. Keep the lane's full solution list in each task: ratios
pair on `(run_dir, dataset_id)`. Give each task its own `--out` to avoid overwriting
`run_meta.json`. Share only the dataset store, `ANDREY_BENCH_DATA`, and materialize it once with
`--prepare-only` before launching the array.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/r_packages.py --lane score \\
        --out outputs/r-score --ladder 20 50 100 --cap-mem-mb 32000
"""

from __future__ import annotations

import argparse
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from andrey_bench import datasets, memcap, rsession, rundir
from andrey_bench.adapters.andrey_fci import AndreyFCI
from andrey_bench.adapters.andrey_ges import andrey_ges_adapters
from andrey_bench.adapters.andrey_hc import andrey_hc_adapters
from andrey_bench.adapters.andrey_pc import AndreyPC
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.bnlearn_hc import BnlearnHC
from andrey_bench.adapters.bnlearn_pc_stable import BnlearnPCStable
from andrey_bench.adapters.bnlearn_tabu import BnlearnTabu
from andrey_bench.adapters.pcalg_fci import PcalgFCI
from andrey_bench.adapters.pcalg_ges import PcalgGES
from andrey_bench.adapters.pcalg_pc import PcalgPC
from andrey_bench.adapters.pcalg_rfci import PcalgRFCI
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores


def _score_solutions() -> list:
    """Return paired GES and hill-climbing solutions in one lane for within-run ratios.

    Pair pcalg's `ges` with Andrey's GES, and bnlearn's `hc` and `tabu` with `andrey.hc`.
    Hill climbing changes one DAG edge at a time; GES searches CPDAG space over neighbor subsets.
    Include all Andrey GES builds to cover every backend. Allow for the `numba` reachability
    kernel's first-call compilation during warm-up in `--cap-wall-s`.
    """
    return [
        *andrey_ges_adapters(),
        *andrey_hc_adapters(),
        PcalgGES(),
        BnlearnHC(),
        BnlearnTabu(),
        BaselineEmpty(),
    ]


#: The solution list of each lane, and the R packages that lane needs installed.
LANES: dict[str, tuple] = {
    "score": (_score_solutions, ("pcalg", "bnlearn")),
    "pc": (
        lambda: [AndreyPC(), PcalgPC(), BnlearnPCStable(), BaselineEmpty()],
        ("pcalg", "bnlearn"),
    ),
    "fci": (lambda: [AndreyFCI(), PcalgFCI(), PcalgRFCI(), BaselineEmpty()], ("pcalg",)),
}

#: Algorithm-specific parallel thresholds. GES and HC each read their own variable; assigning
#: one to both algorithms would record an unused setting.
WORK_GATE_ENV: dict[str, str] = {
    "ges": "ANDREY_GES_PARALLEL_MIN_WORK",
    "hc": "ANDREY_HC_PARALLEL_MIN_WORK",
}

#: The canonical sparse slice: linear-Gaussian on Erdos-Renyi at average degree 2, where GES, PC and
#: FCI are all identifiable and every solution above can run.
SLICE = {
    "topology": "er",
    "functional": "linear",
    "noise": "gaussian",
    "standardize": "standardized",
    "density": datasets.CANONICAL_DENSITY,
}

#: Untimed fits before each timed fit. R startup and package loading run in `setup()`, outside
#: fit timing even at `--warmup 0`.
WARMUP = 1


def select(solutions: list, only: list[str] | None) -> list:
    """Return all lane solutions or a named subset, always retaining `baseline.empty`.

    Run shared sizes together for paired ratios. At larger sizes, use `--only` for solutions
    that still finish: each timeout consumes the full `--cap-wall-s`.
    """
    if not only:
        return solutions
    by_name = {a.name: a for a in solutions}
    unknown = sorted(set(only) - set(by_name))
    if unknown:
        raise SystemExit(f"--only names solutions this lane does not have: {unknown}")
    keep = set(only) | {BaselineEmpty().name}
    return [a for a in solutions if a.name in keep]


def ladder_keys(ladder: list[int], seeds: int, seed_start: int = 0) -> list[DatasetKey]:
    """Return one dataset per `(d, seed)` on :data:`SLICE`, with `n = 10 * d`.

    Seeds span `seed_start .. seed_start + seeds - 1`, allowing one job per seed. `dataset_id`
    hashes the generator call, so each seed produces identical bytes in array and sequential runs.
    """
    return [
        DatasetKey(**SLICE, d=d, n=10 * d, seed=seed)
        for d in ladder
        for seed in range(seed_start, seed_start + seeds)
    ]


def groups(solutions: list, min_work: Mapping[str, int | None]):
    """Group solutions by environment as `(solutions, extra_env)` pairs.

    Work thresholds apply only to parallel Andrey solutions; serial fits use one worker and R
    solutions have no threshold. GES and HC read separate variables, so each algorithm with a
    threshold gets its own group and environment overlay.
    """
    gated: dict[str, list] = {}
    for adapter in solutions:
        env = WORK_GATE_ENV.get(adapter.algorithm)
        if getattr(adapter, "mode", None) == "parallel" and env and min_work.get(env) is not None:
            gated.setdefault(env, []).append(adapter)
    if not gated:
        return [(solutions, None)]
    claimed = {a.name for group in gated.values() for a in group}
    rest = [a for a in solutions if a.name not in claimed]
    return [(rest, None)] + [
        (group, {env: str(min_work[env])}) for env, group in sorted(gated.items())
    ]


def _log(msg: str = "") -> None:
    """Print to batch-job `stdout` and flush immediately."""
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="andrey vs pcalg and bnlearn across graph sizes")
    p.add_argument("--lane", choices=sorted(LANES), required=True, help="which method to measure")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument("--ladder", type=int, nargs="+", required=True, help="d sizes, run in order")
    p.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="run this subset of the lane's solutions, plus baseline.empty. Use it to take the "
        "solutions that are still finishing above the rung the rest of the lane stops at, rather "
        "than paying a wall cap per dead fit. Ratios pair inside one run, so the sizes being "
        "compared must stay in the shared run",
    )
    p.add_argument("--seeds", type=int, default=3, help="datasets per rung (default: 3)")
    p.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="first data seed, so a campaign can be cut into one job per seed, each with its own "
        "--out. A dataset_id is a content hash of the generator call, so seed k is the same data "
        "whichever job makes it",
    )
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="materialize this run's datasets and stop, without measuring anything. Run once before "
        "a job array: the .npz files are written through a rename and are safe to create "
        "concurrently, but registry.json is read-modify-write and parallel jobs would drop each "
        "other's provenance entries",
    )
    p.add_argument("--repeats", type=int, default=5, help="timed repeats per fit (default: 5)")
    p.add_argument(
        "--warmup", type=int, default=WARMUP, help="untimed fits per repeat; 0 for reach"
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
        "--ges-min-work",
        type=int,
        default=None,
        help="parallel work gate for andrey's parallel GES, in estimated candidate evaluations; "
        "omitted leaves the shipped default in force. Recorded in run_meta.json, so --resume "
        "refuses a continuation under a different value rather than pooling two gates in one table",
    )
    p.add_argument(
        "--hc-min-work",
        type=int,
        default=None,
        help="parallel work gate for andrey's parallel HC, in scanned moves per pass - a different "
        "unit and a different environment variable from the GES gate. The shipped default has "
        "never been measured on any host, so a campaign should declare one",
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
    build_solutions, r_packages = LANES[args.lane]
    solutions = select(build_solutions(), args.only)
    work_gates = {
        "ANDREY_GES_PARALLEL_MIN_WORK": args.ges_min_work,
        "ANDREY_HC_PARALLEL_MIN_WORK": args.hc_min_work,
    }
    # Check R availability and record its environment before materializing or measuring.
    r_environment = rsession.check_environment(*r_packages)

    # Record requested settings before fitting so they survive an interrupted benchmark.
    run_meta = {
        "driver": Path(__file__).name,
        "lane": args.lane,
        "solution list": [a.name for a in solutions],
        "params": {a.name: dict(a.params()) for a in solutions},
        "r environment": r_environment,
        # `aggregate.STAMPS` uses this name to group pooled results by GES threshold.
        # Record the HC threshold alongside it.
        "ges_min_work": args.ges_min_work,
        "hc_min_work": args.hc_min_work,
        "slice": SLICE,
        "ladder": args.ladder,
        "seeds": args.seeds,
        "seed_start": args.seed_start,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cap_wall_s": args.cap_wall_s,
        "cap_mem_mb": args.cap_mem_mb,
        "mem_route": memcap.resolve_route(args.cap_mem_mb, args.mem_route),
        "cores": args.cores,
        "machine_id": args.machine_id,
    }
    if args.prepare_only:
        store = datasets.store_dir(out)
        keys = ladder_keys(args.ladder, args.seeds, args.seed_start)
        registry = datasets.materialize(keys, store)
        _log(
            f"materialized {len(keys)} dataset(s) into {store}; registry has {len(registry['keys'])}"
        )
        return 0

    # Read `run_meta.json` before writing to validate `--resume` settings.
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())

    _log(f"=== andrey vs pcalg and bnlearn, {args.lane} -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:14s}: {value}")
    _log(f"{'git_sha':14s}: {prov['git_sha']}")
    _log("")

    records: list = []
    errors: list[str] = []
    skipped = 0
    start = time.perf_counter()

    for d in args.ladder:
        # Materialize data as each graph size is reached.
        rung: list = []
        for group, extra_env in groups(solutions, work_gates):
            result = run_batch(
                datasets.store_dir(out),
                group,
                ladder_keys([d], args.seeds, args.seed_start),
                machine_id=args.machine_id,
                cap_wall_s=args.cap_wall_s,
                cap_mem_mb=args.cap_mem_mb,
                mem_route=args.mem_route,
                repeats=args.repeats,
                warmup=args.warmup,
                cores=args.cores,
                records_dir=records_dir,
                run_dir=out,
                extra_env=extra_env,
            )
            rung.extend(result.records)
            errors.extend(result.errors)
            skipped += result.skipped
        records.extend(rung)

        # Fit time or failure status for each solution at this graph size.
        ran = []
        for adapter in solutions:
            mine = [r for r in rung if r.name == adapter.name]
            walls = [r.wall_s for r in mine if r.status == "ok" and r.wall_s is not None]
            statuses = sorted({r.status for r in mine})
            cost = (
                f"{statistics.median(walls):.3g}s" if walls else "/".join(statuses) or "no-records"
            )
            ran.append(f"{adapter.name}={cost}")
        _log(f"[d={d}] {'  '.join(ran)} | elapsed {time.perf_counter() - start:.0f}s")

    # Assemble saved parts so a resumed run includes earlier measurements even without new fits.
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
