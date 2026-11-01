#!/usr/bin/env python
"""Benchmark the functional-causal-model methods CAM and RESIT against their reference packages.

Each lane runs one method on the same datasets in the same job:

    --lane cam    Andrey cam, CAM 1.0 (R, CRAN archive), the empty graph
    --lane resit  Andrey resit, the authors' R RESIT, lingam RESIT, the empty graph

The datasets are drawn by ``andrey.data.sample_scm`` on Erdos-Renyi graphs of mean degree 2 with
``n = 1000`` rows, at the regimes below, z-scored (the default) and at raw scale. Every method
returns a DAG, scored against the true DAG.

    anm_er, anm_er_raw                  additive noise, Gaussian
    anm_gumbel_er, anm_gumbel_er_raw    additive noise, Gumbel
    lsnm_er, lsnm_er_raw                location-scale noise
    linear_gauss_er, linear_gauss_er_raw  linear Gaussian, where the DAG is not identifiable

The Andrey fits run in the interpreter named by ``$ANDREY_BENCH_CAM_PYTHON`` or
``$ANDREY_BENCH_RESIT_PYTHON``, an environment holding the build under test; the driver's own
environment draws and scores the data and runs the references. Load R as `start_r.sh --env` prints,
install CAM 1.0 with `start_cam_r.sh`, and fetch the authors' RESIT code with `start_resit_r.sh`.

Use `--seed-start` and `--regimes` to cut a campaign into small runs, each with its own `--out`;
ratios pair inside one run, so keep a lane's full solution list in each. Share the dataset store,
`ANDREY_BENCH_DATA`, and fill it once with `--prepare-only` before launching them.

    $ANDREY_BENCH_PYTHON benchmarks/fcm_cpu.py --lane cam --out outputs/cam \\
        --ladder 10 20 --seeds 3 --repeats 1 --warmup 0 --cores 1
"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from andrey_bench import datasets, memcap, rsession, rundir
from andrey_bench.adapters.andrey_cam import AndreyCAM
from andrey_bench.adapters.andrey_resit import AndreyRESIT
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.cam_cam import CamR
from andrey_bench.adapters.lingam_resit import LingamRESIT
from andrey_bench.adapters.peters_resit import PetersRESIT, code_dir
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores

#: Each lane's solutions, the R packages it loads, and the variable naming its Andrey build.
LANES = {
    "cam": ([AndreyCAM(), CamR(), BaselineEmpty()], ("CAM",), "ANDREY_BENCH_CAM_PYTHON"),
    "resit": (
        [AndreyRESIT(), PetersRESIT(), LingamRESIT(), BaselineEmpty()],
        ("mgcv", "kernlab"),
        "ANDREY_BENCH_RESIT_PYTHON",
    ),
}

#: The regimes, in `datasets.REGIMES`: z-scored first, then the same four at raw scale.
REGIMES = [
    "anm_er",
    "anm_gumbel_er",
    "lsnm_er",
    "linear_gauss_er",
    "anm_er_raw",
    "anm_gumbel_er_raw",
    "lsnm_er_raw",
    "linear_gauss_er_raw",
]

#: Rows per dataset.
ROWS = 1000

#: Untimed fits before each measurement. The fits are deterministic and seconds to minutes long, so
#: none run.
WARMUP = 0


def regime_keys(regime: str, d: int, seeds: range) -> list[DatasetKey]:
    """Return one dataset per seed of `regime` at `d` variables and :data:`ROWS` rows."""
    topology, functional, noise, standardize = datasets.REGIMES[regime]
    return [
        DatasetKey(
            topology=topology,
            functional=functional,
            noise=noise,
            standardize=standardize,
            density=datasets.CANONICAL_DENSITY,
            d=d,
            n=ROWS,
            seed=seed,
        )
        for seed in seeds
    ]


def _log(msg: str = "") -> None:
    """Print and flush batch output immediately."""
    print(msg, flush=True)


def _venv_of(var: str) -> Path:
    """The environment root of the interpreter `$var` names."""
    raw = os.environ.get(var, "").strip()
    if not raw:
        raise SystemExit(f"${var} is unset: export the interpreter of the build under test")
    return Path(raw).absolute().parent.parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CAM and RESIT against their reference packages")
    p.add_argument("--lane", choices=sorted(LANES), required=True, help="which method to measure")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument("--ladder", type=int, nargs="+", required=True, help="d sizes, run in order")
    p.add_argument(
        "--regimes",
        nargs="+",
        choices=REGIMES,
        default=REGIMES,
        help="regimes to run, in order (default: all eight)",
    )
    p.add_argument("--seeds", type=int, default=20, help="datasets per cell (default: 20)")
    p.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="first data seed, so a campaign can be cut into runs of a few seeds, each with its own "
        "--out. A dataset_id is a content hash of the generator call, so seed k is the same data "
        "whichever run makes it",
    )
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="materialize this run's datasets and stop, without measuring anything",
    )
    p.add_argument("--repeats", type=int, default=1, help="timed repeats per fit (default: 1)")
    p.add_argument("--warmup", type=int, default=WARMUP, help="untimed fits per repeat")
    p.add_argument("--cap-wall-s", type=float, default=1800.0, help="wall cap per repeat, seconds")
    p.add_argument("--cap-mem-mb", type=float, default=None, help="memory cap per repeat, MB")
    p.add_argument("--mem-route", choices=memcap.ROUTES, default=None, help="cap enforcement route")
    p.add_argument(
        "--cores",
        type=int,
        default=available_cores(),
        help="core budget per fit; every solution is serial (default: every core this job is "
        "allowed)",
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

    out = Path(args.out)
    solutions, r_packages, python_var = LANES[args.lane]
    seeds = range(args.seed_start, args.seed_start + args.seeds)

    if args.prepare_only:
        store = datasets.store_dir(out)
        keys = [k for r in args.regimes for d in args.ladder for k in regime_keys(r, d, seeds)]
        registry = datasets.materialize(keys, store)
        _log(
            f"materialized {len(keys)} dataset(s) into {store}; registry has {len(registry['keys'])}"
        )
        return 0

    # Check the environment before the first fit: one error here, not one per unit.
    if args.lane == "resit":
        code_dir()
    r_environment = rsession.check_environment(*r_packages)
    build_venv = _venv_of(python_var)

    # Written before the first fit, so an interrupted benchmark still says what it was asked for.
    run_meta = {
        "driver": Path(__file__).name,
        "lane": args.lane,
        "solution list": [a.name for a in solutions],
        "params": {a.name: dict(a.params()) for a in solutions},
        "r environment": r_environment,
        "regimes": args.regimes,
        "rows": ROWS,
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
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    prov = rundir.provenance_once(
        out,
        timestamp=datetime.now(timezone.utc).isoformat(),
        extra_venvs={args.lane: build_venv},
    )

    _log(f"=== {args.lane}: Andrey against its references -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:14s}: {value}")
    _log(f"{'git_sha':14s}: {prov['git_sha']}")
    _log("")

    records: list = []
    errors: list[str] = []
    skipped = 0
    start = time.perf_counter()

    for d in args.ladder:
        for regime in args.regimes:
            result = run_batch(
                datasets.store_dir(out),
                solutions,
                regime_keys(regime, d, seeds),
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

            # What each solution cost in this cell, or what stopped it.
            ran = []
            for adapter in solutions:
                mine = [r for r in result.records if r.name == adapter.name]
                walls = [r.wall_s for r in mine if r.status == "ok" and r.wall_s is not None]
                statuses = sorted({r.status for r in mine})
                cost = (
                    f"{statistics.median(walls):.3g}s"
                    if walls
                    else "/".join(statuses) or "no-records"
                )
                ran.append(f"{adapter.name}={cost}")
            _log(f"[{regime} d={d}] {'  '.join(ran)} | elapsed {time.perf_counter() - start:.0f}s")

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
