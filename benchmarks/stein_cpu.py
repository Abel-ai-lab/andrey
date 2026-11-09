#!/usr/bin/env python
"""Screen SCORE and NoGAM, through their authors' code, in the setting their papers claim.

The screen decides whether Andrey implements the two methods; it runs no Andrey implementation of
either. Every solution fits the same datasets in the same run:

    rolland.score, rolland.score.order          the authors' SCORE (Rolland et al. 2022)
    dodiscover.nogam, dodiscover.nogam.order    the authors' NoGAM (Montagna et al. 2023)
    andrey.cam.numpy, andrey.cam.order.numpy    Andrey's cam, shown beside every cell
    causaldisco.r2_sort_regress,                R2-SortnRegress (Reisach et al. 2023), the
    causaldisco.r2_sort.order                   sortability control
    baseline.empty                              the empty graph, the floor for SHD

A ``.order`` solution returns the full DAG of the method's causal order
(``andrey_bench.adapters.order_dag``): its arrowhead recall is the share of true edges the order
gets right, one minus the papers' order divergence over the true edges.

The datasets are the papers' simulation (``andrey_bench.gp_models``): links drawn from a Gaussian
process, noise variances uniform on [0.4, 0.8], ``n = 1000`` rows, Erdos-Renyi graphs with ``d``
edges (``--densities er1``) or ``4 d`` (``er4``). The regimes ``score_paper``, ``score_paper_laplace``,
and ``score_paper_gumbel`` are the papers' raw scale, which reproduces their numbers; the
``_standardized`` and ``_rescaled`` regimes are the decision cells. SCORE's claim is Gaussian noise
at ``d = 10, 20, 50``, NoGAM's any noise at ``d = 10, 20``.

``--oracle`` runs the oracle arm instead of the solutions (``andrey_bench.score_oracle``): SCORE's
leaf rule on the exact Hessian of the true model, on the additive-noise regimes ``anm_er``,
``anm_laplace_er``, and ``anm_gumbel_er``, where the Hessian has a closed form. It writes one JSON
line per dataset to ``oracle.jsonl`` in ``--out``; the solutions fit the same datasets in a run of
their own.

A method passes the screen when, in most of its decision cells, its SHD interval lies below the
empty graph's and the interval of its order's correct share lies above R2-SortnRegress's.

The reference solutions run in the interpreter ``$ANDREY_BENCH_STEIN_PYTHON`` names and read the
SCORE files from ``$ANDREY_BENCH_SCORE``; `start_stein.sh` builds both and prints the exports.
SCORE's pruning starts ``Rscript``, which needs ``mgcv``: load R as `start_r.sh --env` prints.
Andrey's ``cam`` runs in ``$ANDREY_BENCH_CAM_PYTHON``, an environment holding the build under test.

Use ``--seed-start``, ``--regimes``, ``--densities``, and ``--solutions`` to cut a campaign into
small runs, each with its own ``--out``. Share the dataset store, ``ANDREY_BENCH_DATA``, and fill it
once with ``--prepare-only`` before launching them.

    $ANDREY_BENCH_PYTHON benchmarks/stein_cpu.py --out outputs/stein \\
        --regimes score_paper_standardized --ladder 10 --densities er1 --seeds 2 --cores 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from andrey_bench import datasets, memcap, rundir, score_oracle
from andrey_bench.adapters.andrey_cam import AndreyCAM, AndreyCAMOrder
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.causaldisco_r2sort import CausalDiscoR2Sort, CausalDiscoR2SortnRegress
from andrey_bench.adapters.dodiscover_nogam import DodiscoverNoGAM, DodiscoverNoGAMOrder
from andrey_bench.adapters.rolland_score import RollandSCORE, RollandSCOREOrder, code_dir
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores

#: Every solution, in the order each dataset runs them.
SOLUTIONS = [
    RollandSCORE(),
    RollandSCOREOrder(),
    DodiscoverNoGAM(),
    DodiscoverNoGAMOrder(),
    AndreyCAM(),
    AndreyCAMOrder(),
    CausalDiscoR2SortnRegress(),
    CausalDiscoR2Sort(),
    BaselineEmpty(),
]

#: The papers' simulation (`datasets.REGIMES`): raw first, then the decision scales.
REGIMES = [
    "score_paper",
    "score_paper_standardized",
    "score_paper_rescaled",
    "score_paper_laplace",
    "score_paper_laplace_standardized",
    "score_paper_laplace_rescaled",
    "score_paper_gumbel",
    "score_paper_gumbel_standardized",
    "score_paper_gumbel_rescaled",
]

#: The oracle's additive-noise regimes, z-scored, whose Hessian has a closed form.
ORACLE_REGIMES = ["anm_er", "anm_laplace_er", "anm_gumbel_er"]

#: The papers' graphs, by mean degree: ``d`` edges (ER1) and ``4 d`` (ER4).
DENSITIES = {"er1": 2.0, "er4": 8.0}

#: Rows per dataset, both papers' ``n``.
ROWS = 1000

#: Untimed fits before each measurement. Each solution warms on a tiny table in its setup.
WARMUP = 0


def regime_keys(regime: str, d: int, density: float, seeds: range) -> list[DatasetKey]:
    """Return one dataset per seed of `regime` at `d` variables, `density`, and :data:`ROWS` rows."""
    topology, functional, noise, standardize = datasets.REGIMES[regime]
    return [
        DatasetKey(
            topology=topology,
            functional=functional,
            noise=noise,
            standardize=standardize,
            density=density,
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
        raise SystemExit(f"${var} is unset: export the interpreter `start_stein.sh` prints")
    return Path(raw).absolute().parent.parent


def run_oracle(out: Path, keys: list[DatasetKey]) -> int:
    """Write one :func:`~andrey_bench.score_oracle.oracle_steps` line per key to ``oracle.jsonl``.

    Keys already in the file are skipped, so an interrupted run continues where it stopped.
    """
    out.mkdir(parents=True, exist_ok=True)
    path = out / "oracle.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["dataset_id"] for line in path.read_text().splitlines() if line}
    with path.open("a") as fh:
        for key in keys:
            if key.digest in done:
                continue
            row = score_oracle.oracle_steps(key)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            _log(f"[oracle] {key.label}: {row['wrong_steps']} wrong of {row['steps']} steps")
    return 0


def main(argv: list[str] | None = None) -> int:
    names = [a.name for a in SOLUTIONS]
    p = argparse.ArgumentParser(description="SCORE and NoGAM in their papers' setting")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument("--ladder", type=int, nargs="+", required=True, help="d sizes, run in order")
    p.add_argument(
        "--regimes",
        nargs="+",
        choices=REGIMES + ORACLE_REGIMES,
        default=REGIMES,
        help="regimes to run, in order (default: the papers' simulation at every scale)",
    )
    p.add_argument(
        "--densities",
        nargs="+",
        choices=sorted(DENSITIES),
        default=sorted(DENSITIES),
        help="the papers' graph densities: er1 has d edges, er4 4 d (default: both)",
    )
    p.add_argument(
        "--solutions",
        nargs="+",
        choices=names,
        default=names,
        help="solutions to run (default: all); NoGAM's claim stops at d = 20",
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
    p.add_argument(
        "--oracle",
        action="store_true",
        help="run the oracle arm on the additive-noise regimes instead of the solutions",
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
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    cells = [
        (regime, d, DENSITIES[density])
        for d in args.ladder
        for density in args.densities
        for regime in args.regimes
    ]

    if args.prepare_only:
        store = datasets.store_dir(out)
        keys = [k for regime, d, density in cells for k in regime_keys(regime, d, density, seeds)]
        registry = datasets.materialize(keys, store)
        _log(
            f"materialized {len(keys)} dataset(s) into {store}; registry has {len(registry['keys'])}"
        )
        return 0
    if args.oracle:
        if not set(args.regimes) <= set(ORACLE_REGIMES):
            raise SystemExit(f"--oracle runs on {ORACLE_REGIMES} alone")
        keys = [k for regime, d, density in cells for k in regime_keys(regime, d, density, seeds)]
        return run_oracle(out, keys)

    solutions = [a for a in SOLUTIONS if a.name in args.solutions]
    # Check the environment before the first fit: one error here, not one per unit.
    if any(isinstance(a, RollandSCORE) for a in solutions):
        code_dir()
        if shutil.which("Rscript") is None:
            raise SystemExit("SCORE's pruning starts Rscript: load R as start_r.sh --env prints")
    venvs = {
        var: _venv_of(var)
        for var in sorted({getattr(a, "python_env", None) for a in solutions} - {None})
    }

    # Written before the first fit, so an interrupted benchmark still says what it was asked for.
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in solutions],
        "params": {a.name: dict(a.params()) for a in solutions},
        "regimes": args.regimes,
        "densities": args.densities,
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
        extra_venvs={var.lower(): root for var, root in venvs.items()},
    )

    _log(f"=== Stein-score screen -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:14s}: {value}")
    _log(f"{'git_sha':14s}: {prov['git_sha']}")
    _log("")

    records: list = []
    errors: list[str] = []
    skipped = 0
    start = time.perf_counter()

    for regime, d, density in cells:
        result = run_batch(
            datasets.store_dir(out),
            solutions,
            regime_keys(regime, d, density, seeds),
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
            cost = f"{statistics.median(walls):.3g}s" if walls else "/".join(statuses) or "none"
            ran.append(f"{adapter.name}={cost}")
        _log(
            f"[{regime} d={d} deg={density:g}] {'  '.join(ran)} | "
            f"elapsed {time.perf_counter() - start:.0f}s"
        )

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
