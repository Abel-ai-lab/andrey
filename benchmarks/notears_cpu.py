#!/usr/bin/env python
"""Benchmark Andrey's NOTEARS and DAGMA against the authors' code in the papers' setting.

Every arm fits the same datasets in the same run:

    andrey.notears, andrey.dagma                Andrey at its defaults
    zheng.notears                               the authors' NOTEARS (Zheng et al. 2018)
    bello.dagma                                 the authors' DAGMA (Bello et al. 2022)
    andrey.notears.standardize,                 Andrey with ``standardize=True``, on the scales
    andrey.dagma.standardize                    whose columns keep their own variances
    causaldisco.r2_sort_regress,                R2-SortnRegress and Var-SortnRegress (Reisach et
    causaldisco.var_sort_regress                al. 2021, 2023), the sortability controls
    baseline.empty                              the empty graph, the floor for SHD

``--robust`` adds ``andrey.notears.robust`` and ``andrey.dagma.robust``, the likelihood path of Deng
et al. (2024), on the standardized cells of at most :data:`ROBUST_MAX_D` variables.

The datasets are the papers' simulation, drawn by ``andrey.data.sample_scm``: a linear model with
weights from +/-[0.5, 2] and unit noise variances, ``n = 1000`` rows, Erdos-Renyi graphs with
``2 d`` edges (``er2``) and scale-free graphs with ``4 d`` (``sf4``), Gaussian, exponential, and
Gumbel noise. A cell is a graph, a size, a noise, and a scale; its datasets differ by seed. The
scale ``raw`` is the papers' own, where the variances grow along the causal order (Reisach et al.
2021); ``standardized`` z-scores each column and holds the decision cells; ``rescaled`` divides each
variable by its standard deviation as it is generated; ``unitless`` draws weights and noise scales
that give every variable unit variance.

The reference and control arms run in the interpreter ``$ANDREY_BENCH_NOTEARS_PYTHON`` names, and
the NOTEARS reference loads the authors' file from ``$ANDREY_BENCH_NOTEARS``; `start_notears.sh`
builds both and prints the exports.

``--part`` and ``--parts`` cut the plan into runs, each with its own ``--out``; share the dataset
store, ``ANDREY_BENCH_DATA``, between them. A run that visits every unit writes ``done.json``.
`notears_summary.py` pairs the records into rows.

    $ANDREY_BENCH_PYTHON benchmarks/notears_cpu.py --out outputs/notears \\
        --sizes 10 --graphs er2 --noises gaussian --scales standardized --seeds 2 --cores 1
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import socket
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from andrey_bench import datasets, rundir
from andrey_bench.adapters.andrey_notears import METHODS, AndreyContinuous
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.bello_dagma import BelloDAGMA
from andrey_bench.adapters.causaldisco_r2sort import (
    CausalDiscoR2SortnRegress,
    CausalDiscoVarSortnRegress,
)
from andrey_bench.adapters.zheng_notears import ZhengNOTEARS, code_file
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores

#: The papers' graphs as ``(topology, mean degree)``: ER2 has ``2 d`` edges and SF4 ``4 d``.
GRAPHS = {"er2": ("er", 4.0), "sf4": ("scale_free", 8.0)}
#: The papers' noise families.
NOISES = ("gaussian", "exponential", "gumbel")
#: Scale name -> the key's ``standardize``. The first three run by default.
SCALES = {
    "raw": "raw",
    "standardized": "standardized",
    "rescaled": "ancestral",
    "unitless": "unitless",
}
DEFAULT_SCALES = ("raw", "standardized", "rescaled")
#: The scale of the decision cells.
DECISION_SCALE = "standardized"
#: The scales whose columns already have unit variance, where ``standardize=True`` changes nothing.
UNIT_VARIANCE_SCALES = ("standardized", "unitless")
#: Variables per dataset, and rows, the papers' ``n``.
SIZES = (10, 20, 50)
ROWS = 1000
#: The largest size the robust arms run at: a path step costs a matrix exponential per iteration.
ROBUST_MAX_D = 20
#: The interpreter of the reference and control arms.
REFERENCE_ENV = "ANDREY_BENCH_NOTEARS_PYTHON"
#: Each method's reference arm.
REFERENCES = {"notears": "zheng.notears", "dagma": "bello.dagma"}


@dataclass(frozen=True)
class Unit:
    """One dataset and the arms fitted on it."""

    key: DatasetKey
    arms: tuple[Any, ...]


def _control(adapter: Any) -> Any:
    """`adapter` set to run in the references' environment, which holds CausalDisco."""
    adapter.python_env = REFERENCE_ENV
    return adapter


def arms(scale: str, d: int, *, robust: bool = False) -> tuple[Any, ...]:
    """The arms of a dataset at `scale` with `d` variables, in the order they run."""
    out: list[Any] = []
    for method in METHODS:
        out.append(AndreyContinuous(method))
        if scale not in UNIT_VARIANCE_SCALES:
            out.append(AndreyContinuous(method, "standardize"))
        if robust and scale == DECISION_SCALE and d <= ROBUST_MAX_D:
            out.append(AndreyContinuous(method, "robust"))
    out += [
        ZhengNOTEARS(),
        BelloDAGMA(),
        _control(CausalDiscoR2SortnRegress()),
        _control(CausalDiscoVarSortnRegress()),
        BaselineEmpty(),
    ]
    return tuple(out)


def cell_key(graph: str, d: int, noise: str, scale: str, seed: int) -> DatasetKey:
    """The dataset of one seed of a cell."""
    topology, density = GRAPHS[graph]
    return DatasetKey(
        topology=topology,
        functional="linear",
        noise=noise,
        standardize=SCALES[scale],
        density=density,
        d=d,
        n=ROWS,
        seed=seed,
    )


def plan(
    sizes: Any = SIZES,
    graphs: Any = tuple(GRAPHS),
    noises: Any = NOISES,
    scales: Any = DEFAULT_SCALES,
    seeds: int = 20,
    first_seed: int = 0,
    *,
    robust: bool = False,
    solutions: Any = None,
) -> list[Unit]:
    """Every unit of the grid, sizes in order; `solutions` keeps the arms with those names."""
    units = []
    for d, graph, noise, scale, seed in itertools.product(
        sizes, graphs, noises, scales, range(first_seed, first_seed + seeds)
    ):
        kept = tuple(
            a for a in arms(scale, d, robust=robust) if solutions is None or a.name in solutions
        )
        if kept:
            units.append(Unit(cell_key(graph, d, noise, scale, seed), kept))
    return units


def solution_names() -> list[str]:
    """The name of every arm the lane can run."""
    names = {a.name for scale in SCALES for a in arms(scale, ROBUST_MAX_D, robust=True)}
    return sorted(names)


def _log(msg: str = "") -> None:
    """Print and flush: a batch job's output is read while it runs."""
    print(msg, flush=True)


def _reference_venv() -> Path:
    """The environment root of the references' interpreter, or ``SystemExit`` when unset."""
    raw = os.environ.get(REFERENCE_ENV, "").strip()
    if not raw:
        raise SystemExit(f"${REFERENCE_ENV} is unset: export what `start_notears.sh` prints")
    return Path(raw).absolute().parent.parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="NOTEARS and DAGMA in their papers' setting")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument("--sizes", type=int, nargs="+", default=list(SIZES), help="variables")
    p.add_argument("--graphs", nargs="+", choices=sorted(GRAPHS), default=sorted(GRAPHS))
    p.add_argument("--noises", nargs="+", choices=NOISES, default=list(NOISES))
    p.add_argument(
        "--scales",
        nargs="+",
        choices=sorted(SCALES),
        default=list(DEFAULT_SCALES),
        help="column scales (default: raw, standardized, rescaled)",
    )
    p.add_argument(
        "--solutions",
        nargs="+",
        choices=solution_names(),
        default=None,
        help="arms to run (default: every arm of each cell)",
    )
    p.add_argument(
        "--robust",
        action="store_true",
        help=f"add the robust=True arms on standardized cells of at most {ROBUST_MAX_D} variables",
    )
    p.add_argument("--seeds", type=int, default=20, help="datasets per cell (default: 20)")
    p.add_argument(
        "--first-seed", type=int, default=0, help="first seed; fresh seeds for a rerun (default: 0)"
    )
    p.add_argument("--part", type=int, default=0, help="this run's part, 0-based (default: 0)")
    p.add_argument("--parts", type=int, default=1, help="parts the plan splits into (default: 1)")
    p.add_argument("--cap-wall-s", type=float, default=1800.0, help="wall cap per fit, seconds")
    p.add_argument("--cap-mem-mb", type=float, default=None, help="memory cap per fit, MB")
    p.add_argument("--cores", type=int, default=available_cores(), help="core budget per fit")
    p.add_argument(
        "--budget-s", type=float, default=None, help="stop between units after this many seconds"
    )
    p.add_argument(
        "--machine-id",
        default=socket.gethostname(),
        help="machine id stamped on every record; fix it to resume on another node",
    )
    p.add_argument("--resume", action="store_true", help="continue the run already in --out")
    args = p.parse_args(argv)
    if not 0 <= args.part < args.parts:
        raise SystemExit("--part must lie in [0, --parts)")

    units = plan(
        args.sizes,
        args.graphs,
        args.noises,
        args.scales,
        args.seeds,
        args.first_seed,
        robust=args.robust,
        solutions=args.solutions,
    )[args.part :: args.parts]
    names = sorted({a.name for u in units for a in u.arms})
    # Check the environment before the first fit: one error here, not one per unit.
    venvs = {}
    if any(getattr(a, "python_env", None) == REFERENCE_ENV for u in units for a in u.arms):
        venvs[REFERENCE_ENV.lower()] = _reference_venv()
    if ZhengNOTEARS.name in names:
        code_file()

    out = Path(args.out)
    run_meta = {
        "driver": Path(__file__).name,
        "sizes": args.sizes,
        "graphs": args.graphs,
        "noises": args.noises,
        "scales": args.scales,
        "robust": args.robust,
        "rows": ROWS,
        "seeds": args.seeds,
        "first_seed": args.first_seed,
        "part": args.part,
        "parts": args.parts,
        "units": len(units),
        "solution list": names,
        "repeats": 1,
        "warmup": 0,
        "cap_wall_s": args.cap_wall_s,
        "cap_mem_mb": args.cap_mem_mb,
        "cores": args.cores,
        "machine_id": args.machine_id,
    }
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat(), extra_venvs=venvs)
    _log(f"=== NOTEARS and DAGMA, part {args.part}/{args.parts}: {len(units)} units -> {out}")

    start = time.perf_counter()
    errors: list[str] = []
    visited = 0
    for unit in units:
        if args.budget_s is not None and time.perf_counter() - start > args.budget_s:
            _log(f"budget spent after {visited} units")
            break
        result = run_batch(
            datasets.store_dir(out),
            list(unit.arms),
            [unit.key],
            machine_id=args.machine_id,
            cap_wall_s=args.cap_wall_s,
            cap_mem_mb=args.cap_mem_mb,
            cores=args.cores,
            records_dir=records_dir,
            run_dir=out,
        )
        errors.extend(result.errors)
        visited += 1
        if result.records:
            cost = "  ".join(
                f"{r.name}={r.wall_s:.3g}s"
                if r.status == "ok" and r.wall_s
                else f"{r.name}={r.status}"
                for r in result.records
            )
            _log(f"[{visited}/{len(units)}] {unit.key.label} | {cost}")

    table = rundir.assemble(records_dir)
    table.to_parquet(out / "runs.parquet", index=False)
    rundir.finish_run(out)
    expected = sum(len(u.arms) for u in units)
    summary = {
        "part": args.part,
        "parts": args.parts,
        "units": len(units),
        "visited": visited,
        "records": len(table),
        "records_expected": expected,
        "status": dict(Counter(table["status"])) if len(table) else {},
        "errors": errors[:20],
        "elapsed_s": round(time.perf_counter() - start, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    done = visited == len(units) and len(table) >= expected
    if done:
        (out / "done.json").write_text(json.dumps({"records": len(table)}))
    _log(json.dumps(summary))
    return 0 if done else 3


if __name__ == "__main__":
    raise SystemExit(main())
