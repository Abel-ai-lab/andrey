#!/usr/bin/env python
"""The missing-value lanes: MVPC on ``pc`` (M1) and structural EM on the covariance searches (M2).

Every cell masks a ``sample_scm`` sample with :mod:`andrey.data.missingness` and fits these arms
on the same file: the method with ``missing`` set, test-wise PC (``pc``'s default), listwise
deletion and mean imputation before the method (:mod:`andrey_bench.adapters.missing_arms`), and the
empty graph. The complete-data arm fits the method on the same key without ``missing``, the
complete values of every mask drawn over it. Lanes:

- ``m1-generic``: ``pc(missing="mvpc")`` on standardized Erdos-Renyi graphs of mean degree 2 and
  4, 10 to 100 variables, 500 to 10,000 rows, five masks at rates 0.1 to 0.3; every PC arm at
  ``alpha`` 0.05 (Andrey's default) and 0.01 (the MVPC authors' ``demo.R``).
- ``m1-paper``: the MVPC authors' generator (``mvpc2019`` keys), with their R MVPC (PermC) and
  test-wise PC beside Andrey's arms. Needs R (``benchmarks/start_mvpc_r.sh``).
- ``m2-a``: MissDAG's equal-variance setting, raw linear Gaussian data (weights from
  ``U(+-[0.5, 2])``, unit noise variances) on ER1, ER2, and SF2 graphs, 20 variables, 100 rows,
  MCAR 0.1 to 0.5: ``notears(missing="em")`` beside MissDAG's ``miss_dag_gaussian`` with its
  NOTEARS, and both NOTEARS on the complete data. Needs ``benchmarks/start_missdag.sh``.
- ``m2-b``: 15 variables, 100 rows, MCAR 0.1 to 0.5: ``ges``, ``boss``, and ``grasp`` with
  ``missing="em"``, each at ``ebic_gamma`` 1.0 (Andrey's default) and 0.0 (the standard BIC).
- ``m2-c``: the ``m2-a`` and ``m2-b`` cells under MissDAG's Appendix F.2 masks: 30% of the columns
  always observed, logistic MAR and logistic self-masking. Needs MissDAG.
- ``m2-generic``: ``ges``, ``boss``, and ``grasp`` with ``missing="em"`` on the ``m1-generic``
  graphs at 100 to 2,000 rows.
- ``m2-reduced``: ``hc``, ``tabu``, ``xges``, ``exact_search``, ``calm``, ``notears``, and
  ``dagma`` with ``missing="em"`` at 20 variables and 500 rows, MCAR and logistic MAR at 0.2.

``--part i --parts k`` runs every k-th unit of the lane from the i-th, so a lane splits into
independent jobs; ``--first-seed`` moves the seeds, for a rerun on fresh ones; ``--budget-s`` stops
between units once the budget is spent, and ``--resume`` continues from the records on disk.
``done.json`` appears when every unit of the part has its records, and ``summary.json`` after every
sitting. ``missing_summary.py`` pairs the records into the rows of the published summary.

    $ANDREY_BENCH_PYTHON benchmarks/missing_cpu.py --lane m2-b --out outputs/m2-b-0 --resume
"""

from __future__ import annotations

import argparse
import itertools
import json
import socket
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from andrey_bench import datasets, rundir
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.missing_arms import AndreyMissing, Listwise, MeanImpute
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores

#: The masks of the generic grids: ``(mechanism, form)``. MNAR takes the quantile form of Tu et al.
#: (2019) and self-masking the logistic form of MissDAG, the negative control.
GENERIC_MASKS = (
    ("mcar", ""),
    ("mar", "logistic"),
    ("mar", "quantile"),
    ("mnar", "quantile"),
    ("self_masking", "logistic"),
)
GENERIC_RATES = (0.1, 0.2, 0.3)
#: MissDAG's MCAR rates (Section 6) and the Appendix F.2 rates the lane runs.
PAPER_RATES = (0.1, 0.2, 0.3, 0.4, 0.5)
F2_RATES = (0.1, 0.3, 0.5)
#: MissDAG's graphs: Erdos-Renyi of mean degree 1 and 2, and scale-free of mean degree 2.
PAPER_GRAPHS = (("er", 1.0), ("er", 2.0), ("scale_free", 2.0))
#: The rows of the MVPC paper's sample-size experiments that the lane runs.
MVPC_ROWS = (100, 1000, 5000, 10000)

M2_SEARCHES = ("ges", "boss", "grasp")
M2_REDUCED = ("hc", "tabu", "xges", "exact_search", "calm", "notears", "dagma")


@dataclass(frozen=True)
class Unit:
    """One dataset and the arms fitted on it."""

    key: DatasetKey
    arms: tuple[Any, ...]


def _pc_arms(alpha: float) -> list[Any]:
    """Test-wise PC, MVPC, and test-wise PC after listwise deletion and mean imputation."""
    testwise = AndreyMissing("pc", None, alpha=alpha)
    return [
        testwise,
        AndreyMissing("pc", "mvpc", alpha=alpha, seed=0),
        Listwise(testwise),
        MeanImpute(testwise),
    ]


def _em_arms(method: str, **options: Any) -> tuple[list[Any], list[Any]]:
    """``(masked arms, complete arms)`` of one EM method: EM, listwise, mean; the method alone."""
    plain = AndreyMissing(method, None, **options)
    return [AndreyMissing(method, "em", **options), Listwise(plain), MeanImpute(plain)], [plain]


def _cells(
    graphs: Any, sizes: Any, masks: Any, rates: Any, seeds: int, standardize: str
) -> list[tuple[DatasetKey, DatasetKey]]:
    """``(masked key, complete key)`` pairs over the grid; linear Gaussian ``sample_scm`` data."""
    pairs = []
    for (topology, density), (d, n), (mechanism, form), rate, seed in itertools.product(
        graphs, sizes, masks, rates, range(seeds)
    ):
        complete = DatasetKey(
            topology=topology,
            functional="linear",
            noise="gaussian",
            standardize=standardize,
            density=density,
            d=d,
            n=n,
            seed=seed,
        )
        masked = replace(complete, missing=mechanism, missing_form=form, missing_rate=rate)
        pairs.append((masked, complete))
    return pairs


def _units(pairs: list[tuple[DatasetKey, DatasetKey]], masked: list, complete: list) -> list[Unit]:
    """One unit per masked key, and one per distinct complete key."""
    empty = BaselineEmpty()
    out = [Unit(m, (*masked, empty)) for m, _ in pairs]
    seen = dict.fromkeys(c for _, c in pairs)
    out += [Unit(c, (*complete, empty)) for c in seen]
    return out


def plan(lane: str, seeds: int, first: int = 0) -> list[Unit]:
    """Every unit of `lane` with `seeds` seeds per cell, from seed `first`."""
    return [u for u in _plan(lane, first + seeds) if u.key.seed >= first]


def _plan(lane: str, seeds: int) -> list[Unit]:
    """Every unit of `lane` over the seeds below `seeds`."""
    if lane == "m1-generic":
        pairs = _cells(
            [("er", 2.0), ("er", 4.0)],
            itertools.product((10, 20, 50, 100), (500, 2000, 10000)),
            GENERIC_MASKS,
            GENERIC_RATES,
            seeds,
            "standardized",
        )
        masked = _pc_arms(0.05) + _pc_arms(0.01)
        complete = [AndreyMissing("pc", None, alpha=a) for a in (0.05, 0.01)]
        return _units(pairs, masked, complete)
    if lane == "m1-paper":
        from andrey_bench.adapters.mvpc_r import PAPER_SIZES, MvpcR

        units = []
        for (d, n), mode, seed in itertools.product(
            itertools.product(PAPER_SIZES, MVPC_ROWS), ("mar", "mnar"), range(seeds)
        ):
            base = DatasetKey(
                topology="er",
                functional=datasets.MVPC_PAPER,
                noise="gaussian",
                standardize="raw",
                density=2.0,
                d=d,
                n=n,
                seed=seed,
                missing=mode,
                missing_form="quantile",
            )
            arms = (
                *_pc_arms(0.05),
                *_pc_arms(0.01),
                MvpcR("permc"),
                MvpcR("testwise"),
                BaselineEmpty(),
            )
            units.append(Unit(base, arms))
            # The authors' MCAR reference: the same rows of each column removed at random.
            units.append(Unit(replace(base, missing_form="reference"), arms))
            complete = [AndreyMissing("pc", None, alpha=a) for a in (0.05, 0.01)]
            units.append(Unit(replace(base, missing_form="complete"), (*complete, BaselineEmpty())))
        return units
    if lane in ("m2-a", "m2-b", "m2-c"):
        from andrey_bench.adapters.missdag import MissDagGaussian, MissDagNotears

        mcar = [("mcar", "")]
        f2 = [("mar", "logistic"), ("self_masking", "logistic")]
        units: list[Unit] = []
        if lane in ("m2-a", "m2-c"):
            masks, rates = (mcar, PAPER_RATES) if lane == "m2-a" else (f2, F2_RATES)
            pairs = _cells(PAPER_GRAPHS, [(20, 100)], masks, rates, seeds, "raw")
            em, plain = _em_arms("notears")
            masked = [*em, MissDagGaussian(), AndreyMissing("pc", None)]
            units += _units(pairs, masked, [*plain, MissDagNotears()])
        if lane in ("m2-b", "m2-c"):
            masks, rates = (mcar, PAPER_RATES) if lane == "m2-b" else (f2, F2_RATES)
            pairs = _cells(PAPER_GRAPHS, [(15, 100)], masks, rates, seeds, "raw")
            masked, complete = [AndreyMissing("pc", None)], []
            for method, gamma in itertools.product(M2_SEARCHES, (1.0, 0.0)):
                em, plain = _em_arms(method, ebic_gamma=gamma)
                masked += em
                complete += plain
            units += _units(pairs, masked, complete)
        return units
    if lane == "m2-generic":
        pairs = _cells(
            [("er", 2.0), ("er", 4.0)],
            itertools.product((10, 20, 50, 100), (100, 500, 2000)),
            GENERIC_MASKS,
            GENERIC_RATES,
            seeds,
            "standardized",
        )
        masked, complete = [AndreyMissing("pc", None)], []
        for method in M2_SEARCHES:
            em, plain = _em_arms(method)
            masked += em
            complete += plain
        return _units(pairs, masked, complete)
    if lane == "m2-reduced":
        pairs = _cells(
            [("er", 2.0)],
            [(20, 500)],
            [("mcar", ""), ("mar", "logistic")],
            [0.2],
            seeds,
            "standardized",
        )
        masked, complete = [AndreyMissing("pc", None)], []
        for method in M2_REDUCED:
            em, plain = _em_arms(method)
            masked += em
            complete += plain
        return _units(pairs, masked, complete)
    raise SystemExit(f"unknown lane {lane!r}; one of {', '.join(LANES)}")


LANES = ("m1-generic", "m1-paper", "m2-a", "m2-b", "m2-c", "m2-generic", "m2-reduced")
#: Seeds per cell: 50 in the MVPC paper's setting, 20 elsewhere.
DEFAULT_SEEDS = {"m1-paper": 50}


def _log(msg: str = "") -> None:
    """Print and flush: a batch job's output is read while it runs."""
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="missing-value lanes, CPU")
    p.add_argument("--lane", required=True, choices=LANES)
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--seeds", type=int, default=None, help="seeds per cell (default: 20; 50 m1-paper)"
    )
    p.add_argument(
        "--first-seed", type=int, default=0, help="first seed; fresh seeds for a rerun (default: 0)"
    )
    p.add_argument("--part", type=int, default=0, help="this job's part, 0-based (default: 0)")
    p.add_argument("--parts", type=int, default=1, help="parts the lane splits into (default: 1)")
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

    seeds = args.seeds or DEFAULT_SEEDS.get(args.lane, 20)
    units = plan(args.lane, seeds, args.first_seed)[args.part :: args.parts]
    out = Path(args.out)
    run_meta = {
        "driver": Path(__file__).name,
        "lane": args.lane,
        "seeds": seeds,
        "first_seed": args.first_seed,
        "part": args.part,
        "parts": args.parts,
        "units": len(units),
        "solution list": sorted({a.name for u in units for a in u.arms}),
        "repeats": 1,
        "warmup": 0,
        "cap_wall_s": args.cap_wall_s,
        "cap_mem_mb": args.cap_mem_mb,
        "cores": args.cores,
        "machine_id": args.machine_id,
    }
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())
    _log(f"=== missing-value lane {args.lane}, part {args.part}/{args.parts}: {len(units)} units")

    start = time.perf_counter()
    errors: list[str] = []
    visited = 0
    for unit in units:
        if args.budget_s is not None and time.perf_counter() - start > args.budget_s:
            _log(f"budget spent after {visited} units")
            break
        result = run_batch(
            datasets.store_dir(out),
            unit.arms,
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
            statuses = Counter(r.status for r in result.records)
            _log(f"[{visited}/{len(units)}] {unit.key.label} {dict(statuses)}")

    table = rundir.assemble(records_dir)
    table.to_parquet(out / "runs.parquet", index=False)
    rundir.finish_run(out)
    expected = sum(len(u.arms) for u in units)
    summary = {
        "lane": args.lane,
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
