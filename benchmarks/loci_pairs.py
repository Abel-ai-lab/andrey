#!/usr/bin/env python
"""Benchmark Andrey's LOCI against the authors' code on location-scale cause-effect pairs, CPU.

Every solution runs on the same pairs in the same job (:data:`SOLUTIONS`; ``--solutions`` picks
some of them): Andrey's ``loci`` under the independence criterion and under the likelihood
criterion, the authors' LOCI with spline features under each (``immer.loci.*``,
:mod:`~andrey_bench.adapters.immer_loci`), and Andrey's ``anm``, ``pnl``, and ``cam`` as context.
Each Andrey ``loci`` arm is compared with the authors' arm of the same criterion
(:data:`COMPARATOR`). The regimes (:data:`REGIMES`), each pair of two nodes and one edge in a
direction the seed draws:

    lsnm_pair, lsnm_gumbel_pair, lsnm_laplace_pair     location-scale, z-scored     500 1000 2000
    lsnm_rescaled_pair, ..._gumbel_..., ..._laplace_..  location-scale, rescaled     500 1000 2000
    anm_pair                                            additive noise, z-scored     500 1000 2000
    tagasovska_{an,ans,ls,lss,mnu}_pair                 Tagasovska et al. (2020)     1000

Every regime lies in LOCI's model, ``y = f(x) + g(x) n`` with the noise independent of a cause of
unbounded support (Immer et al. 2023): additive noise is the case of a constant ``g``, and
Tagasovska et al.'s multiplicative MN-U, ``y = f(x) u``, the case of the location ``f / 2`` and the
scale ``|f|``, the noise ``u - 1/2`` being symmetric. The likelihood criterion assumes Gaussian
noise, so it is misspecified on the Gumbel, Laplace, and MN-U regimes
(:func:`likelihood_misspecified`).

Each fit is scored by whether it names the true direction (:func:`anm_pairs.direction_scores`), on
top of the standard scores ``run_batch`` records; a timeout is a timeout and has no scores.

**The authors' code runs in its own interpreter**, the one ``ANDREY_BENCH_LOCI_PYTHON`` names, and
reads the code from ``ANDREY_BENCH_LOCI``; ``benchmarks/start_loci.sh`` builds both. Andrey's
``cam`` runs in the interpreter ``ANDREY_BENCH_CAM_PYTHON`` names, bench-env's by default. Use
``--seed-start`` and ``--regimes`` to cut a campaign into small runs, each with its own ``--out``;
share the dataset store, ``ANDREY_BENCH_DATA``, and fill it once with ``--prepare-only``.

    ANDREY_BENCH_LOCI_PYTHON=... ANDREY_BENCH_LOCI=... ANDREY_BENCH_CAM_PYTHON=$ANDREY_BENCH_PYTHON \\
        $ANDREY_BENCH_PYTHON benchmarks/loci_pairs.py --out outputs/loci --regimes lsnm_pair --cores 1
"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from anm_pairs import PAIR_DENSITY, add_direction_scores

from andrey_bench import datasets, memcap, rundir
from andrey_bench.adapters.andrey_anm import AndreyANM
from andrey_bench.adapters.andrey_cam import AndreyCAM
from andrey_bench.adapters.andrey_loci import AndreyLOCI, AndreyLOCILikelihood
from andrey_bench.adapters.andrey_pnl import AndreyPNL
from andrey_bench.adapters.immer_loci import ImmerLOCI, ImmerLOCILikelihood
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores, check_interpreters

#: The regimes, in ``datasets.REGIMES``: Andrey's generator first, then the paper's pair sets.
REGIMES = [
    "lsnm_pair",
    "lsnm_gumbel_pair",
    "lsnm_laplace_pair",
    "lsnm_rescaled_pair",
    "lsnm_gumbel_rescaled_pair",
    "lsnm_laplace_rescaled_pair",
    "anm_pair",
    "tagasovska_an_pair",
    "tagasovska_ans_pair",
    "tagasovska_ls_pair",
    "tagasovska_lss_pair",
    "tagasovska_mnu_pair",
]

#: Each regime's rows per pair: the paper's 1,000 for its sets, a ladder around it for the rest.
SIZES: dict[str, list[int]] = {
    regime: [1000] if regime.startswith("tagasovska_") else [500, 1000, 2000] for regime in REGIMES
}

#: Smallest run that touches every solution: one small pair.
SMOKE = {"regimes": ["lsnm_pair"], "sizes": [200], "seeds": 1, "cap_wall_s": 600.0}

#: Every solution, by name.
SOLUTIONS = {
    a.name: a
    for a in (
        AndreyLOCI(),
        AndreyLOCILikelihood(),
        ImmerLOCI(),
        ImmerLOCILikelihood(),
        AndreyANM(),
        AndreyPNL(),
        AndreyCAM(),
    )
}

#: Each Andrey ``loci`` arm and the authors' arm of the same criterion.
COMPARATOR = {
    AndreyLOCI.name: ImmerLOCI.name,
    AndreyLOCILikelihood.name: ImmerLOCILikelihood.name,
}


def likelihood_misspecified(regime: str) -> bool:
    """Whether `regime`'s noise is not Gaussian, the likelihood criterion's assumption."""
    return datasets.REGIMES[regime][2] != "gaussian"


def regime_keys(regime: str, n: int, seeds: int, seed_start: int = 0) -> list[DatasetKey]:
    """Return one pair of `regime` at `n` rows per seed, `seeds` of them from `seed_start` on."""
    topology, functional, noise, standardize = datasets.REGIMES[regime]
    return [
        DatasetKey(
            topology=topology,
            functional=functional,
            noise=noise,
            standardize=standardize,
            density=PAIR_DENSITY,
            d=2,
            n=n,
            seed=seed,
        )
        for seed in range(seed_start, seed_start + seeds)
    ]


def _log(msg: str = "") -> None:
    """Print and flush batch output immediately."""
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LOCI direction test, Andrey against the authors, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--regimes", nargs="+", choices=REGIMES, default=REGIMES, help="regimes to run, in order"
    )
    p.add_argument(
        "--solutions",
        nargs="+",
        choices=list(SOLUTIONS),
        default=list(SOLUTIONS),
        help="solutions to run (default: all)",
    )
    p.add_argument(
        "--sizes", type=int, nargs="+", default=None, help="rows per pair (default: each regime's)"
    )
    p.add_argument("--seeds", type=int, default=100, help="pairs per cell (default: 100)")
    p.add_argument("--seed-start", type=int, default=0, help="first data seed (default: 0)")
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="materialize this run's datasets and stop, without measuring anything",
    )
    p.add_argument("--repeats", type=int, default=1, help="timed repeats per fit (default: 1)")
    p.add_argument("--warmup", type=int, default=0, help="untimed fits before each timed one")
    p.add_argument("--cap-wall-s", type=float, default=1800.0, help="wall cap per repeat, seconds")
    p.add_argument("--cap-mem-mb", type=float, default=None, help="memory cap per repeat, MB")
    p.add_argument("--mem-route", choices=memcap.ROUTES, default=None, help="cap enforcement route")
    p.add_argument("--cores", type=int, default=available_cores(), help="core budget per fit")
    p.add_argument("--machine-id", default=socket.gethostname(), help="machine id on every record")
    p.add_argument("--smoke", action="store_true", help="run one pair of 200 rows")
    p.add_argument("--resume", action="store_true", help="continue the run already in --out")
    p.add_argument(
        "--rescore",
        action="store_true",
        help="fit nothing: recompute the direction columns of the run in --out (the first run's "
        "flags)",
    )
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)

    out = Path(args.out)
    cells = [(r, n) for r in args.regimes for n in (args.sizes or SIZES[r])]
    cell_keys = {cell: regime_keys(*cell, args.seeds, args.seed_start) for cell in cells}
    keys = {key.digest: key for ks in cell_keys.values() for key in ks}

    if args.prepare_only:
        store = datasets.store_dir(out)
        registry = datasets.materialize(list(keys.values()), store)
        _log(
            f"materialized {len(keys)} dataset(s) into {store}; registry has "
            f"{len(registry['keys'])}"
        )
        return 0

    field = [SOLUTIONS[name] for name in args.solutions]
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in field],
        "params": {a.name: dict(a.params()) for a in field},
        "comparator": {a: b for a, b in COMPARATOR.items() if a in args.solutions},
        "regimes": args.regimes,
        "likelihood_misspecified": {r: likelihood_misspecified(r) for r in args.regimes},
        "cells": [
            {"regime": r, "n": n, "seeds": [k.seed for k in cell_keys[(r, n)]]} for r, n in cells
        ],
        "density": PAIR_DENSITY,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cap_wall_s": args.cap_wall_s,
        "cap_mem_mb": args.cap_mem_mb,
        "mem_route": memcap.resolve_route(args.cap_mem_mb, args.mem_route),
        "cores": args.cores,
        "machine_id": args.machine_id,
    }
    if args.rescore:
        records_dir = out / rundir.RECORDS_DIRNAME
        if not any(records_dir.glob("*.parquet")):
            raise SystemExit(f"--rescore: {records_dir} holds no records")
        add_direction_scores(out, records_dir, keys, rescore=True)
        table = rundir.assemble(records_dir)
        table.to_parquet(out / "runs.parquet", index=False)
        _log(f"rescored    : {len(table)} rows  -> {out / 'runs.parquet'}")
        _summarize(table)
        return 0

    interpreters = check_interpreters(field)
    loci_python = next((interpreters[a.name] for a in field if isinstance(a, ImmerLOCI)), None)
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    prov = rundir.provenance_once(
        out,
        timestamp=datetime.now(timezone.utc).isoformat(),
        extra_venvs={"loci": loci_python.parent.parent} if loci_python else None,
    )
    _log(f"=== LOCI direction test, Andrey and the authors -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:14s}: {value}")
    _log(f"{'loci python':14s}: {os.environ.get(ImmerLOCI.python_env, '')}")
    _log(f"{'git_sha':14s}: {prov['git_sha']}")
    _log("")

    errors: list[str] = []
    start = time.perf_counter()
    for cell in cells:
        regime, n = cell
        result = run_batch(
            datasets.store_dir(out),
            field,
            cell_keys[cell],
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
        add_direction_scores(out, records_dir, keys)
        cost = []
        for adapter in field:
            mine = [r for r in result.records if r.name == adapter.name]
            cpus = [r.cpu_s for r in mine if r.status == "ok" and r.cpu_s is not None]
            statuses = "/".join(sorted({r.status for r in mine})) or "no-records"
            cost.append(f"{adapter.name}={statistics.median(cpus):.3g}s" if cpus else statuses)
        _log(f"[{regime} n={n}] {'  '.join(cost)} | elapsed {time.perf_counter() - start:.0f}s")

    # From the parts, not from memory: a resume that measured nothing still writes the table for
    # everything measured before it.
    add_direction_scores(out, records_dir, keys)
    table = rundir.assemble(records_dir)
    table.to_parquet(out / "runs.parquet", index=False)
    rundir.finish_run(out)
    failed = table[table.status == "error"]
    _log("")
    _log(f"elapsed     : {time.perf_counter() - start:.1f}s")
    _log(f"table       : {len(table)} rows  -> {out / 'runs.parquet'}")
    _log(f"errors      : {len(errors)}; failed fits: {len(failed)}")
    for err in errors:
        _log(f"  ! {err}")
    _summarize(table)
    return 1 if errors or len(failed) else 0


def _summarize(table: pd.DataFrame) -> None:
    """Print, per cell and solution, the finished fits, their median CPU time, and their accuracy.

    ``accuracy`` is the share of finished fits that name the true direction, ``reversed`` and
    ``none`` the shares that name the other direction or no edge. ``agree`` is, for an Andrey
    ``loci`` arm, the share of pairs both it and the authors' arm of its criterion finished on
    which the two return the same graph.
    """
    if table.empty:
        return
    regime_of = {spec: regime for regime, spec in datasets.REGIMES.items() if regime in REGIMES}
    cell_columns = ["topology", "functional", "noise", "standardize", "n"]
    _log("")
    _log(
        f"{'regime':27s} {'n':>5s} {'solution':31s} {'ok/all':>7s} {'cpu_s':>7s} "
        f"{'accuracy':>8s} {'reversed':>8s} {'none':>6s} {'agree':>6s}"
    )
    for (*spec, n), cell in table.groupby(cell_columns, sort=False):
        ok = cell[cell.status == "ok"]
        hashes = ok.pivot_table(
            index="dataset_id", columns="name", values="structure_hash", aggfunc="first"
        )
        for name, rows in cell.groupby("name", sort=False):
            mine = rows[rows.status == "ok"]
            agree = ""
            other = COMPARATOR.get(str(name))
            if other in hashes.columns and name in hashes.columns:
                both = hashes[[name, other]].dropna()
                agree = f"{(both[name] == both[other]).mean():6.3f}" if len(both) else ""
            _log(
                f"{regime_of.get(tuple(spec), '?'):27s} {n:5d} {name:31s} "
                f"{len(mine):3d}/{len(rows):<3d} {mine['cpu_s'].median():7.3g} "
                f"{mine['direction_correct'].mean():8.3f} {mine['direction_reversed'].mean():8.3f} "
                f"{mine['direction_none'].mean():6.3f} {agree:>6s}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
