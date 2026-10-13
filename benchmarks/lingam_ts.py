#!/usr/bin/env python
"""Benchmark Andrey's VARMA-LiNGAM or longitudinal LiNGAM against lingam's, on CPU.

``--method varma`` runs on the ``varma_lingam_paper`` regime (Hyvarinen et al. 2010, Sect. 5.4,
with one moving-average matrix): each rung is ``N`` variables (``--ladder``) and a series of ``T``
time steps (``--lengths``), one series per seed. ``--method longitudinal`` runs on the
``longitudinal_lingam_paper`` regime (Kadowaki et al. 2013): each rung is ``N`` variables, ``T``
time points (``--lengths``), and ``M`` units observed at each (``--samples``), one panel per seed.
:mod:`andrey_bench.lingam_models` gives both setups.

Both packages run the same settings (:mod:`~andrey_bench.adapters.andrey_lingam_ts`,
:mod:`~andrey_bench.adapters.lingam_ts`), and every estimate is scored coefficient by coefficient
(:func:`andrey_bench.scoring.coefficient_scores`).

**lingam fits run in their own interpreter.** lingam 1.13.0 installs on Python 3.12, not on the
bench environment's Python, so ``benchmarks/start_lingam_ts.sh`` builds ``.venv-lingam-ts``, the
adapter names ``ANDREY_BENCH_LINGAM_TS_PYTHON``, and the runner starts its fits there.

    ANDREY_BENCH_LINGAM_TS_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/lingam_ts.py \\
        --method longitudinal --out outputs/longitudinal --ladder 3 5 --lengths 20 --samples 500
"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from andrey_bench import datasets, memcap, rundir
from andrey_bench.adapters.andrey_lingam_ts import AndreyLongitudinalLiNGAM, AndreyVarmaLiNGAM
from andrey_bench.adapters.lingam_ts import (
    PYTHON_ENV_VAR,
    lingam_longitudinal_adapter,
    lingam_varma_adapter,
)
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores, check_interpreters

#: Each method's regime.
REGIMES = {"varma": "varma_lingam_paper", "longitudinal": "longitudinal_lingam_paper"}

#: The smallest run that covers both solutions of a method.
SMOKE = {
    "varma": {"ladder": [3], "lengths": [300], "samples": None},
    "longitudinal": {"ladder": [3], "lengths": [5], "samples": [200]},
}


def solutions(method: str) -> list:
    """Andrey's solution and lingam's, built so lingam's reads its version from its interpreter."""
    if method == "varma":
        return [AndreyVarmaLiNGAM(), lingam_varma_adapter()]
    return [AndreyLongitudinalLiNGAM(), lingam_longitudinal_adapter()]


def ladder_keys(
    method: str, ladder: list[int], lengths: list[int], samples: list[int] | None, seeds: int
) -> list[DatasetKey]:
    """One dataset per ``(N, T, M, seed)``; a series stores ``n = T``, a panel ``n = M``."""
    topology, functional, noise, standardize = datasets.REGIMES[REGIMES[method]]
    common = {
        "topology": topology,
        "functional": functional,
        "noise": noise,
        "standardize": standardize,
        "density": datasets.TEMPORAL_REGIMES[REGIMES[method]],
    }
    if method == "varma":
        return [
            DatasetKey(**common, d=d, n=length, seed=seed)
            for d in ladder
            for length in lengths
            for seed in range(seeds)
        ]
    return [
        DatasetKey(**common, d=d, n=m, seed=seed, times=length)
        for d in ladder
        for length in lengths
        for m in samples or []
        for seed in range(seeds)
    ]


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Andrey against lingam: VARMA-LiNGAM, LongitudinalLiNGAM"
    )
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument("--method", choices=sorted(REGIMES), required=True)
    p.add_argument("--ladder", type=int, nargs="+", help="variable counts N, run in order")
    p.add_argument("--lengths", type=int, nargs="+", help="series lengths or time points T")
    p.add_argument("--samples", type=int, nargs="+", help="units M per time point (longitudinal)")
    p.add_argument("--seeds", type=int, default=5, help="datasets per rung (default: 5)")
    p.add_argument("--repeats", type=int, default=1, help="timed repeats per fit (default: 1)")
    p.add_argument("--warmup", type=int, default=0, help="untimed fits before each timed one")
    p.add_argument("--cap-wall-s", type=float, default=1800.0, help="wall cap per repeat, seconds")
    p.add_argument("--cap-mem-mb", type=float, default=None, help="memory cap per repeat, MB")
    p.add_argument("--mem-route", choices=memcap.ROUTES, default=None, help="cap enforcement route")
    p.add_argument("--cores", type=int, default=available_cores(), help="core budget per fit")
    p.add_argument("--machine-id", default=socket.gethostname(), help="machine id on every record")
    p.add_argument("--smoke", action="store_true", help="run the smallest rung, one seed")
    p.add_argument("--resume", action="store_true", help="continue the run already in --out")
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE[args.method], seeds=1, repeats=1, cap_wall_s=600.0)
    if not args.ladder or not args.lengths:
        p.error("--ladder and --lengths are required unless --smoke is given")
    if args.method == "longitudinal" and not args.samples:
        p.error("--samples is required for --method longitudinal")
    if args.method == "varma" and args.samples:
        p.error("--samples applies to --method longitudinal only")

    out = Path(args.out)
    field = solutions(args.method)
    check_interpreters(field)
    python = os.environ.get(PYTHON_ENV_VAR, "").strip()
    venv = Path(python).absolute().parent.parent

    rungs = [
        [d, length, m]
        for d in args.ladder
        for length in args.lengths
        for m in (args.samples if args.method == "longitudinal" else [1])
    ]
    run_meta = {
        "driver": Path(__file__).name,
        "method": args.method,
        "solution list": [a.name for a in field],
        "lingam_python": {PYTHON_ENV_VAR: python},
        "params": {a.name: dict(a.params()) for a in field},
        "regime": REGIMES[args.method],
        "ladder": rungs,
        "seeds": args.seeds,
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
        extra_venvs={"lingam": venv},
    )
    _log(f"=== Andrey vs lingam, {args.method}, CPU -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:16s}: {value}")
    _log(f"{'git_sha':16s}: {prov['git_sha']}")
    _log("")

    errors: list[str] = []
    start = time.perf_counter()
    for d, length, m in rungs:
        keys = ladder_keys(
            args.method, [d], [length], [m] if args.method == "longitudinal" else None, args.seeds
        )
        result = run_batch(
            datasets.store_dir(out),
            field,
            keys,
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
        cost = []
        for adapter in field:
            mine = [r for r in result.records if r.name == adapter.name]
            walls = [r.wall_s for r in mine if r.status == "ok" and r.wall_s is not None]
            statuses = "/".join(sorted({r.status for r in mine})) or "no-records"
            cost.append(f"{adapter.name}={statistics.median(walls):.3g}s" if walls else statuses)
        size = f"N={d} T={length}" + (f" M={m}" if args.method == "longitudinal" else "")
        _log(f"[{size}] {'  '.join(cost)} | elapsed {time.perf_counter() - start:.0f}s")

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
    for row in failed.head(5).itertuples():
        _log(f"  ! {row.name} @ N={row.d} n={row.n} seed={row.data_seed}: {row.error_text}")
    return 1 if errors or len(failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
