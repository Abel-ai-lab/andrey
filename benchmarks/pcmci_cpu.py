#!/usr/bin/env python
"""Benchmark Andrey's PCMCI, PCMCI+, or LPCMCI against tigramite's over a ladder of sizes, on CPU.

``--method`` picks the method, its regime, and its solutions (:data:`METHODS`). The rungs are the
product of the variable counts (``--ladder``) and the series lengths (``--lengths``): each ``(d, T)``
pair is one rung, and each seed one series of ``T`` time steps.

Each method runs on the simulation of its own paper (:mod:`andrey_bench.temporal_models`), with
the same expected number of cross-links per variable at every ``d``. ``pcmci`` runs on
``pcmci_paper``, whose links all have lag 1 or 2. Every solution tests lags ``1..tau_max`` (default
2) and is scored lag by lag. Andrey runs both of its partial-correlation tests: ``student_t`` uses
the p-value of tigramite's ``ParCorr(significance="analytic")``, and ``fisherz`` Fisher's
z-transform. The empty lag stack is the floor. Both packages run ``pc_alpha=0.2`` and
``alpha_level=0.05``.

``pcmciplus`` runs on ``pcmciplus_paper`` (30% of the cross-links within a time step, the rest at
lags 1 to 5; default ``tau_max`` 5), and ``lpcmci`` on ``lpcmci_paper`` (30% within a time step,
the rest at lags 1 to 3, 30% of the series unobserved; default ``tau_max`` 3). Every solution tests
lags ``0..tau_max`` with the Student-t test and returns window marks, scored link by link
(:func:`andrey_bench.scoring.window_scores`); the empty window is the floor. LPCMCI's estimates are
not scored against the truth, which has unobserved series. Both packages run the method's defaults:
PCMCI+ ``pc_alpha=0.01`` with the majority collider rule, LPCMCI ``pc_alpha=0.05`` with one
preliminary round. ``--ladder`` counts every generated series, unobserved ones included.

``--tau-max`` sets the lag window for every solution and is in every solution name, because
``params`` are not part of a ``run_id``.

**tigramite fits run in their own interpreter.** tigramite is GPL-licensed, so it is installed only
in the environment ``benchmarks/start_tigramite.sh`` builds, the adapter names
``ANDREY_BENCH_TIGRAMITE_PYTHON``, and the runner starts its fits there. That environment is
recorded in ``provenance.json``. A run whose solution list needs it fails at launch when it is
unset.

    ANDREY_BENCH_TIGRAMITE_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/pcmci_cpu.py \\
        --method pcmciplus --out outputs/pcmciplus --ladder 5 10 20 --lengths 500 1000
"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from andrey_bench import datasets, memcap, rundir
from andrey_bench.adapters.andrey_pcmci import (
    DEFAULT_TAU_MAX,
    andrey_lpcmci_adapter,
    andrey_pcmci_adapters,
    andrey_pcmciplus_adapter,
)
from andrey_bench.adapters.baseline_empty import baseline_empty_temporal, baseline_empty_window
from andrey_bench.adapters.tigramite_pcmci import (
    PYTHON_ENV_VAR,
    tigramite_adapter,
    tigramite_lpcmci_adapter,
    tigramite_pcmciplus_adapter,
)
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import BENCH_ROOT, available_cores, check_interpreters

#: Where ``benchmarks/start_tigramite.sh`` builds the tigramite environment. Only a default for the
#: error message: what runs is whatever ``$ANDREY_BENCH_TIGRAMITE_PYTHON`` names.
DEFAULT_TIGRAMITE_VENV = BENCH_ROOT / ".venv-tigramite"


def slice_of(regime: str) -> dict:
    """The dataset coordinates of a temporal regime, spelled out."""
    topology, functional, noise, standardize = datasets.REGIMES[regime]
    return {
        "topology": topology,
        "functional": functional,
        "noise": noise,
        "standardize": standardize,
        "density": datasets.TEMPORAL_REGIMES[regime],
    }


#: PCMCI's regime: the PCMCI paper's simulation, lag-only links at lags 1 and 2.
REGIME = "pcmci_paper"
SLICE = slice_of(REGIME)

#: Untimed fits before each timed one.
WARMUP = 1

#: The smallest run that covers every solution: one series of 200 steps over 5 variables.
SMOKE = {"ladder": [5], "lengths": [200], "seeds": 1, "repeats": 1, "cap_wall_s": 300.0}


def tigramite_venv() -> Path:
    """The venv tigramite fits run in, read from the variable the runner reads.

    Derived from the interpreter, so ``provenance.json`` records the environment the fits entered.
    """
    raw = os.environ.get(PYTHON_ENV_VAR, "").strip()
    if not raw:
        raise SystemExit(
            f"${PYTHON_ENV_VAR} is unset. Build the tigramite environment with "
            f"benchmarks/start_tigramite.sh and export it (default: "
            f"{DEFAULT_TIGRAMITE_VENV}/bin/python)."
        )
    # <venv>/bin/python -> <venv>, without resolving the symlink to the base interpreter.
    python = Path(raw).absolute()
    venv = python.parent.parent
    if venv / "bin" / "python" != python:
        raise SystemExit(
            f"${PYTHON_ENV_VAR}={raw!r} is not a <venv>/bin/python, so the environment recorded in "
            f"provenance.json would not be the one the fits ran in. Point it at a venv interpreter."
        )
    return venv


def solutions(tau_max: int) -> list:
    """Both Andrey tests, tigramite, and the empty lag stack, all over lags ``1..tau_max``.

    Built rather than declared because tigramite reads its version out of the interpreter that will
    run it.
    """
    return [
        *andrey_pcmci_adapters(tau_max),
        tigramite_adapter(tau_max),
        baseline_empty_temporal(tau_max),
    ]


def min_length(d: int, tau_max: int) -> int:
    """The shortest series ``andrey.pcmci`` accepts: every test keeps a degree of freedom."""
    return 2 * tau_max * (d + 1) + 3


def pcmciplus_solutions(tau_max: int) -> list:
    """Andrey's PCMCI+, tigramite's, and the empty window, all over lags ``0..tau_max``."""
    return [
        andrey_pcmciplus_adapter(tau_max),
        tigramite_pcmciplus_adapter(tau_max),
        baseline_empty_window(tau_max),
    ]


def pcmciplus_min_length(d: int, tau_max: int) -> int:
    """The shortest series ``andrey.pcmciplus`` accepts."""
    return 2 * tau_max * (d + 1) + d + 1


def lpcmci_solutions(tau_max: int) -> list:
    """Andrey's LPCMCI, tigramite's, and the empty window, all over lags ``0..tau_max``."""
    return [
        andrey_lpcmci_adapter(tau_max),
        tigramite_lpcmci_adapter(tau_max),
        baseline_empty_window(tau_max),
    ]


def lpcmci_min_length(d: int, tau_max: int) -> int:
    """The shortest series ``andrey.lpcmci`` accepts."""
    return d * (tau_max + 1) + 2 * tau_max + 1


@dataclass(frozen=True)
class Method:
    """What ``--method`` selects: the regime, the default lag window, the field, the shortest series."""

    regime: str
    tau_max: int
    solutions: Callable[[int], list]
    min_length: Callable[[int, int], int]
    title: str


METHODS = {
    "pcmci": Method(REGIME, DEFAULT_TAU_MAX, solutions, min_length, "PCMCI"),
    # tau_max covers each paper's largest true lag.
    "pcmciplus": Method("pcmciplus_paper", 5, pcmciplus_solutions, pcmciplus_min_length, "PCMCI+"),
    "lpcmci": Method("lpcmci_paper", 3, lpcmci_solutions, lpcmci_min_length, "LPCMCI"),
}


def ladder_keys(
    ladder: list[int], lengths: list[int], seeds: int, regime: str = REGIME
) -> list[DatasetKey]:
    """One series per ``(d, T, seed)`` on the regime's slice, with ``n = T`` time steps."""
    coordinates = slice_of(regime)
    return [
        DatasetKey(**coordinates, d=d, n=length, seed=seed, latents=datasets.latents_for(regime, d))
        for d in ladder
        for length in lengths
        for seed in range(seeds)
    ]


def _log(msg: str = "") -> None:
    """Print and flush immediately - a batch job's stdout is only useful unbuffered."""
    print(msg, flush=True)


def _agreement(table) -> list[str]:
    """Per dataset, the solutions grouped by identical estimated graph (``structure_hash``)."""
    lines = []
    ok = table[(table.status == "ok") & table.structure_hash.notna()]
    for (d, n, seed), rows in ok.groupby(["d", "n", "data_seed"], sort=True):
        groups = rows.groupby("structure_hash")["name"].apply(lambda s: sorted(set(s)))
        same = " | ".join(" = ".join(names) for names in sorted(groups))
        lines.append(f"d={d} T={n} seed={seed}: {same}")
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Andrey against tigramite: PCMCI, PCMCI+, LPCMCI, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--method",
        choices=sorted(METHODS),
        default="pcmci",
        help="the method, its regime, and its solutions (default: pcmci)",
    )
    p.add_argument("--ladder", type=int, nargs="+", help="variable counts d, run in order")
    p.add_argument("--lengths", type=int, nargs="+", help="series lengths T, run in order per d")
    p.add_argument("--seeds", type=int, default=3, help="series per rung (default: 3)")
    p.add_argument("--repeats", type=int, default=5, help="timed repeats per fit (default: 5)")
    p.add_argument(
        "--warmup",
        type=int,
        default=WARMUP,
        help="untimed fits before each timed one; 0 on a reach run (default: 1)",
    )
    p.add_argument("--cap-wall-s", type=float, default=1800.0, help="wall cap per repeat, seconds")
    p.add_argument("--cap-mem-mb", type=float, default=None, help="memory cap per repeat, MB")
    p.add_argument("--mem-route", choices=memcap.ROUTES, default=None, help="cap enforcement route")
    p.add_argument(
        "--cores",
        type=int,
        default=available_cores(),
        help="core budget per fit, the BLAS threads of a serial solution (default: every core "
        "this job is allowed)",
    )
    p.add_argument(
        "--tau-max",
        type=int,
        default=None,
        help=f"largest lag every solution tests, in every solution name (default: "
        f"{DEFAULT_TAU_MAX} for pcmci, 5 for pcmciplus, 3 for lpcmci)",
    )
    p.add_argument(
        "--solutions",
        nargs="+",
        default=None,
        help="run only these solution names (default: every solution)",
    )
    p.add_argument(
        "--machine-id",
        default=socket.gethostname(),
        help="machine id stamped on every record and in run_id (default: this host's name)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="run d=5, T=200, one seed, one repeat, and a 300 s cap, replacing those flags",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue the run already in --out: skip units whose records are on disk. Needs "
        "the same run_meta.json, so pass the flags the first run was given",
    )
    args = p.parse_args(argv)
    method = METHODS[args.method]
    if args.tau_max is None:
        args.tau_max = method.tau_max
    if args.smoke:
        vars(args).update(SMOKE)
    if not args.ladder or not args.lengths:
        p.error("--ladder and --lengths are required unless --smoke is given")
    if args.seeds < 1 or args.repeats < 1:
        p.error("--seeds and --repeats must be at least 1: a run with no fits records nothing")
    if args.warmup < 0:
        p.error(f"--warmup must not be negative, got {args.warmup}")
    if args.tau_max < 1:
        p.error(f"--tau-max must be at least 1, got {args.tau_max}")
    # Andrey rejects a shorter series, so every one of its fits at that rung would be an error.
    short = [
        (d, length, method.min_length(d, args.tau_max))
        for d in args.ladder
        for length in args.lengths
        if length < method.min_length(d, args.tau_max)
    ]
    if short:
        p.error(
            f"series too short for {args.method} at tau_max={args.tau_max}: (d, T, shortest) "
            f"{short}"
        )

    out = Path(args.out)
    field = method.solutions(args.tau_max)
    if args.solutions:
        known = {a.name for a in field}
        unknown = sorted(set(args.solutions) - known)
        # A typo would otherwise run fewer solutions and say nothing.
        if unknown:
            p.error(f"unknown solutions {unknown}; the solutions are {sorted(known)}")
        field = [a for a in field if a.name in args.solutions]

    # Every interpreter, before any rung runs.
    check_interpreters(field)
    venv = tigramite_venv() if any(getattr(a, "python_env", None) for a in field) else None

    # Written before the first fit, so an interrupted benchmark still says what it was asked for.
    run_meta = {
        "driver": Path(__file__).name,
        "method": args.method,
        "solution list": [a.name for a in field],
        "tigramite_python": {PYTHON_ENV_VAR: os.environ.get(PYTHON_ENV_VAR, "")},
        "params": {a.name: dict(a.params()) for a in field},
        "regime": method.regime,
        "slice": slice_of(method.regime),
        "tau_max": args.tau_max,
        # `ladder` counts rungs for STATUS.json, so it holds one entry per (d, T) rung.
        "ladder": [[d, length] for d in args.ladder for length in args.lengths],
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
    prov = rundir.provenance_once(
        out,
        timestamp=datetime.now(timezone.utc).isoformat(),
        extra_venvs={"tigramite": venv} if venv else None,
    )

    _log(f"=== Andrey {method.title} vs tigramite {method.title}, CPU -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:16s}: {value}")
    _log(f"{'git_sha':16s}: {prov['git_sha']}")
    tigramite_env = prov.get("environments", {}).get("tigramite")
    if tigramite_env:
        packages = tigramite_env["versions"].get("packages", {})
        _log(f"{'tigramite':16s}: {packages.get('tigramite')} numpy {packages.get('numpy')}")
    _log("")

    records: list = []
    errors: list[str] = []
    skipped = 0
    start = time.perf_counter()

    for d in args.ladder:
        for length in args.lengths:
            # One rung at a time, so a rung's data is materialized only when the ladder reaches it.
            result = run_batch(
                datasets.store_dir(out),
                field,
                ladder_keys([d], [length], args.seeds, method.regime),
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
            for adapter in field:
                mine = [r for r in result.records if r.name == adapter.name]
                walls = [r.wall_s for r in mine if r.status == "ok" and r.wall_s is not None]
                statuses = sorted({r.status for r in mine})
                cost = (
                    f"{statistics.median(walls):.3g}s"
                    if walls
                    else "/".join(statuses) or "no-records"
                )
                ran.append(f"{adapter.name}={cost}")
            _log(
                f"[d={d} T={length}] {'  '.join(ran)} | elapsed {time.perf_counter() - start:.0f}s"
            )

    # From the parts, not from memory: a resume that measured nothing still writes the table for
    # everything measured before it.
    table = rundir.assemble(records_dir)
    table.to_parquet(out / "runs.parquet", index=False)
    rundir.finish_run(out)

    _log("")
    _log("identical graphs:")
    for line in _agreement(table):
        _log(f"  {line}")
    _log("")
    _log(f"elapsed     : {time.perf_counter() - start:.1f}s")
    _log(f"measured    : {len(records)} records, {skipped} units skipped")
    _log(f"table       : {len(table)} rows  -> {out / 'runs.parquet'}")
    _log(f"errors      : {len(errors)}")
    for err in errors:
        _log(f"  ! {err}")

    # A fit that launched and failed is an `error` record, not an entry in `errors`; counted from
    # the table so an earlier sitting's failure counts after a resume. `timeout` and `oom` are
    # measurements of a cap.
    failed = table[table.status == "error"]
    if len(failed):
        _log(f"failed fits : {len(failed)}")
        for row in failed.head(5).itertuples():
            _log(f"  ! {row.name} @ d={row.d} T={row.n} seed={row.data_seed}: {row.error_text}")
    return 1 if errors or len(failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
