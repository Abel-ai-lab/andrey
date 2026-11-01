#!/usr/bin/env python
"""Benchmark Andrey's DYNOTEARS against CausalNex's on the structural VARs of the DYNOTEARS paper.

Every series is drawn by :mod:`andrey_bench.svar_models`, one per seed: ``d`` variables
(``--ladder``) and ``n`` fitted steps (``--length``), one lag. The regimes (:data:`REGIMES`):

    dynotears_paper               Erdos-Renyi graphs, Gaussian noise, raw scale       (in claim)
    dynotears_paper_exponential   the same graphs, centered exponential noise         (in claim)
    dynotears_paper_ba_sbm        Barabasi-Albert and block-model graphs, Gaussian    (in claim)
    dynotears_paper_standardized  dynotears_paper with every column z-scored          (out)
    dynotears_paper_uncentered    exponential noise of mean 1, not centered           (out)
    dynotears_sin_tanh            sin of same-step and tanh of lagged parents         (out)

Every solution in a run takes one of the paper's two settings (:data:`SETTINGS`): ``n500``, the
paper's for 500 steps and Andrey's defaults, and ``n50``, the paper's for 50 steps. The default is
the paper's for ``--length``; a run at another length, or at the other settings, is out of claim
(:func:`claim`). The solutions are Andrey's DYNOTEARS, CausalNex's with the same settings and
thresholds (:mod:`~andrey_bench.adapters.causalnex_dynotears`), and the empty window, the floor;
``--centered`` adds CausalNex on centered columns. Each name carries its settings, because
``params`` are not part of a ``run_id``. Every estimate is a window over lags 0 and 1, scored link by
link (:func:`andrey_bench.scoring.window_scores`), with the edge SHD and F1 of the paper.

**CausalNex fits run in their own interpreter.** CausalNex 0.12.2 installs on Python 3.10 with
NumPy below 1.24, so ``benchmarks/start_causalnex.sh`` builds ``.venv-causalnex``, the adapter names
``ANDREY_BENCH_CAUSALNEX_PYTHON``, and the runner starts its fits there.

Use ``--seed-start`` and ``--regimes`` to cut a campaign into small runs, each with its own
``--out``; pairs form inside one run, so keep the full solution list in each. Share the dataset
store, ``ANDREY_BENCH_DATA``, and fill it once with ``--prepare-only``.

    ANDREY_BENCH_CAUSALNEX_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/dynotears_ts.py \\
        --out outputs/dynotears --ladder 5 10 --length 500 --seeds 20 --cores 1
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
from andrey_bench.adapters.andrey_dynotears import AndreyDYNOTEARS
from andrey_bench.adapters.baseline_empty import baseline_empty_window
from andrey_bench.adapters.causalnex_dynotears import PYTHON_ENV_VAR, causalnex_adapter
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores, check_interpreters

#: The regimes, in ``datasets.REGIMES``: the paper's design first, then the three outside it.
REGIMES = [
    "dynotears_paper",
    "dynotears_paper_exponential",
    "dynotears_paper_ba_sbm",
    "dynotears_paper_standardized",
    "dynotears_paper_uncentered",
    "dynotears_sin_tanh",
]

#: The regimes of the paper's design, Pamfil et al. (2020), Appendix B.4.
PAPER_REGIMES = frozenset(REGIMES[:3])

#: The paper's settings, Appendix B.3, in Andrey's names: for 500 steps (Andrey's defaults) and 50.
SETTINGS = {
    "n500": {"lambda1": 0.05, "w_threshold": 0.3, "lag_threshold": 0.1},
    "n50": {"lambda1": 0.2, "w_threshold": 0.3, "lag_threshold": 0.2},
}

#: The paper's settings for each series length it simulates.
PAPER_SETTINGS = {500: "n500", 50: "n50"}

#: The largest lag, the order of the paper's simulations.
TAU_MAX = 1

#: The smallest run that covers every solution.
SMOKE = {"ladder": [5], "seeds": 1, "repeats": 1, "cap_wall_s": 600.0}


def claim(regime: str, length: int, settings: str) -> str:
    """``"in"`` for the paper's design at the paper's settings for its length, else ``"out"``."""
    return "in" if regime in PAPER_REGIMES and PAPER_SETTINGS.get(length) == settings else "out"


def solutions(settings: str, centered: bool) -> list:
    """Andrey's DYNOTEARS, CausalNex's, and the empty window, each named for its settings."""
    values = {**SETTINGS[settings], "tau_max": TAU_MAX}
    field = [
        AndreyDYNOTEARS(name=f"andrey.dynotears.{settings}", **values),
        causalnex_adapter(name=f"causalnex.dynotears.{settings}", **values),
    ]
    if centered:
        field.append(
            causalnex_adapter(
                name=f"causalnex.dynotears.centered.{settings}", centered=True, **values
            )
        )
    return [*field, baseline_empty_window(TAU_MAX)]


def regime_keys(regime: str, d: int, length: int, seeds: range) -> list[DatasetKey]:
    """One series of `regime` per seed, at `d` variables and `length` fitted steps."""
    topology, functional, noise, standardize = datasets.REGIMES[regime]
    return [
        DatasetKey(
            topology=topology,
            functional=functional,
            noise=noise,
            standardize=standardize,
            density=datasets.TEMPORAL_REGIMES[regime],
            d=d,
            n=length,
            seed=seed,
        )
        for seed in seeds
    ]


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Andrey against CausalNex: DYNOTEARS")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument("--ladder", type=int, nargs="+", help="variable counts d, run in order")
    p.add_argument(
        "--regimes",
        nargs="+",
        choices=REGIMES,
        default=REGIMES[:2],
        help="regimes to run, in order (default: the Erdos-Renyi pair of the paper's design)",
    )
    p.add_argument("--length", type=int, default=500, help="fitted steps n per series")
    p.add_argument(
        "--settings",
        choices=sorted(SETTINGS),
        default=None,
        help="the paper's settings for 500 or 50 steps (default: the paper's for --length)",
    )
    p.add_argument("--centered", action="store_true", help="add CausalNex on centered columns")
    p.add_argument("--seeds", type=int, default=20, help="series per cell (default: 20)")
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
    p.add_argument("--smoke", action="store_true", help="run d=5, one seed")
    p.add_argument("--resume", action="store_true", help="continue the run already in --out")
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)
    if not args.ladder:
        p.error("--ladder is required unless --smoke is given")
    settings = args.settings or PAPER_SETTINGS.get(args.length)
    if settings is None:
        p.error(f"--settings is required at --length {args.length}")

    out = Path(args.out)
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    if args.prepare_only:
        store = datasets.store_dir(out)
        keys = [
            k
            for r in args.regimes
            for d in args.ladder
            for k in regime_keys(r, d, args.length, seeds)
        ]
        registry = datasets.materialize(keys, store)
        _log(
            f"materialized {len(keys)} dataset(s) into {store}; registry has {len(registry['keys'])}"
        )
        return 0

    field = solutions(settings, args.centered)
    check_interpreters(field)
    python = os.environ.get(PYTHON_ENV_VAR, "").strip()
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in field],
        "causalnex_python": {PYTHON_ENV_VAR: python},
        "params": {a.name: dict(a.params()) for a in field},
        "regimes": args.regimes,
        "claim": {r: claim(r, args.length, settings) for r in args.regimes},
        "length": args.length,
        "settings": settings,
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
        extra_venvs={"causalnex": Path(python).absolute().parent.parent},
    )
    _log(f"=== Andrey vs CausalNex, DYNOTEARS, CPU -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:16s}: {value}")
    _log(f"{'git_sha':16s}: {prov['git_sha']}")
    _log("")

    errors: list[str] = []
    start = time.perf_counter()
    for d in args.ladder:
        for regime in args.regimes:
            result = run_batch(
                datasets.store_dir(out),
                field,
                regime_keys(regime, d, args.length, seeds),
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
                cost.append(
                    f"{adapter.name}={statistics.median(walls):.3g}s" if walls else statuses
                )
            _log(f"[{regime} d={d}] {'  '.join(cost)} | elapsed {time.perf_counter() - start:.0f}s")

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
        _log(f"  ! {row.name} @ d={row.d} n={row.n} seed={row.data_seed}: {row.error_text}")
    return 1 if errors or len(failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
