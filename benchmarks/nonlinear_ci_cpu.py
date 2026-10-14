#!/usr/bin/env python
"""Benchmark PC with the nonlinear CI tests on CPU: Andrey against causal-learn and tigramite.

Runs Andrey's PC with each nonlinear test (KCI, RCoT, CMIknn, GPDC) and with Fisher-Z, causal-learn's
PC with KCI and RCoT, tigramite's PC with CMIknn and GPDC, and the empty graph as a quality baseline.
The data is additive nonlinear noise (``x = sum w (p + sin p) + e``, Gaussian ``e``) on Erdos-Renyi
graphs at average degree 2, standardized. A nonlinear test's cost grows with the rows, so rows are
their own axis (``--rows``) rather than a multiple of ``d``. Every solution is serial and runs at
Andrey's defaults; the adapters map them to each package's names.

tigramite's solutions run in an environment of their own: build it with
``benchmarks/start_tigramite.sh`` and export ``ANDREY_BENCH_TIGRAMITE_PYTHON``. A run whose
``--solutions`` leaves them out does not need it.

    $ANDREY_BENCH_PYTHON benchmarks/nonlinear_ci_cpu.py --out outputs/nonlinear-ci \\
        --cap-mem-mb 32000
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

from andrey_bench import datasets, memcap, rundir
from andrey_bench.adapters.andrey_pc import AndreyPC
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.causal_learn_pc import CausalLearnPC
from andrey_bench.adapters.tigramite_pc import PYTHON_ENV_VAR, tigramite_pc_adapter
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores, check_interpreters

#: Andrey's CI tests in this lane; ``fisherz`` shows what partial correlation gets on this data.
ANDREY_TESTS = ("fisherz", "kci", "rcot", "cmiknn", "gpdc")

#: Additive nonlinear noise on Erdos-Renyi graphs.
REGIME = "anm_er"

#: Smallest run that touches every solution: one tiny dataset.
SMOKE = {"ladder": [4], "rows": [100], "seeds": 1, "repeats": 1, "cap_wall_s": 300.0}

#: Untimed fits before each measurement.
WARMUP = 1


def solutions() -> list:
    """Every solution, Andrey's first; tigramite's read their interpreter from the environment."""
    return [
        *[AndreyPC(test) for test in ANDREY_TESTS],
        CausalLearnPC("kci"),
        CausalLearnPC("rcot"),
        tigramite_pc_adapter("cmiknn"),
        tigramite_pc_adapter("gpdc"),
        BaselineEmpty(),
    ]


def regime_slice() -> dict:
    """Return the :class:`DatasetKey` coordinates of :data:`REGIME`."""
    topology, functional, noise, standardize = datasets.REGIMES[REGIME]
    return {
        "topology": topology,
        "functional": functional,
        "noise": noise,
        "standardize": standardize,
        "density": datasets.CANONICAL_DENSITY,
    }


def ladder_keys(ladder: list[int], rows: list[int], seeds: int) -> list[DatasetKey]:
    """Return one dataset per ``(d, n, seed)``."""
    return [
        DatasetKey(**regime_slice(), d=d, n=n, seed=seed)
        for d in ladder
        for n in rows
        for seed in range(seeds)
    ]


def tigramite_venv() -> Path | None:
    """The venv tigramite's fits run in, from the variable the runner reads; ``None`` if unset."""
    raw = os.environ.get(PYTHON_ENV_VAR, "").strip()
    # <venv>/bin/python -> <venv>, without following the symlink to the base interpreter.
    return Path(raw).parent.parent if raw else None


def _log(msg: str = "") -> None:
    """Print and flush batch output immediately."""
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PC with nonlinear CI tests, three packages, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--ladder",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="d sizes (default: 5 10 20)",
    )
    p.add_argument(
        "--rows", type=int, nargs="+", default=[200, 500], help="n sizes (default: 200 500)"
    )
    p.add_argument("--seeds", type=int, default=10, help="datasets per (d, n) (default: 10)")
    p.add_argument("--repeats", type=int, default=1, help="timed repeats per fit (default: 1)")
    p.add_argument(
        "--warmup", type=int, default=WARMUP, help="untimed fits per repeat (default: 1)"
    )
    p.add_argument("--cap-wall-s", type=float, default=3600.0, help="wall cap per repeat, seconds")
    p.add_argument("--cap-mem-mb", type=float, default=None, help="memory cap per repeat, MB")
    p.add_argument("--mem-route", choices=memcap.ROUTES, default=None, help="cap enforcement route")
    p.add_argument(
        "--cores",
        type=int,
        default=available_cores(),
        help="core budget per fit (default: every core this job is allowed)",
    )
    p.add_argument(
        "--solutions",
        default=None,
        help="run only these solution names, comma-separated (default: every solution)",
    )
    p.add_argument(
        "--machine-id",
        default=socket.gethostname(),
        help="machine id stamped on every record and in run_id (default: this host's name)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="run d=4, n=100, one seed, one repeat, and a 300 s cap, replacing those flags",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue the run already in --out: skip units whose records are on disk. Requires "
        "the same run_meta.json, so pass the flags the first run was given",
    )
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)

    field = solutions()
    if args.solutions is not None:
        chosen = set(args.solutions.split(","))
        known = {a.name for a in field}
        unknown = sorted(chosen - known)
        # A typo would otherwise run fewer solutions, and the missing rows would read as failures.
        if unknown:
            p.error(f"unknown solutions {unknown}; the solutions are {sorted(known)}")
        field = [a for a in field if a.name in chosen]

    # Every interpreter before any dataset, and tigramite's environment only when it is used.
    check_interpreters(field)
    venv = tigramite_venv() if any(getattr(a, "python_env", None) for a in field) else None

    out = Path(args.out)
    # Written before the first fit, so an interrupted benchmark still says what it was asked for.
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in field],
        "params": {a.name: dict(a.params()) for a in field},
        "tigramite_python": {PYTHON_ENV_VAR: os.environ.get(PYTHON_ENV_VAR, "")},
        "slice": regime_slice(),
        "ladder": args.ladder,
        "rows": args.rows,
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

    _log(f"=== PC with nonlinear CI tests, Andrey, causal-learn, and tigramite -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:16s}: {value}")
    _log(f"{'git_sha':16s}: {prov['git_sha']}")
    _log("")

    records: list = []
    errors: list[str] = []
    skipped = 0
    start = time.perf_counter()

    for d in args.ladder:
        for n in args.rows:
            # One rung at a time, so a rung's data is materialized only when the ladder reaches it.
            result = run_batch(
                datasets.store_dir(out),
                field,
                ladder_keys([d], [n], args.seeds),
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
            _log(f"[d={d} n={n}] {'  '.join(ran)} | elapsed {time.perf_counter() - start:.0f}s")

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
    _summarize(table)
    return 1 if errors else 0


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """One row per solution, ``d``, and ``n``: finished fits, median wall time, and median SHD.

    Every solution returns a CPDAG, so ``shd`` compares across the table. A timeout has no score
    and is not counted as any graph; ``ok`` and ``fits`` show how many fits each median covers.
    """
    ok = table[table["status"] == "ok"]
    rows = []
    for (name, d, n), fits in table.groupby(["name", "d", "n"], sort=True):
        done = ok[(ok["name"] == name) & (ok["d"] == d) & (ok["n"] == n)]
        rows.append(
            {
                "name": name,
                "d": int(d),
                "n": int(n),
                "ok": len(done),
                "fits": len(fits),
                "wall_s": done["wall_s"].median(),
                "shd": done["shd"].median(),
            }
        )
    return pd.DataFrame(rows)


def _summarize(table: pd.DataFrame) -> None:
    """Print :func:`summary`, one line per solution, ``d``, and ``n``."""
    if table.empty:
        return
    _log("")
    _log(f"{'solution':28s} {'d':>4s} {'n':>5s} {'ok/all':>7s} {'wall_s':>8s} {'shd':>6s}")
    for row in summary(table).itertuples():
        _log(
            f"{row.name:28s} {row.d:4d} {row.n:5d} {row.ok:3d}/{row.fits:<3d} "
            f"{row.wall_s:8.3g} {row.shd:6.3g}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
