#!/usr/bin/env python
"""Benchmark Andrey GES and XGES over a ladder of graph sizes on CPU.

XGES is a distinct search in the GES family. It shares GES's CPDAG search space and
linear-Gaussian BIC score but uses a different move order and an optional extended search.
causal-learn supplies a GES reference, separating differences between GES implementations from
differences between GES and XGES.

The Andrey `numpy` and `numba` serial rows use one process and the same core budget as XGES. The
`numpy.parallel` row uses a worker pool, so its timing also reflects the additional workers.

``--ges-min-work`` comes with that arm. Andrey fans a pass out only when its estimated candidate
work exceeds the gate; below it the pass runs the identical in-process scan as the serial backend,
and the two solutions publish one code path under two names. The shipped default is a fixed constant
in ``src/andrey/search/_parallel_ges.py``. Declare the campaign's gate and report it beside the
table; ``andrey calibrate`` measures a suitable value for the target host.

A copy of ``ges_cpu.py`` with a different solution list, which is how a benchmark is written.

**XGES fits run in their own interpreter.** The package pins `numpy<2`, so the adapter names
``ANDREY_BENCH_XGES_PYTHON`` and the runner spawns its fits there instead of in bench-env. Both
environments are recorded in ``provenance.json``. Build the second one with
``benchmarks/start_xges.sh``; a run started without it fails at launch rather than silently
dropping the solution.

``--xges-alpha`` declares the penalty XGES searches under. It matters because ``alpha`` and Andrey's
``lambda_value`` scale the same ``log n`` per-parameter term (see the adapter), so the shipped
defaults - 2.0 against 1.0 - would put a sparsity setting into the quality column. The default here
is the matched value.

    ANDREY_BENCH_XGES_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/xges_ges.py \\
        --out outputs/xges --ladder 20 50 100 --cap-mem-mb 32000
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
from andrey_bench.adapters.andrey_ges import andrey_ges_adapters
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.causal_learn_ges import CausalLearnGES
from andrey_bench.adapters.xges_ges import (
    DEFAULT_ALPHA,
    PYTHON_ENV_VAR,
    SHIPPED_ALPHA,
    xges_adapter,
)
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import BENCH_ROOT, available_cores, check_interpreters

#: Where ``benchmarks/start_xges.sh`` builds the XGES environment. Only a default for the error
#: message: what actually runs is whatever ``$ANDREY_BENCH_XGES_PYTHON`` names.
DEFAULT_XGES_VENV = BENCH_ROOT / ".venv-xges"


def xges_venv() -> Path:
    """The venv XGES fits actually run in, read from the variable the runner reads.

    Deriving this from the interpreter rather than hardcoding ``.venv-xges`` is the whole point:
    the runner resolves ``$ANDREY_BENCH_XGES_PYTHON``, so a fixed path here would freeze into
    ``provenance.json`` and ``run_meta.json`` the versions of an environment the fits never
    entered - which is the confusion this benchmark exists to prevent, committed by the benchmark
    itself.
    """
    raw = os.environ.get(PYTHON_ENV_VAR, "").strip()
    if not raw:
        raise SystemExit(
            f"${PYTHON_ENV_VAR} is unset. Build the XGES environment with "
            f"benchmarks/start_xges.sh and export it (default: {DEFAULT_XGES_VENV}/bin/python)."
        )
    # <venv>/bin/python -> <venv>. Not resolve(): the venv's python is a symlink to the base
    # interpreter, and following it would name the base install instead of the environment.
    python = Path(raw).absolute()
    venv = python.parent.parent
    # That derivation assumes the standard layout, and provenance then probes `<venv>/bin/python`
    # again. An interpreter somewhere else - a wrapper script, a versioned name - would leave
    # provenance describing a different interpreter that happens to sit at that path, which is a
    # silent mislabel rather than a missing file. Checked, so it fails instead.
    if venv / "bin" / "python" != python:
        raise SystemExit(
            f"${PYTHON_ENV_VAR}={raw!r} is not a <venv>/bin/python, so the environment recorded in "
            f"provenance.json would not be the one the fits ran in. Point it at a venv interpreter."
        )
    return venv


#: The XGES package's default BIC weight, on the scale both algorithms share.
HEAVY_LAMBDA = SHIPPED_ALPHA


def solutions(alpha: float) -> list:
    """The solutions at matched and XGES-default penalties, with a reference and baseline.

    Built rather than declared because XGES carries a penalty from the command line and reads its
    own version out of the interpreter that will run it - neither is known at import.

    Each penalty has distinct solution names because ``params`` are excluded from ``run_id``; using
    one name for two weights would collide in the part files. causal-learn supplies the GES
    reference used to distinguish implementation differences within GES from algorithm differences
    between GES and XGES.
    """
    return [
        *andrey_ges_adapters(),
        *andrey_ges_adapters(lambda_value=HEAVY_LAMBDA, suffix=".heavy"),
        CausalLearnGES(),
        xges_adapter(alpha=alpha),
        xges_adapter(alpha=SHIPPED_ALPHA, name="xges.ges.shipped"),
        BaselineEmpty(),
    ]


def groups(field: list, min_work: int | None):
    """Solutions split by the environment each declares, as ``(solutions, extra_env)`` pairs.

    Only parallel-mode Andrey reads the work gate. A serial fit is pinned to one worker and never
    consults it, and no competitor has one, so declaring it on them would claim a knob they do not
    have.
    """
    if min_work is None:
        return [(field, None)]
    gated = [a for a in field if getattr(a, "mode", None) == "parallel"]
    rest = [a for a in field if a not in gated]
    pairs = [(rest, None), (gated, {"ANDREY_GES_PARALLEL_MIN_WORK": str(min_work)})]
    return [(group, env) for group, env in pairs if group]


#: The canonical sparse slice: linear-Gaussian on Erdos-Renyi at average degree 2, where GES is
#: identifiable. The same slice ``ges_cpu.py`` uses, so the two campaigns share datasets.
SLICE = {
    "topology": "er",
    "functional": "linear",
    "noise": "gaussian",
    "standardize": "standardized",
    "density": datasets.CANONICAL_DENSITY,
}

#: Untimed fits before each timed one. Both numba paths compile on their first call, and without a
#: warm-up that compilation is published as their speed.
WARMUP = 1


def ladder_keys(ladder: list[int], seeds: int, seed_start: int = 0) -> list[DatasetKey]:
    """One dataset per ``(d, seed)`` on :data:`SLICE`, at ``n = 10*d``.

    ``seed_start`` shifts the window so a rung's seeds can be split across jobs. That is the one
    axis a campaign may safely fan out on below the rung: a ratio forms within
    ``(run_dir, dataset_id)``, so every solution for a given dataset has to stay in one job, and one
    seed is one dataset. Splitting by *solution* instead would put the compared arms in different
    run directories on different nodes, and no ratio would form at all.

    ``dataset_id`` is a content hash of the generator call, so seed 1 names the same bytes whether
    it was reached by ``--seeds 3`` or by ``--seeds 1 --seed-start 1``.
    """
    return [
        DatasetKey(**SLICE, d=d, n=10 * d, seed=seed)
        for d in ladder
        for seed in range(seed_start, seed_start + seeds)
    ]


def _log(msg: str = "") -> None:
    """Print and flush immediately - a batch job's stdout is only useful unbuffered."""
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Andrey GES vs XGES, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument("--ladder", type=int, nargs="+", required=True, help="d sizes, run in order")
    p.add_argument("--seeds", type=int, default=3, help="datasets per rung (default: 3)")
    p.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="first seed of the window this run takes, so one rung's seeds can be split across "
        "jobs. Every solution for a given seed stays in one job, which is what a paired ratio "
        "needs; splitting by solution instead would leave no ratio to form",
    )
    p.add_argument("--repeats", type=int, default=5, help="timed repeats per fit (default: 5)")
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
        "--xges-alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="XGES BIC penalty, on the same scale as andrey's lambda_value. The default matches "
        "andrey's shipped 1.0; the package's own default is 2.0, which is twice the penalty",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=WARMUP,
        help="untimed fits before each timed one; use 0 when measuring how far a solution reaches, "
        "so the cap buys one fit rather than two (default: 1)",
    )
    p.add_argument(
        "--ges-min-work",
        type=int,
        default=None,
        help="parallel work gate (estimated candidate evals) for andrey's parallel mode; omitted "
        "leaves the shipped default in force. Recorded in run_meta.json, so --resume refuses a "
        "continuation under a different value rather than pooling two gates in one table",
    )
    p.add_argument(
        "--solutions",
        nargs="+",
        default=None,
        help="run only these solution names (default: every solution). A rung the probe showed a "
        "solution cannot finish costs the wall cap to learn nothing twice, so the top of a ladder "
        "is run with the solution list trimmed to what reaches it",
    )
    p.add_argument(
        "--machine-id",
        default=socket.gethostname(),
        help="machine id stamped on every record and in run_id (default: this host's name)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue the run already in --out: skip units whose records are on disk. Needs "
        "the same run_meta.json, so pass the flags the first run was given",
    )
    args = p.parse_args(argv)

    out = Path(args.out)
    field = solutions(args.xges_alpha)
    if args.solutions:
        known = {a.name for a in field}
        unknown = sorted(set(args.solutions) - known)
        # A typo would otherwise run fewer solutions and say nothing, and the missing rows would
        # read as a solution that failed rather than one that was never asked.
        if unknown:
            p.error(f"unknown solutions {unknown}; the solutions are {sorted(known)}")
        field = [a for a in field if a.name in args.solutions]

    if args.seeds < 1:
        p.error(
            f"--seeds must be at least 1, got {args.seeds}: a run with no datasets would "
            "finish immediately and be recorded as complete"
        )
    if args.seed_start < 0:
        p.error(f"--seed-start must not be negative, got {args.seed_start}")
    if args.repeats < 1:
        p.error(
            f"--repeats must be at least 1, got {args.repeats}: a run with no timed fits would "
            "write no records and exit 0"
        )
    if args.warmup < 0:
        p.error(f"--warmup must not be negative, got {args.warmup}")

    # Every interpreter in the *whole* solution list, before any rung runs. `run_batch` pre-flights
    # only the group it was handed, and `groups()` can put the solutions that share an interpreter
    # first - so a per-group check alone would measure a whole group before rejecting the next
    # one's missing environment.
    check_interpreters(field)

    # Resolved from the variable the runner reads, and only when a solution in this run needs it -
    # a solution list trimmed to Andrey arms has no second environment to record.
    venv = xges_venv() if any(getattr(a, "python_env", None) for a in field) else None

    # Written before the first fit, so an interrupted benchmark still says what it was asked for.
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in field],
        "xges_alpha": args.xges_alpha,
        "ges_min_work": args.ges_min_work,
        "gated solutions": [
            a.name for a in field if args.ges_min_work is not None and a.mode == "parallel"
        ],
        "xges_python": {PYTHON_ENV_VAR: os.environ.get(PYTHON_ENV_VAR, "")},
        "params": {a.name: dict(a.params()) for a in field},
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
    # Reads run_meta.json before writing it, so --resume has something to check against.
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    prov = rundir.provenance_once(
        out,
        timestamp=datetime.now(timezone.utc).isoformat(),
        extra_venvs={"xges": venv} if venv else None,
    )

    _log(f"=== Andrey GES vs XGES, CPU -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:14s}: {value}")
    _log(f"{'git_sha':14s}: {prov['git_sha']}")
    xges_env = prov.get("environments", {}).get("xges")
    envs = [("bench", prov["blas"])]
    if xges_env:
        _log(f"{'xges':14s}: {xges_env['versions'].get('packages', {}).get('xges')}")
        envs.append(("xges", xges_env["blas"]))
    # The two environments carry different numpy builds - that is what the pin forces - and their
    # BLAS thread ceilings differ with them. Printed because it bounds what the core budget buys.
    for label, entry in envs:
        blas = entry.get("blas") or {}
        _log(
            f"{'blas.' + label:14s}: numpy {entry.get('numpy')} "
            f"{blas.get('openblas configuration')}"
        )
    _log("")

    records: list = []
    errors: list[str] = []
    skipped = 0
    start = time.perf_counter()

    for d in args.ladder:
        # One rung at a time, so a rung's data is materialized only when the ladder reaches it.
        rung: list = []
        for group, extra_env in groups(field, args.ges_min_work):
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

        # What each solution cost at this rung, or what stopped it.
        ran = []
        for adapter in field:
            mine = [r for r in rung if r.name == adapter.name]
            walls = [r.wall_s for r in mine if r.status == "ok" and r.wall_s is not None]
            statuses = sorted({r.status for r in mine})
            cost = (
                f"{statistics.median(walls):.3g}s" if walls else "/".join(statuses) or "no-records"
            )
            ran.append(f"{adapter.name}={cost}")
        _log(f"[d={d}] {'  '.join(ran)} | elapsed {time.perf_counter() - start:.0f}s")

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

    # A fit that launched and then failed produces an `error` *record*, not an entry in `errors` -
    # so counting only the latter lets a run whose every fit errored exit 0 and read as a success.
    # Counted from the table, not this sitting's records: a resume skips every part file already on
    # disk, errored or not, so an earlier sitting's failure is only in the table. `timeout` and
    # `oom` are excluded: those are measurements of a cap, and a reach rung is expected to produce
    # them.
    failed = table[table.status == "error"]
    if len(failed):
        _log(f"failed fits : {len(failed)}")
        for row in failed.head(5).itertuples():
            _log(f"  ! {row.name} @ d={row.d} seed={row.data_seed}: {row.error_text}")
    return 1 if errors or len(failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
