#!/usr/bin/env python
"""A benchmark: DirectLiNGAM on CPU against the same algorithm on GPU.

``andrey.direct_lingam.torch-cuda`` is the one GPU-shaped workload the harness can measure today.
The comparison is Andrey's two backends against each other and against the reference lingam
package, on identical data.

Two things make this different from ``ges_cpu.py``, and both are visible in the literals below.

**The regime is not Gaussian.** DirectLiNGAM is unidentifiable on Gaussian noise, so this runs the
``lingam_sf`` regime (scale-free, linear, uniform noise). That is not a special case for the LiNGAM
solutions — it is a different task, and the Gaussian slice does not apply to them.

**The device is declared per group.** The Andrey adapters pin their device *inside* ``fit``
(``andrey.config(backend=...)``), which outranks ``ANDREY_DEVICE``; setting the env alone moves
nothing. But ``device`` is a ``run_id`` field that ``runner.resolved_env_fields`` reads off the
child's env, so a run that leaves it unset writes ``device="cpu"`` on records produced on a GPU.
``extra_env`` is per ``run_batch`` call, not per adapter, so the solutions are split into groups and
each group declares what its adapters pin. Composing calls like this is what a benchmark script
is for.

Runs on a CPU node too — the CUDA fits fail and become ``status=error`` records, which is the harness
working, not a reason for a gate.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/lingam_gpu.py \\
        --out outputs/lingam-gpu --ladder 20 50 100 --cap-mem-mb 32000
"""

from __future__ import annotations

import argparse
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from andrey_bench import datasets, memcap, rundir
from andrey_bench.adapters.andrey_direct_lingam import andrey_direct_lingam_adapters
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.lingam_direct import LingamDirect
from andrey_bench.datasets import CANONICAL_DENSITY, DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores

_ANDREY = {a.backend: a for a in andrey_direct_lingam_adapters()}

#: Solutions grouped by the device they pin, with the env each group declares. Nothing here changes
#: what a solution computes — the pin inside ``fit`` already decided that — it makes the ``device``
#: field on the record true.
GROUPS = [
    ([_ANDREY["numpy"], LingamDirect(), BaselineEmpty()], None),
    ([_ANDREY["torch-cuda"]], {"ANDREY_DEVICE": "cuda"}),
]

#: Every solution across every group, in one list — for the run_meta record and the per-rung log.
SOLUTIONS = [adapter for group, _ in GROUPS for adapter in group]

#: The ``lingam_sf`` regime, spelled out.
SLICE = {
    "topology": "scale_free",
    "functional": "linear",
    "noise": "uniform",
    "standardize": "standardized",
    "density": CANONICAL_DENSITY,
}

#: Untimed fits before each timed one. The CUDA variant allocates its context and loads kernels on
#: the first call; without a warm-up that setup is published as its speed.
WARMUP = 1


def ladder_keys(ladder: list[int], seeds: int) -> list[DatasetKey]:
    """One dataset per ``(d, seed)`` on :data:`SLICE`, at ``n = 10·d`` and seeds ``0 .. seeds-1``."""
    return [DatasetKey(**SLICE, d=d, n=10 * d, seed=seed) for d in ladder for seed in range(seeds)]


def _log(msg: str = "") -> None:
    """Print and flush immediately — a batch job's stdout is only useful unbuffered."""
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="DirectLiNGAM CPU vs GPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument("--ladder", type=int, nargs="+", required=True, help="d sizes, run in order")
    p.add_argument("--seeds", type=int, default=3, help="datasets per rung (default: 3)")
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
        "--machine-id",
        default=socket.gethostname(),
        help="machine id stamped on every record and in run_id (default: this host's name)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue the run already in --out: skip units whose records are on disk. Requires the "
        "same run_meta.json, so pass the flags the first run was given",
    )
    args = p.parse_args(argv)

    out = Path(args.out)

    # Written before the first fit, so an interrupted benchmark still says what it was asked for.
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in SOLUTIONS],
        "device groups": {
            a.name: (env or {}).get("ANDREY_DEVICE", "cpu") for group, env in GROUPS for a in group
        },
        "slice": SLICE,
        "ladder": args.ladder,
        "seeds": args.seeds,
        "repeats": args.repeats,
        "warmup": WARMUP,
        "cap_wall_s": args.cap_wall_s,
        "cap_mem_mb": args.cap_mem_mb,
        "mem_route": memcap.resolve_route(args.cap_mem_mb, args.mem_route),
        "cores": args.cores,
        "machine_id": args.machine_id,
    }
    # Reads run_meta.json before writing it, so --resume has something to check against.
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())

    _log(f"=== DirectLiNGAM, CPU vs GPU -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:14s}: {value}")
    _log(f"{'git_sha':14s}: {prov['git_sha']}")
    _log(f"{'gpu':14s}: {prov['gpu']}")
    _log("")

    records: list = []
    errors: list[str] = []
    skipped = 0
    start = time.perf_counter()

    for d in args.ladder:
        keys = ladder_keys([d], args.seeds)
        rung: list = []
        # One call per device declaration. The datasets are materialized by the first call and the
        # second finds the same keys, so both groups read identical bytes.
        for group, extra_env in GROUPS:
            result = run_batch(
                datasets.store_dir(out),
                group,
                keys,
                machine_id=args.machine_id,
                cap_wall_s=args.cap_wall_s,
                cap_mem_mb=args.cap_mem_mb,
                mem_route=args.mem_route,
                repeats=args.repeats,
                warmup=WARMUP,
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
        for adapter in SOLUTIONS:
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
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
