#!/usr/bin/env python
"""Benchmark Tetrad's PC, FGES, BOSS, and GRaSP against Andrey's on continuous data.

Runs each Andrey algorithm and Tetrad's on identical data: PC with the Fisher z test at
``alpha = 0.05``; GES, uncapped, in one process and in worker processes, against FGES; BOSS and
GRaSP against Tetrad's, each with the BIC penalty the BOSS/GRaSP lane of the harness runs. The
empty graph provides a quality baseline. Tetrad runs on tetrad-lib's shaded jar, named by
``ANDREY_BENCH_TETRAD_GIN_JAR``, through a Java driver that times Tetrad's search inside its JVM and
reports the JVM's own time apart, in ``process_s``. Every Tetrad setting is written out in the
solution's ``params``.

The data are standardized linear-Gaussian Erdos-Renyi graphs with mean degree 2 and ``n = 10 * d``
rows, as in ``gfci_latent.py``. Every estimate is scored against the true graph's CPDAG by
``shd``. The summary prints each pair's speed-up and how many datasets both packages returned the
same graph on; where they differ, the README's "Tetrad on continuous data" section gives why.

    ANDREY_BENCH_PYTHON=... ANDREY_BENCH_TETRAD_GIN_JAR=... $ANDREY_BENCH_PYTHON \\
        benchmarks/tetrad_cpu.py --out outputs/tetrad-cpu --cap-mem-mb 30000
"""

from __future__ import annotations

import argparse
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from andrey_bench import datasets, memcap, pairing, rundir
from andrey_bench.adapters.andrey_boss import AndreyBOSS
from andrey_bench.adapters.andrey_ges import AndreyGES, _andrey_version
from andrey_bench.adapters.andrey_grasp import AndreyGRaSP
from andrey_bench.adapters.andrey_pc import AndreyPC
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.tetrad_search import TetradBOSS, TetradFGES, TetradGRaSP, TetradPC
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores

#: GES without the parent cap the harness's GES lane sets, as FGES has none, in one process and in
#: worker processes.
_GES_SERIAL, _GES_PARALLEL = (
    AndreyGES(
        name=f"andrey.ges.numpy.{mode}.uncapped",
        backend="numpy",
        mode=mode,
        package_version=_andrey_version(),
        max_parents=None,
    )
    for mode in ("serial", "parallel")
)

_PC, _TETRAD_PC = AndreyPC(), TetradPC()
_FGES = TetradFGES()
_BOSS, _TETRAD_BOSS = AndreyBOSS(), TetradBOSS()
_GRASP, _TETRAD_GRASP = AndreyGRaSP(), TetradGRaSP()

#: Andrey's run of one algorithm and Tetrad's run of it, by ``"<algorithm>"`` and, for GES, its mode.
PAIRS = {
    "pc": (_PC, _TETRAD_PC),
    "ges serial": (_GES_SERIAL, _FGES),
    "ges parallel": (_GES_PARALLEL, _FGES),
    "boss": (_BOSS, _TETRAD_BOSS),
    "grasp": (_GRASP, _TETRAD_GRASP),
}
SOLUTIONS = [
    _PC,
    _TETRAD_PC,
    _GES_SERIAL,
    _GES_PARALLEL,
    _FGES,
    _BOSS,
    _TETRAD_BOSS,
    _GRASP,
    _TETRAD_GRASP,
    BaselineEmpty(),
]

#: Standardized linear-Gaussian Erdos-Renyi graphs with mean degree 2. Part of each dataset's
#: identity: changing it changes every ``dataset_id``.
SLICE = {
    "topology": "er",
    "functional": "linear",
    "noise": "gaussian",
    "standardize": "standardized",
    "density": datasets.CANONICAL_DENSITY,
}

#: Smallest run that touches every solution: two datasets at the smallest size.
SMOKE = {"ladder": [20], "seeds": 2, "repeats": 1, "cap_wall_s": 300.0}

#: Untimed fits before each measurement. Tetrad's warm-up searches run in the JVM of its timed
#: fit, so the timed search finds its code compiled, as a Python package's does.
WARMUP = 1


def ladder_keys(ladder: list[int], seeds: int) -> list[DatasetKey]:
    """Return one :data:`SLICE` dataset per ``(d, seed)`` at ``n = 10 * d``."""
    return [DatasetKey(**SLICE, d=d, n=10 * d, seed=seed) for d in ladder for seed in range(seeds)]


def _log(msg: str = "") -> None:
    """Print and flush batch output immediately."""
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tetrad's PC, FGES, BOSS, GRaSP against Andrey's")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--ladder", type=int, nargs="+", default=[20, 50, 100, 200], help="d sizes, run in order"
    )
    p.add_argument("--seeds", type=int, default=5, help="datasets per rung (default: 5)")
    p.add_argument("--repeats", type=int, default=1, help="timed repeats per fit (default: 1)")
    p.add_argument(
        "--warmup",
        type=int,
        default=WARMUP,
        help="untimed fits per repeat; the wall cap covers them too (default: 1)",
    )
    p.add_argument(
        "--solutions",
        nargs="+",
        default=None,
        help="run only these solution names (default: every solution)",
    )
    p.add_argument("--cap-wall-s", type=float, default=1800.0, help="wall cap per repeat, seconds")
    p.add_argument("--cap-mem-mb", type=float, default=None, help="memory cap per repeat, MB")
    p.add_argument("--mem-route", choices=memcap.ROUTES, default=None, help="cap enforcement route")
    p.add_argument(
        "--cores",
        type=int,
        default=available_cores(),
        help="core budget per fit, which Tetrad's JVM sees whole "
        "(default: every core this job is allowed)",
    )
    p.add_argument(
        "--machine-id",
        default=socket.gethostname(),
        help="machine id stamped on every record and in run_id (default: this host's name)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="run d=20, two seeds, one repeat, and a 300 s cap, replacing those flags",
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
    solutions = SOLUTIONS
    if args.solutions is not None:
        unknown = sorted(set(args.solutions) - {a.name for a in SOLUTIONS})
        if unknown:
            raise SystemExit(f"unknown solutions {unknown}; known: {[a.name for a in SOLUTIONS]}")
        solutions = [a for a in SOLUTIONS if a.name in args.solutions]

    out = Path(args.out)

    # Written before the first fit, so an interrupted benchmark still says what it was asked for.
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in solutions],
        "params": {a.name: a.params() for a in solutions},
        "slice": SLICE,
        "ladder": args.ladder,
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
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())

    _log(f"=== Tetrad against Andrey, continuous data -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:12s}: {value}")
    _log(f"{'git_sha':12s}: {prov['git_sha']}")
    _log("")

    records: list = []
    errors: list[str] = []
    skipped = 0
    start = time.perf_counter()

    for d in args.ladder:
        # One rung at a time, so a rung's data is materialized only when the ladder reaches it.
        result = run_batch(
            datasets.store_dir(out),
            solutions,
            ladder_keys([d], args.seeds),
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
        for adapter in solutions:
            mine = [r for r in result.records if r.name == adapter.name]
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
    summarize(table)
    return 1 if errors else 0


def summarize(table: pd.DataFrame) -> None:
    """Print each solution's finished fits, median search time, whole-call time, and median
    ``shd`` per ``d``, then each pair's speed-up and agreement.

    ``process_s`` is Tetrad's whole fit call, with the JVM start; it is empty for the others. A
    pair's speed-up is Tetrad's median ``wall_s`` over Andrey's on the datasets both finished, and
    ``same`` counts the datasets both returned the same graph on, of those both finished.
    """
    if table.empty:
        return
    ok = table[table["status"] == "ok"]
    _log("")
    _log(f"{'solution':32s} {'d':>4s} {'ok/all':>7s} {'wall_s':>8s} {'process_s':>9s} {'shd':>6s}")
    for (name, d), rows in table.groupby(["name", "d"], sort=True):
        done = ok[(ok["name"] == name) & (ok["d"] == d)]
        _log(
            f"{name:32s} {d:4d} {len(done):3d}/{len(rows):<3d} {done['wall_s'].median():8.3g} "
            f"{done['process_s'].median():9.3g} {done['shd'].median():6.3g}"
        )
    _log("")
    _log(f"{'pair':14s} {'d':>4s} {'paired':>6s} {'speed-up':>8s} {'same':>6s}")
    for method, (andrey, tetrad) in PAIRS.items():
        for d in sorted(ok["d"].unique()):
            a = ok[(ok["name"] == andrey.name) & (ok["d"] == d)]
            t = ok[(ok["name"] == tetrad.name) & (ok["d"] == d)]
            both = a.set_index("dataset_id")["wall_s"].index.intersection(
                t.set_index("dataset_id")["wall_s"].index
            )
            if not both.empty:
                a_s = a.set_index("dataset_id")["wall_s"][both].median()
                t_s = t.set_index("dataset_id")["wall_s"][both].median()
                same = "{}/{}".format(*pairing.same_graph(a, t))
                _log(f"{method:14s} {d:4d} {len(both):6d} {t_s / a_s:8.3g} {same:>6s}")


if __name__ == "__main__":
    raise SystemExit(main())
