#!/usr/bin/env python
"""Benchmark the categorical tests and score on CPU: Andrey against causal-learn.

Runs PC with the chi-square and the G-square test and GES with the BDeu score from both packages on
the ``discrete_er`` regime: category codes sampled from a discrete network with Dirichlet tables on
an Erdos-Renyi DAG (:data:`datasets.DISCRETE_SETTINGS` gives the categories per node and the
concentration). The empty graph is the quality baseline. Every solution is serial, and both packages
run a method with the same settings: ``alpha=0.05`` for PC, BDeu's default priors and at most four
parents for GES. Andrey's GES runs on the numpy backend, so no compile lands in a fit.

Every estimate is scored against the true DAG's CPDAG (``shd`` and the skeleton and arrowhead
columns); ``--rescore`` recomputes those columns from the stored estimates. A timeout is a timeout:
it has no scores.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/discrete_cpu.py \\
        --out outputs/discrete-cpu --cap-mem-mb 32000
"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from andrey import GraphStructure
from andrey_bench import datasets, memcap, rundir, scoring
from andrey_bench.adapters.andrey_ges import andrey_ges_bdeu
from andrey_bench.adapters.andrey_pc import AndreyPCChisq, AndreyPCGsq
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.causal_learn_ges import CausalLearnGESBDeu
from andrey_bench.adapters.causal_learn_pc import CausalLearnPCChisq, CausalLearnPCGsq
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores

REGIME = "discrete_er"

#: The solutions, as (Andrey, causal-learn) pairs that run one method with one setting, then the
#: empty graph.
PAIRS = {
    "pc chisq": (AndreyPCChisq(), CausalLearnPCChisq()),
    "pc gsq": (AndreyPCGsq(), CausalLearnPCGsq()),
    "ges bdeu": (andrey_ges_bdeu(), CausalLearnGESBDeu()),
}
SOLUTIONS = [*(a for pair in PAIRS.values() for a in pair), BaselineEmpty()]

#: Smallest run that touches every solution: one tiny dataset.
SMOKE = {"ladder": [5], "seeds": 1, "n": 500, "cap_wall_s": 300.0}

#: Untimed fits before each measurement, so imports and first-call caches stay out of the timing.
#: The per-repeat wall cap covers warm-up and fit together.
WARMUP = 1


def regime_slice() -> dict:
    """Return the :class:`DatasetKey` coordinates of the categorical regime."""
    topology, functional, noise, standardize = datasets.REGIMES[REGIME]
    return {
        "topology": topology,
        "functional": functional,
        "noise": noise,
        "standardize": standardize,
        "density": datasets.CANONICAL_DENSITY,
    }


def ladder_keys(ladder: list[int], seeds: int, n: int) -> list[DatasetKey]:
    """Return one dataset of `n` rows per ``(d, seed)``."""
    return [
        DatasetKey(**regime_slice(), d=d, n=n, seed=seed) for d in ladder for seed in range(seeds)
    ]


def _log(msg: str = "") -> None:
    """Print and flush batch output immediately."""
    print(msg, flush=True)


#: The metric columns :func:`rescore` recomputes.
SCORE_COLUMNS = (
    "shd",
    "skeleton_precision",
    "skeleton_recall",
    "skeleton_f1",
    "arrowhead_precision",
    "arrowhead_recall",
    "arrowhead_f1",
)


def rescore(
    out: Path,
    records_dir: Path,
    keys: Mapping[str, DatasetKey],
    columns: tuple[str, ...] = SCORE_COLUMNS,
) -> None:
    """Recompute `columns` of every part file from its stored estimates.

    `keys` maps each ``dataset_id`` of the run to its :class:`DatasetKey`. Each fit's estimate is
    read back and scored with :func:`scoring.score_result` against the true DAG; a fit with no
    estimate, or a column its estimate's kind has no score for, keeps ``NaN``. Each part is
    rewritten atomically.
    """
    store = datasets.store_dir(out)
    kinds = datasets.load_registry(store)["keys"]
    truths: dict[str, GraphStructure] = {}
    for part in sorted(records_dir.glob("*.parquet")):
        frame = pd.read_parquet(part)
        for column in columns:
            frame[column] = float("nan")
        for index, record in frame.iterrows():
            estimate = rundir.read_structure(out, record.run_id) if record.status == "ok" else None
            if estimate is None:
                continue
            if record.dataset_id not in truths:
                if record.dataset_id not in keys:
                    raise SystemExit(
                        f"{part.name}: dataset {record.dataset_id} is not in this run's ladder; "
                        "pass the first run's flags"
                    )
                key = keys[record.dataset_id]
                _, marks = datasets.load(store, key)
                truths[record.dataset_id] = GraphStructure.from_numpy(
                    marks, kind=kinds[key.label]["graph_kind"]
                )
            metrics = scoring.score_result(estimate, truths[record.dataset_id])
            for column in columns:
                frame.at[index, column] = float(metrics.get(column, float("nan")))
        scratch = part.with_name(f".{part.name}.tmp")
        frame.to_parquet(scratch, index=False)
        os.replace(scratch, part)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Categorical tests and score, Andrey and causal-learn")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--ladder", type=int, nargs="+", default=[10, 20, 40], help="d sizes (default: 10 20 40)"
    )
    p.add_argument("--seeds", type=int, default=5, help="datasets per rung (default: 5)")
    p.add_argument("--n", type=int, default=2000, help="rows per dataset (default: 2000)")
    p.add_argument("--repeats", type=int, default=1, help="timed repeats per fit (default: 1)")
    p.add_argument(
        "--warmup",
        type=int,
        default=WARMUP,
        help="untimed fits per repeat; the wall cap covers them too (default: 1)",
    )
    p.add_argument("--cap-wall-s", type=float, default=1800.0, help="wall cap per repeat, seconds")
    p.add_argument("--cap-mem-mb", type=float, default=None, help="memory cap per repeat, MB")
    p.add_argument("--mem-route", choices=memcap.ROUTES, default=None, help="cap enforcement route")
    p.add_argument(
        "--cores",
        type=int,
        default=available_cores(),
        help="core budget per fit (default: every core this job is allowed)",
    )
    p.add_argument(
        "--machine-id",
        default=socket.gethostname(),
        help="machine id stamped on every record and in run_id (default: this host's name)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="run d=5, one seed, 500 rows, and a 300 s cap, replacing those flags",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue the run already in --out: skip units whose records are on disk. Requires "
        "the same run_meta.json, so pass the flags the first run was given",
    )
    p.add_argument(
        "--rescore",
        action="store_true",
        help="fit nothing: recompute the score columns of the run already in --out and rewrite "
        "runs.parquet. Takes the first run's flags, as --resume does",
    )
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)

    out = Path(args.out)
    keys = {key.digest: key for key in ladder_keys(args.ladder, args.seeds, args.n)}

    # Written before the first fit, so an interrupted benchmark still says what it was asked for.
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in SOLUTIONS],
        "slice": regime_slice(),
        "generator settings": datasets.DISCRETE_SETTINGS,
        "ladder": args.ladder,
        "n": args.n,
        "seeds": args.seeds,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cap_wall_s": args.cap_wall_s,
        "cap_mem_mb": args.cap_mem_mb,
        "mem_route": memcap.resolve_route(args.cap_mem_mb, args.mem_route),
        "cores": args.cores,
        "machine_id": args.machine_id,
    }
    if args.rescore:
        # Fits nothing, so the run's STATUS.json and sittings stay as the fitting runs left them.
        records_dir = out / rundir.RECORDS_DIRNAME
        if not any(records_dir.glob("*.parquet")):
            raise SystemExit(f"--rescore: {records_dir} holds no records")
        rescore(out, records_dir, keys)
        table = rundir.assemble(records_dir)
        table.to_parquet(out / "runs.parquet", index=False)
        _log(f"rescored    : {len(table)} rows  -> {out / 'runs.parquet'}")
        _summarize(table)
        return 0
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())

    _log(f"=== Categorical tests and score, Andrey and causal-learn -> {out} ===")
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
            SOLUTIONS,
            ladder_keys([d], args.seeds, args.n),
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
        for adapter in SOLUTIONS:
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
    _summarize(table)
    return 1 if errors else 0


def _summarize(table: pd.DataFrame) -> None:
    """Print, per solution and rung, the finished fits with median wall time and SHD, then each
    pair's speed ratio.

    Wall time and SHD are medians over the finished fits; the F1 columns are means. ``speed-up`` is
    causal-learn's median wall time over Andrey's, on the datasets where both finished.
    """
    if table.empty:
        return
    _log("")
    _log(
        f"{'solution':24s} {'d':>3s} {'ok/all':>7s} {'wall_s':>8s} {'shd':>6s} "
        f"{'f1 skel':>7s} {'f1 arrow':>8s}"
    )
    for (name, d), rows in table.groupby(["name", "d"], sort=True):
        ok = rows[rows["status"] == "ok"]
        _log(
            f"{name:24s} {d:3d} {len(ok):3d}/{len(rows):<3d} {ok['wall_s'].median():8.3g} "
            f"{ok['shd'].median():6.3g} {ok['skeleton_f1'].mean():7.3f} "
            f"{ok['arrowhead_f1'].mean():8.3f}"
        )
    _log("")
    _log(f"{'method':10s} {'d':>3s} {'paired':>6s} {'speed-up':>8s}")
    ok = table[table["status"] == "ok"]
    for method, (mine, theirs) in PAIRS.items():
        for d in sorted(ok["d"].unique()):
            a = ok[(ok["name"] == mine.name) & (ok["d"] == d)].set_index("dataset_id")["wall_s"]
            b = ok[(ok["name"] == theirs.name) & (ok["d"] == d)].set_index("dataset_id")["wall_s"]
            both = a.index.intersection(b.index)
            if both.empty:
                continue
            ratio = b[both].median() / a[both].median()
            _log(f"{method:10s} {d:3d} {len(both):6d} {ratio:8.3g}")


if __name__ == "__main__":
    raise SystemExit(main())
