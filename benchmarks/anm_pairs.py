#!/usr/bin/env python
"""Benchmark the ANM direction test on CPU: Andrey against causal-learn.

Runs Andrey's ``anm`` and causal-learn's ``ANM`` on cause-effect pairs from the ``anm_pair``
regime: two nodes, one edge in a direction the seed draws, a nonlinear additive-noise mechanism, and
Gaussian noise. The ladder runs over the number of rows, since a pair always has two columns. Both
solutions decide the edge with the same rule, the direction with the larger p-value, so a pair gets
an edge whenever the p-values differ. On equal p-values, Andrey takes the direction with the smaller
HSIC statistic relative to its mean under independence; causal-learn reports no statistic, so a tie
gives it no edge. A timeout is a timeout: it has no scores.

Each fit is scored by whether it names the true direction (:func:`direction_scores`), on top of the
standard scores ``run_batch`` records.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/anm_pairs.py \\
        --out outputs/anm-pairs --cap-mem-mb 32000
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
from andrey_bench import datasets, memcap, rundir
from andrey_bench.adapters.andrey_anm import AndreyANM
from andrey_bench.adapters.causal_learn_anm import CausalLearnANM
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores

REGIME = "anm_pair"
LANE = [AndreyANM(), CausalLearnANM()]

#: Mean total degree of a pair: at ``d = 2`` it gives every pair exactly one edge.
PAIR_DENSITY = 1.0

#: Smallest run that touches every solution: one small pair.
SMOKE = {"sizes": [100], "seeds": 1, "cap_wall_s": 300.0}

#: Untimed fits before each measurement. A fit at 2000 rows takes about half a minute, so the wall
#: cap budget goes to measurements.
WARMUP = 0


def pair_keys(sizes: list[int], seeds: int) -> list[DatasetKey]:
    """Return one ``anm_pair`` dataset per ``(n, seed)``."""
    topology, functional, noise, standardize = datasets.REGIMES[REGIME]
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
        for n in sizes
        for seed in range(seeds)
    ]


def _log(msg: str = "") -> None:
    """Print and flush batch output immediately."""
    print(msg, flush=True)


#: The columns :func:`direction_scores` returns: 1.0 or 0.0 each, exactly one of them 1.0.
SCORE_COLUMNS = ("direction_correct", "direction_reversed", "direction_none")


def direction_scores(estimate: GraphStructure, truth: GraphStructure) -> dict[str, float]:
    """Score a pair's estimate: the true edge, the reversed edge, or no edge.

    `truth` holds exactly one directed edge; `estimate` holds at most one.
    """
    [(source, target, _)] = truth.oriented_edges()
    edges = [(int(i), int(j)) for i, j, _ in estimate.oriented_edges()]
    return {
        "direction_correct": float(edges == [(source, target)]),
        "direction_reversed": float(edges == [(target, source)]),
        "direction_none": float(not edges),
    }


def add_direction_scores(
    out: Path, records_dir: Path, keys: Mapping[str, DatasetKey], rescore: bool = False
) -> None:
    """Add the :data:`SCORE_COLUMNS` to every part file that lacks them.

    `keys` maps each ``dataset_id`` of the run to its :class:`DatasetKey`. Each fit's stored estimate
    is read back and scored with :func:`direction_scores`; a fit with no estimate has ``NaN`` in
    every column. The rewrite is atomic, and a part that already has the columns is left alone, so a
    resumed run scores only the units it measured; `rescore` replaces every part's columns instead.
    """
    store = datasets.store_dir(out)
    for part in sorted(records_dir.glob("*.parquet")):
        frame = pd.read_parquet(part)
        if not rescore and set(SCORE_COLUMNS) <= set(frame.columns):
            continue
        frame = frame.drop(columns=[c for c in frame.columns if c.startswith("direction_")])
        rows = []
        for record in frame.itertuples():
            estimate = rundir.read_structure(out, record.run_id) if record.status == "ok" else None
            if estimate is None:
                rows.append({})
                continue
            if record.dataset_id not in keys:
                raise SystemExit(
                    f"{part.name}: dataset {record.dataset_id} is not in this run's ladder; "
                    "pass the first run's flags"
                )
            _, marks = datasets.load(store, keys[record.dataset_id])
            rows.append(direction_scores(estimate, GraphStructure.from_numpy(marks, kind="dag")))
        scores = pd.DataFrame(rows, index=frame.index, columns=list(SCORE_COLUMNS)).astype(float)
        scratch = part.with_name(f".{part.name}.tmp")
        pd.concat([frame, scores], axis=1).to_parquet(scratch, index=False)
        os.replace(scratch, part)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ANM direction test, Andrey against causal-learn, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[200, 500, 1000, 2000],
        help="rows per pair (default: 200 500 1000 2000)",
    )
    p.add_argument("--seeds", type=int, default=20, help="pairs per size (default: 20)")
    p.add_argument("--repeats", type=int, default=1, help="timed repeats per fit (default: 1)")
    p.add_argument(
        "--warmup",
        type=int,
        default=WARMUP,
        help="untimed fits per repeat; the wall cap covers them too (default: 0)",
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
        help="run one pair of 100 rows with a 300 s cap, replacing those flags",
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
        help="fit nothing: recompute the direction columns of the run already in --out and "
        "rewrite runs.parquet. Takes the first run's flags, as --resume does",
    )
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)

    out = Path(args.out)
    keys = {key.digest: key for key in pair_keys(args.sizes, args.seeds)}

    # Written before the first fit, so an interrupted benchmark still says what it was asked for.
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in LANE],
        "regime": REGIME,
        "density": PAIR_DENSITY,
        "ladder": args.sizes,
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
        add_direction_scores(out, records_dir, keys, rescore=True)
        table = rundir.assemble(records_dir)
        table.to_parquet(out / "runs.parquet", index=False)
        _log(f"rescored    : {len(table)} rows  -> {out / 'runs.parquet'}")
        _summarize(table)
        return 0
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())

    _log(f"=== ANM direction test, Andrey and causal-learn -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:12s}: {value}")
    _log(f"{'git_sha':12s}: {prov['git_sha']}")
    _log("")

    records: list = []
    errors: list[str] = []
    skipped = 0
    start = time.perf_counter()

    for n in args.sizes:
        # One size at a time, so its pairs are materialized only when the ladder reaches it.
        result = run_batch(
            datasets.store_dir(out),
            LANE,
            pair_keys([n], args.seeds),
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

        # What each solution cost at this size, or what stopped it.
        ran = []
        for adapter in LANE:
            mine = [r for r in result.records if r.name == adapter.name]
            walls = [r.wall_s for r in mine if r.status == "ok" and r.wall_s is not None]
            statuses = sorted({r.status for r in mine})
            cost = (
                f"{statistics.median(walls):.3g}s" if walls else "/".join(statuses) or "no-records"
            )
            ran.append(f"{adapter.name}={cost}")
        _log(f"[n={n}] {'  '.join(ran)} | elapsed {time.perf_counter() - start:.0f}s")
        add_direction_scores(out, records_dir, keys)

    # From the parts, not from memory: a resume that measured nothing still writes the table for
    # everything measured before it.
    add_direction_scores(out, records_dir, keys)
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
    """Print, per size and solution, the finished fits, their median time, and their accuracy.

    ``accuracy`` is the share of finished fits that name the true direction, ``reversed`` and
    ``none`` the shares that name the other direction or no edge. ``agree`` is the share of pairs,
    among those both solutions finished, on which they return the same graph.
    """
    if table.empty:
        return
    _log("")
    _log(
        f"{'n':>5s} {'solution':22s} {'ok/all':>7s} {'wall_s':>8s} {'accuracy':>8s} "
        f"{'reversed':>8s} {'none':>6s} {'agree':>6s}"
    )
    for n, size in table.groupby("n", sort=True):
        ok = size[size["status"] == "ok"]
        hashes = ok.pivot_table(
            index="dataset_id", columns="name", values="structure_hash", aggfunc="first"
        ).dropna()
        agree = (hashes.nunique(axis=1) == 1).mean() if len(hashes.columns) > 1 else float("nan")
        for name, rows in size.groupby("name", sort=True):
            mine = rows[rows["status"] == "ok"]
            _log(
                f"{n:5d} {name:22s} {len(mine):3d}/{len(rows):<3d} {mine['wall_s'].median():8.3g} "
                f"{mine['direction_correct'].mean():8.3f} {mine['direction_reversed'].mean():8.3f} "
                f"{mine['direction_none'].mean():6.3f} {agree:6.3f}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
