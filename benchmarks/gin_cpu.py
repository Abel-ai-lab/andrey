#!/usr/bin/env python
"""Benchmark GIN on CPU: Andrey against Tetrad and causal-learn, on latents and their indicators.

Runs on the ``latent_measurement`` regime (:mod:`andrey_bench.regime_models`): ``m`` latents, each a
cause of every later one, each measured by three observed indicators of its own, linear with
uniform noise. ``--ladder`` gives ``m``; a dataset's ``d`` counts the latents and their indicators,
so its data has ``3 m`` columns. Andrey and causal-learn run gamma-HSIC and Tetrad its KCI, a
gamma-approximation kernel test of the same family, all at ``alpha = 0.05``, listed in ``params``. GIN returns a cluster of indicators per latent and an order of the latents; the driver
scores both against the truth after each rung (:func:`andrey_bench.scoring.measurement_scores`):
the clustering's adjusted Rand index, the share of true clusters found exactly, and the share of
pairs of found latents put in the true order. The empty graph, which finds no cluster, is the
quality baseline. Every solution is serial.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/gin_cpu.py \\
        --out outputs/gin-cpu --ladder 2 3 4 --cap-mem-mb 32000
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
from andrey_bench import datasets, memcap, pairing, rundir, scoring
from andrey_bench.adapters.andrey_gin import AndreyGIN
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.causal_learn_gin import CausalLearnGIN
from andrey_bench.adapters.tetrad_gin import TetradGIN
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.regime_models import MODELS
from andrey_bench.runner import available_cores

#: Each package's GIN and the empty-graph baseline.
SOLUTIONS = [AndreyGIN(), TetradGIN(), CausalLearnGIN(), BaselineEmpty()]

REGIME = "latent_measurement"

#: Observed indicators per latent.
INDICATORS = MODELS["measurement"].indicators

#: Smallest run that touches every solution: one small dataset.
SMOKE = {"ladder": [2], "seeds": 1, "n": 500, "repeats": 1, "cap_wall_s": 300.0}

#: Untimed fits before each measurement. A GIN fit runs every HSIC test on every group of
#: indicators, so a warm-up fit would double its cost against the wall cap.
WARMUP = 0


def regime_slice() -> dict:
    """Return the :class:`DatasetKey` coordinates of :data:`REGIME`."""
    topology, functional, noise, standardize = datasets.REGIMES[REGIME]
    return {
        "topology": topology,
        "functional": functional,
        "noise": noise,
        "standardize": standardize,
        "density": float(INDICATORS),
    }


def ladder_keys(ladder: list[int], seeds: int, n: int) -> list[DatasetKey]:
    """Return one dataset of `n` rows per ``(m, seed)``, for `m` latents in `ladder`."""
    keys = []
    for m in ladder:
        d = m * (INDICATORS + 1)
        for seed in range(seeds):
            keys.append(DatasetKey(**regime_slice(), d=d, n=n, seed=seed, latents=m))
    return keys


def _log(msg: str = "") -> None:
    """Print and flush batch output immediately."""
    print(msg, flush=True)


def add_measurement_scores(
    out: Path, records_dir: Path, keys: Mapping[str, DatasetKey], rescore: bool = False
) -> None:
    """Add the :data:`scoring.MEASUREMENT_COLUMNS` to every part file that lacks them.

    `keys` maps each ``dataset_id`` of the run to its :class:`DatasetKey`. Each fit's stored estimate
    is read back and scored with :func:`scoring.measurement_scores`; a fit with no estimate has
    ``NaN`` in every column. The rewrite is atomic, and a part that already has the columns is left
    alone unless `rescore` is set.
    """
    store = datasets.store_dir(out)
    kinds = datasets.load_registry(store)["keys"]
    for part in sorted(records_dir.glob("*.parquet")):
        frame = pd.read_parquet(part)
        if not rescore and set(scoring.MEASUREMENT_COLUMNS) <= set(frame.columns):
            continue
        frame = frame.drop(columns=[c for c in frame.columns if c.startswith("measurement_")])
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
            key = keys[record.dataset_id]
            _, marks = datasets.load(store, key)
            truth = GraphStructure.from_numpy(
                marks,
                kind=kinds[key.label]["graph_kind"],
                node_types=datasets.latent_node_types(len(marks), key.latents),
            )
            rows.append(scoring.measurement_scores(estimate, truth))
        scores = pd.DataFrame(
            rows, index=frame.index, columns=list(scoring.MEASUREMENT_COLUMNS)
        ).astype(float)
        scratch = part.with_name(f".{part.name}.tmp")
        pd.concat([frame, scores], axis=1).to_parquet(scratch, index=False)
        os.replace(scratch, part)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GIN, Andrey against causal-learn, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--ladder",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="latent counts, each with its indicators (default: 2 3 4)",
    )
    p.add_argument("--seeds", type=int, default=5, help="datasets per rung (default: 5)")
    p.add_argument("--n", type=int, default=1000, help="rows per dataset (default: 1000)")
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
        help="run two latents, one seed, 500 rows, one repeat, and a 300 s cap, replacing those "
        "flags",
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
        help="fit nothing: recompute the measurement columns of the run already in --out and "
        "rewrite runs.parquet. Takes the first run's flags, as --resume does",
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
        "params": {a.name: dict(a.params()) for a in SOLUTIONS},
        "slice": regime_slice(),
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
        records_dir = out / rundir.RECORDS_DIRNAME
        if not any(records_dir.glob("*.parquet")):
            raise SystemExit(f"--rescore: {records_dir} holds no records")
        add_measurement_scores(out, records_dir, keys, rescore=True)
        table = rundir.assemble(records_dir)
        table.to_parquet(out / "runs.parquet", index=False)
        _log(f"rescored    : {len(table)} rows  -> {out / 'runs.parquet'}")
        summarize(table)
        return 0
    # Reads run_meta.json before writing it, so --resume has something to check against.
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())

    _log(f"=== GIN, Andrey and causal-learn -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:12s}: {value}")
    _log(f"{'git_sha':12s}: {prov['git_sha']}")
    _log("")

    records: list = []
    errors: list[str] = []
    skipped = 0
    start = time.perf_counter()

    for m in args.ladder:
        # One rung at a time, so a rung's data is materialized only when the ladder reaches it.
        result = run_batch(
            datasets.store_dir(out),
            SOLUTIONS,
            ladder_keys([m], args.seeds, args.n),
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
        _log(f"[m={m}] {'  '.join(ran)} | elapsed {time.perf_counter() - start:.0f}s")
        add_measurement_scores(out, records_dir, keys)

    # From the parts, not from memory: a resume that measured nothing still writes the table for
    # everything measured before it.
    add_measurement_scores(out, records_dir, keys)
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
    """Print, per solution and latent count, the finished fits with their medians and means.

    ``ari`` is the mean adjusted Rand index of the clustering, ``found`` the mean share of true
    clusters found exactly, and ``order`` the mean share of pairs of found latents in the true
    order, over the fits where one is defined. ``same`` counts the datasets where causal-learn
    returned Andrey's clusters and order, of those both finished.
    """
    if table.empty:
        return
    _log("")
    _log(
        f"{'solution':20s} {'m':>3s} {'ok/all':>7s} {'wall_s':>8s} {'latents':>7s} {'ari':>6s} "
        f"{'found':>6s} {'order':>6s} {'same':>6s}"
    )
    for (name, latents), rows in table.groupby(["name", "latents"], sort=True):
        ok = rows[rows["status"] == "ok"]
        same = ""
        if name == CausalLearnGIN.name:
            andrey = table[(table["name"] == AndreyGIN.name) & (table["latents"] == latents)]
            same = "{}/{}".format(*pairing.same_graph(andrey, rows))
        _log(
            f"{name:20s} {latents:3d} {len(ok):3d}/{len(rows):<3d} {ok['wall_s'].median():8.3g} "
            f"{ok['measurement_n_latents'].mean():7.2f} {ok['measurement_cluster_ari'].mean():6.3f} "
            f"{ok['measurement_clusters_found'].mean():6.3f} "
            f"{ok['measurement_order_accuracy'].mean():6.3f} {same:>6s}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
