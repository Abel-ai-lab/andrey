#!/usr/bin/env python
"""Benchmark Andrey's CAM-UV against lingam's and causal-learn's on data with hidden variables.

Every solution runs on the same datasets in the same job: Andrey's CAM-UV, lingam's, causal-learn's,
and the empty graph, the floor. The regimes (:data:`REGIMES`), ``d`` counting the hidden nodes:

    latent_anm_er         additive noise, one node in five hidden, d = 5 10 15, 2,000 rows   (in)
    latent_anm_er_tenth   additive noise, one node in ten hidden, d = 11 22 55, 1,000 rows   (in)
    camuv_paper           the CAM-UV paper's simulation, 10 observed + 4 hidden, 1,000 rows  (in)
    anm_er                additive noise, nothing hidden, d = 20, 1,000 rows                 (out)
    latent_gauss_er       linear Gaussian, one node in ten hidden, d = 22, 1,000 rows        (out)

The first three lie in CAM-UV's model class, nonlinear additive effects with hidden variables, and
are in claim (:func:`claim`); the last two lie outside it. Every package runs the paper's settings,
Andrey's defaults (:mod:`~andrey_bench.adapters.andrey_camuv`). A seed whose graph has too few nodes
with two children to hide is skipped, and the next one taken, so a cell has ``--seeds`` datasets.

Every estimate is a PAG, scored against the PAG of the observed nodes (``shd_endpoint``). After
each rung the driver adds, from the estimates stored in ``structures/``: the directed edges against
the edges CAM-UV can identify as direct (``directed_direct_*``), the marked pairs against the pairs
an unobserved path joins (``confounded_unobserved_path_*``), the endpoint SHD with the marked pairs
drawn ``<->`` (``shd_endpoint_bidirected``), which lingam and causal-learn draw and Andrey draws
``o-o``, and ``graph_key``, the directed edges and the marked pairs whatever their mark, for counting
the same graphs. ``--rescore`` recomputes those columns without fitting anything.

**lingam's and causal-learn's fits run in their own interpreter**, the one
``ANDREY_BENCH_LINGAM_PYTHON`` names (:mod:`andrey_bench.adapters.lingam_env`;
``benchmarks/start_lingam.sh`` builds it). Use ``--seed-start`` and ``--regimes`` to cut a campaign
into small runs, each with its own ``--out``; pairs form inside one run, so keep the full solution
list in each. Share the dataset store, ``ANDREY_BENCH_DATA``, and fill it once with
``--prepare-only``.

    ANDREY_BENCH_LINGAM_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/camuv_cpu.py \\
        --out outputs/camuv --regimes latent_anm_er --seeds 20 --cores 1
"""

from __future__ import annotations

import argparse
import hashlib
import os
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from andrey import GraphStructure
from andrey_bench import datasets, memcap, rundir, scoring
from andrey_bench.adapters.andrey_camuv import AndreyCAMUV
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.causal_learn_camuv import causal_learn_camuv_adapter
from andrey_bench.adapters.lingam_camuv import lingam_camuv_adapter
from andrey_bench.adapters.lingam_env import PYTHON_ENV_VAR, lingam_venv
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import run_batch
from andrey_bench.runner import available_cores, check_interpreters

#: The regimes, in ``datasets.REGIMES``: the three in CAM-UV's model class first.
REGIMES = ["latent_anm_er", "latent_anm_er_tenth", "camuv_paper", "anm_er", "latent_gauss_er"]

#: The regimes of CAM-UV's model class, Maeda and Shimizu (2021), Sect. 2: nonlinear additive
#: effects with independent noise, some variables hidden.
CLAIMED = frozenset(REGIMES[:3])

#: Each regime's rows and default ladder; ``d`` counts the hidden nodes.
SETTINGS: dict[str, dict] = {
    "latent_anm_er": {"rows": 2000, "ladder": [5, 10, 15]},
    "latent_anm_er_tenth": {"rows": 1000, "ladder": [11, 22, 55]},
    "camuv_paper": {"rows": 1000, "ladder": [14]},
    "anm_er": {"rows": 1000, "ladder": [20]},
    "latent_gauss_er": {"rows": 1000, "ladder": [22]},
}

#: The observed graph's expected mean degree in the paper's simulation: probability 0.3 over 10
#: observed variables.
PAPER_DENSITY = 2.7

#: Smallest run that touches every solution: one small dataset.
SMOKE = {"regimes": ["latent_anm_er"], "ladder": [5], "seeds": 1, "n": 500, "cap_wall_s": 600.0}

#: Each solution's endpoint mark on a pair it attributes to a hidden variable. The empty graph has
#: no such edges, so either mark stands for it.
CONFOUNDED_MARK = {
    a.name: getattr(a, "confounded_mark", "bidirected")
    for a in (AndreyCAMUV(), lingam_camuv_adapter(), causal_learn_camuv_adapter(), BaselineEmpty())
}

#: The columns :func:`add_scores` adds to each record.
SCORE_COLUMNS = (
    *scoring.CONFOUNDED_COLUMNS,
    *scoring.DIRECTED_COLUMNS,
    "shd_endpoint_bidirected",
    "graph_key",
)


def claim(regime: str) -> str:
    """``"in"`` for a regime of CAM-UV's model class (:data:`CLAIMED`), else ``"out"``."""
    return "in" if regime in CLAIMED else "out"


def solutions() -> list:
    """Andrey's CAM-UV, lingam's, causal-learn's, and the empty graph."""
    return [AndreyCAMUV(), lingam_camuv_adapter(), causal_learn_camuv_adapter(), BaselineEmpty()]


def regime_keys(regime: str, d: int, n: int, seeds: int, seed_start: int = 0) -> list[DatasetKey]:
    """Return `seeds` datasets of `regime` at `d` nodes and `n` rows, from seed `seed_start` on.

    A seed whose graph has fewer nodes with two or more children than nodes to hide
    (:func:`andrey.data.latent.latent_order`), or fewer edges than the paper's simulation hides
    intermediates on, cannot be drawn; it is skipped and the next seed taken.
    """
    topology, functional, noise, standardize = datasets.REGIMES[regime]
    density = PAPER_DENSITY if regime == "camuv_paper" else datasets.CANONICAL_DENSITY
    keys: list[DatasetKey] = []
    seed = seed_start
    while len(keys) < seeds:
        key = DatasetKey(
            topology=topology,
            functional=functional,
            noise=noise,
            standardize=standardize,
            density=density,
            d=d,
            n=n,
            seed=seed,
            latents=datasets.latents_for(regime, d),
        )
        seed += 1
        try:
            datasets.generation_kwargs(key)
            if key.latents:
                datasets.generate_one(key)
        except ValueError as exc:
            if "cannot hide" not in str(exc):
                raise
            continue
        keys.append(key)
    return keys


def as_bidirected(estimate: GraphStructure) -> GraphStructure:
    """`estimate` with every ``o-o`` pair drawn ``<->``, the reference implementations' mark."""
    marks = np.asarray(estimate.to_numpy()).copy()
    pairs = (marks == 3) & (marks.T == 3)
    marks[pairs] = 2
    return GraphStructure.from_numpy(marks, kind=estimate.kind)


def graph_key(estimate: GraphStructure, mark: str) -> str:
    """A hash of the directed edges and the pairs joined by `mark`, the same for either mark."""
    marks = np.asarray(estimate.to_numpy())
    directed = sorted(zip(*map(np.ndarray.tolist, np.nonzero((marks == 1) & (marks.T == 2)))))
    pairs = sorted(scoring._marked_pairs(marks, scoring.CONFOUNDED_MARKS[mark]))
    return hashlib.sha1(f"{len(marks)}:{directed}:{pairs}".encode()).hexdigest()


def add_scores(
    out: Path, records_dir: Path, keys: Mapping[str, DatasetKey], rescore: bool = False
) -> None:
    """Add :data:`SCORE_COLUMNS` to every part file that lacks them.

    `keys` maps each ``dataset_id`` of the run to its :class:`DatasetKey`. Each fit's stored estimate
    is read back and scored with its solution's mark (:data:`CONFOUNDED_MARK`). A fit with no
    estimate has ``NaN`` in every column. The rewrite is atomic, and a part that already has the
    columns is left alone unless `rescore`.
    """
    store = datasets.store_dir(out)
    kinds = datasets.load_registry(store)["keys"]
    truth: tuple[str, GraphStructure] | None = None
    for part in sorted(records_dir.glob("*.parquet")):
        frame = pd.read_parquet(part)
        if not rescore and set(SCORE_COLUMNS) <= set(frame.columns):
            continue
        frame = frame.drop(columns=[c for c in SCORE_COLUMNS if c in frame.columns])
        rows = []
        for record in frame.itertuples():
            estimate = rundir.read_structure(out, record.run_id) if record.status == "ok" else None
            if estimate is None:
                rows.append({})
                continue
            if truth is None or truth[0] != record.dataset_id:
                if record.dataset_id not in keys:
                    raise SystemExit(
                        f"{part.name}: dataset {record.dataset_id} is not in this run's ladder; "
                        "pass the first run's flags"
                    )
                key = keys[record.dataset_id]
                _, marks = datasets.load(store, key)
                truth = (
                    record.dataset_id,
                    GraphStructure.from_numpy(
                        marks,
                        kind=kinds[key.label]["graph_kind"],
                        node_types=datasets.latent_node_types(len(marks), key.latents),
                    ),
                )
            mark = CONFOUNDED_MARK[record.name]
            redrawn = as_bidirected(estimate)
            rows.append(
                scoring.confounded_pair_scores(estimate, truth[1], mark)
                | scoring.directed_edge_scores(estimate, truth[1])
                | {
                    "shd_endpoint_bidirected": scoring.score_result(redrawn, truth[1])[
                        "shd_endpoint"
                    ],
                    "graph_key": graph_key(estimate, mark),
                }
            )
        scores = pd.DataFrame(rows, index=frame.index, columns=list(SCORE_COLUMNS))
        numeric = [c for c in SCORE_COLUMNS if c != "graph_key"]
        scores[numeric] = scores[numeric].astype(float)
        scratch = part.with_name(f".{part.name}.tmp")
        pd.concat([frame, scores], axis=1).to_parquet(scratch, index=False)
        os.replace(scratch, part)


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CAM-UV, Andrey against lingam and causal-learn, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--regimes",
        nargs="+",
        choices=REGIMES,
        default=REGIMES[:3],
        help="regimes to run, in order (default: the three in claim)",
    )
    p.add_argument(
        "--ladder",
        type=int,
        nargs="+",
        help="d sizes, hidden nodes included (default: each regime's)",
    )
    p.add_argument("--n", type=int, default=None, help="rows per dataset (default: each regime's)")
    p.add_argument("--seeds", type=int, default=20, help="datasets per cell (default: 20)")
    p.add_argument("--seed-start", type=int, default=0, help="first data seed (default: 0)")
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="materialize this run's datasets and stop, without measuring anything",
    )
    p.add_argument("--repeats", type=int, default=1, help="timed repeats per fit (default: 1)")
    p.add_argument("--warmup", type=int, default=0, help="untimed fits before each timed one")
    p.add_argument("--cap-wall-s", type=float, default=3600.0, help="wall cap per repeat, seconds")
    p.add_argument("--cap-mem-mb", type=float, default=None, help="memory cap per repeat, MB")
    p.add_argument("--mem-route", choices=memcap.ROUTES, default=None, help="cap enforcement route")
    p.add_argument("--cores", type=int, default=available_cores(), help="core budget per fit")
    p.add_argument("--machine-id", default=socket.gethostname(), help="machine id on every record")
    p.add_argument("--smoke", action="store_true", help="run d=5, one seed, 500 rows")
    p.add_argument("--resume", action="store_true", help="continue the run already in --out")
    p.add_argument(
        "--rescore",
        action="store_true",
        help="fit nothing: recompute the score columns of the run in --out (the first run's flags)",
    )
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)

    out = Path(args.out)
    cells = [
        (regime, d, args.n or SETTINGS[regime]["rows"])
        for regime in args.regimes
        for d in (args.ladder or SETTINGS[regime]["ladder"])
    ]
    cell_keys = {cell: regime_keys(*cell, args.seeds, args.seed_start) for cell in cells}
    keys = {key.digest: key for ks in cell_keys.values() for key in ks}

    if args.prepare_only:
        store = datasets.store_dir(out)
        registry = datasets.materialize(list(keys.values()), store)
        _log(
            f"materialized {len(keys)} dataset(s) into {store}; registry has "
            f"{len(registry['keys'])}"
        )
        return 0

    field = solutions()
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in field],
        "params": {a.name: dict(a.params()) for a in field},
        "regimes": args.regimes,
        "claim": {r: claim(r) for r in args.regimes},
        "cells": [
            {
                "regime": r,
                "d": d,
                "n": n,
                "latents": datasets.latents_for(r, d),
                "seeds": [k.seed for k in cell_keys[(r, d, n)]],
            }
            for r, d, n in cells
        ],
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
        add_scores(out, records_dir, keys, rescore=True)
        table = rundir.assemble(records_dir)
        table.to_parquet(out / "runs.parquet", index=False)
        _log(f"rescored    : {len(table)} rows  -> {out / 'runs.parquet'}")
        _summarize(table)
        return 0

    check_interpreters(field)
    records_dir = rundir.start_run(out, run_meta, resume=args.resume)
    prov = rundir.provenance_once(
        out,
        timestamp=datetime.now(timezone.utc).isoformat(),
        extra_venvs={"lingam": lingam_venv()},
    )
    _log(f"=== CAM-UV, Andrey against lingam and causal-learn -> {out} ===")
    for name, value in run_meta.items():
        _log(f"{name:14s}: {value}")
    _log(f"{'lingam python':14s}: {os.environ.get(PYTHON_ENV_VAR, '')}")
    _log(f"{'git_sha':14s}: {prov['git_sha']}")
    _log("")

    errors: list[str] = []
    start = time.perf_counter()
    for cell in cells:
        regime, d, n = cell
        result = run_batch(
            datasets.store_dir(out),
            field,
            cell_keys[cell],
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
        add_scores(out, records_dir, keys)
        cost = []
        for adapter in field:
            mine = [r for r in result.records if r.name == adapter.name]
            cpus = [r.cpu_s for r in mine if r.status == "ok" and r.cpu_s is not None]
            statuses = "/".join(sorted({r.status for r in mine})) or "no-records"
            cost.append(f"{adapter.name}={statistics.median(cpus):.3g}s" if cpus else statuses)
        _log(
            f"[{regime} d={d} n={n}] {'  '.join(cost)} | elapsed {time.perf_counter() - start:.0f}s"
        )

    add_scores(out, records_dir, keys)
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
    _summarize(table)
    return 1 if errors or len(failed) else 0


def _summarize(table: pd.DataFrame) -> None:
    """Print, per cell and solution, the fits that finished, their median CPU time, and their means.

    ``direct`` is the mean directed-edge errors, ``endpoint`` and ``as <->`` the mean endpoint SHD
    as returned and with the marked pairs drawn ``<->``, ``P / R`` the marked pairs' precision and
    recall pooled over the cell against the unobserved-path pairs, and ``same`` the datasets where a
    reference returned Andrey's graph, whatever the mark, of those both finished.
    """
    if table.empty:
        return
    _log("")
    _log(
        f"{'functional':10s} {'d':>3s} {'n':>5s} {'solution':22s} {'ok/all':>7s} {'cpu_s':>7s} "
        f"{'direct':>7s} {'endpoint':>8s} {'as <->':>7s} {'pairs':>6s} {'P / R':>11s} {'same':>6s}"
    )
    for (functional, d, n, name), rows in table.groupby(["functional", "d", "n", "name"]):
        ok = rows[rows.status == "ok"]
        claimed = ok["confounded_n_est"].sum()
        hits = (ok["confounded_unobserved_path_precision"].fillna(0) * ok["confounded_n_est"]).sum()
        true = ok["confounded_unobserved_path_n_true"].sum()
        pr = f"{hits / claimed if claimed else float('nan'):.2f} / " + (
            f"{hits / true if true else float('nan'):.2f}"
        )
        same = ""
        andrey = table[
            (table.package == "andrey")
            & (table.functional == functional)
            & (table.d == d)
            & (table.n == n)
        ]
        if name not in set(andrey.name) and len(andrey):
            mine = andrey[andrey.status == "ok"].groupby("dataset_id").graph_key.first()
            theirs = ok.groupby("dataset_id").graph_key.first()
            both = mine.index.intersection(theirs.index)
            same = f"{int((mine[both] == theirs[both]).sum())}/{len(both)}"
        _log(
            f"{functional:10s} {d:3d} {n:5d} {name:22s} {len(ok):3d}/{len(rows):<3d} "
            f"{ok['cpu_s'].median():7.3g} {ok['directed_direct_errors'].mean():7.2f} "
            f"{ok['shd_endpoint'].mean():8.2f} {ok['shd_endpoint_bidirected'].mean():7.2f} "
            f"{ok['confounded_n_est'].sum():6.0f} {pr:>11s} {same:>6s}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
