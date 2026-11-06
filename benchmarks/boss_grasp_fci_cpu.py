#!/usr/bin/env python
"""Benchmark BOSS-FCI and GRaSP-FCI on CPU: Andrey against Tetrad, on data with hidden variables.

Every dataset comes from ``andrey.data`` (:func:`dataset`) as Ramsey, Andrews, and Spirtes (2025,
Sec. 8.1.1) simulate theirs with Tetrad's ``RandomForward`` and ``SemSimulation``: a DAG over
``d + L`` nodes in a random forward order with exactly ``degree * d / 2`` edges drawn uniformly,
linear effects with coefficients uniform on ``[-1, 1]``, Gaussian noise with variances uniform on
``[1, 3]``, the generator's own units (``scale="raw"``), and the columns in an order drawn from
the seed. ``L`` nodes are hidden, drawn among the nodes with two or more children; when fewer have
two children, only those are hidden, as Tetrad does, and ``latents_hidden`` records the number.
:func:`grid` lists the cells:

- **P20**, in claim: 20 measured variables, mean degree 2, 4, or 6, 0, 4, or 8 hidden, ``n`` 200,
  500, 1,000, or 5,000, at the paper's settings (penalty discount 2.0, alpha 0.01) and at Andrey's
  defaults (1.0, 0.05), both packages at the same values: 72 rows.
- **P100**, in claim: 100 measured, degree 6, 10 hidden, ``n`` 1,000, conditioning sets of at most
  7 (Tetrad's ``depth``, Andrey's ``max_depth``), the paper's settings.
- Out of claim: **O1** as P100 at degree 10 (10 seeds); **O2**, 200 measured, degree 6, 20 hidden,
  depth 7 (one seed, one reference run, time only); **O3**, ``sample_scm(d=24, latents=4,
  density=4, n=1000)``, the recovery test's setting, at both settings (10 seeds).

The arms (:func:`andrey_fits`, :func:`tetrad_fits`): Andrey's ``boss_fci`` and ``grasp_fci``
against the gated reference, causal-cmd 1.12.0's ``bfci`` and ``grasp-fci`` (tetrad-lib 7.6.3) at
Tetrad's documented defaults (:mod:`~andrey_bench.adapters.tetrad_pag`), run :data:`DRAWS` times
per dataset with no seed. At the paper's settings, P20 adds context arms: tetrad-lib 7.6.10's
``BossFci`` and ``GraspFci``, the paper's code, once (also on P100); causal-cmd's GFCI and FCI and
Andrey's ``gfci`` and ``fci``; Andrey's hybrids with the possible-d-separation step's removals
skipped (``no_possible_dsep``, the step neither Tetrad hybrid has); and ``grasp_fci`` with its tuck
depth at ``d`` (Tetrad's GRaSP has none). P100 and O1 add ``grasp_fci`` at tuck depth 7.

The truth is the PAG of the measured variables (``andrey.data.latent.marginal``); ``check-truth``
compares it with pcalg's ``dag2pag`` on the DAG numbered in a topological order. Every estimate is
scored against it (:func:`pag_scores`): the endpoint SHD (``shd``, the verdict), the adjacency,
arrowhead, and tail precision and recall (arrowhead and tail F1 too, and the arrowheads on the
adjacencies both graphs share), the precision of the paper's path metrics, and the empty graph's
SHD. Each Andrey fit also records the largest possible-d-separation candidate set its step searched.

Per dataset, each package runs in one child process pinned to one core with one BLAS thread:
Andrey's in :mod:`~andrey_bench.adapters.andrey_boss_fci`, Tetrad's in this file
(``python boss_grasp_fci_cpu.py <request.npz> <out.jsonl>``), which starts one JVM per run with one
untimed warm-up search before the timed one. ``run`` writes ``records.jsonl``, ``datasets.jsonl``,
``run_meta.json``, ``provenance.json``, and ``done.json``; ``summarize`` the paired comparisons.

    source the reference environments (ANDREY_BENCH_JAVA, ANDREY_BENCH_TETRAD_JAR,
    ANDREY_BENCH_TETRAD_GIN_JAR), then
    $ANDREY_BENCH_PYTHON benchmarks/boss_grasp_fci_cpu.py run --out outputs/bgf-P20 --blocks P20 \\
        --workers 4
    $ANDREY_BENCH_PYTHON benchmarks/boss_grasp_fci_cpu.py check-truth --runs outputs/bgf-P20 \\
        --out outputs/bgf-truth
    $ANDREY_BENCH_PYTHON benchmarks/boss_grasp_fci_cpu.py summarize --runs outputs/bgf-* \\
        --out outputs/bgf-summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ut_igsp_cpu import compare, floor_interval, parse_seeds, replication, run_child, same_run

HERE = Path(__file__).resolve().parent
ANDREY_CHILD = HERE / "andrey_bench" / "adapters" / "andrey_boss_fci.py"
RECORDS, DATASETS, DONE = "records.jsonl", "datasets.jsonl", "done.json"

#: The two settings, both packages at the same values.
SETTINGS = {
    "paper": {"alpha": 0.01, "lambda_value": 2.0},
    "default": {"alpha": 0.05, "lambda_value": 1.0},
}
#: Runs of the gated reference per dataset; its error is their mean.
DRAWS = 5
#: The packages a run may fit.
PACKAGES = ("andrey", "tetrad-7.6.3", "tetrad-7.6.10")
#: The gated reference solution of each hybrid, causal-cmd 1.12.0's.
GATED = {"boss_fci": "tetrad.cmd.boss_fci.bic", "grasp_fci": "tetrad.cmd.grasp_fci.bic"}


@dataclass(frozen=True)
class Cell:
    """One cell of the grid: a data setting, the packages' setting, and its seeds."""

    block: str
    setting: str
    d: int  # measured variables when all ``latents`` are hidden
    degree: int  # mean degree over the measured variables (``sample_scm``'s density for O3)
    latents: int
    n: int
    depth: int = -1
    claim: str = "in"
    seeds: tuple[int, ...] = tuple(range(20))
    cap_draw_s: float = 1800.0
    generator: str = "forward"

    @property
    def regime(self) -> str:
        """The cell's group over ``n``: ``P20|deg=4|L=8|setting=paper``."""
        return f"{self.block}|deg={self.degree}|L={self.latents}|setting={self.setting}"

    @property
    def name(self) -> str:
        return f"{self.regime}|n={self.n}"


def grid() -> list[Cell]:
    """Every cell: P20 and P100 in claim, O1-O3 out of claim (module docstring)."""
    cells = [
        Cell("P20", setting, 20, degree, latents, n)
        for setting in SETTINGS
        for degree in (2, 4, 6)
        for latents in (0, 4, 8)
        for n in (200, 500, 1000, 5000)
    ]
    cells.append(Cell("P100", "paper", 100, 6, 10, 1000, depth=7, cap_draw_s=3600.0))
    out = {"claim": "out", "cap_draw_s": 3600.0}
    cells.append(Cell("O1", "paper", 100, 10, 10, 1000, depth=7, seeds=tuple(range(10)), **out))
    cells.append(Cell("O2", "paper", 200, 6, 20, 1000, depth=7, seeds=(0,), **out))
    cells += [
        Cell("O3", setting, 20, 4, 4, 1000, seeds=tuple(range(10)), generator="sample_scm", **out)
        for setting in SETTINGS
    ]
    return cells


def cell_named(name: str) -> Cell:
    for cell in grid():
        if cell.name == name:
            return cell
    raise KeyError(name)


# ------------------------------------------------------------------------------------------------
# data and truth
# ------------------------------------------------------------------------------------------------


def _forward(cell: Cell, hidden: int):
    """The paper's SEM on ``cell.d + cell.latents`` nodes, ``hidden`` of them hidden."""
    import andrey.data as data

    total = cell.d + cell.latents
    model = {
        "graph": data.graphs.erdos_renyi(d=total, avg_degree=cell.degree * cell.d / total),
        "functional": data.functional.linear(weight_range=(0.0, 1.0)),
        "noise": data.noise.gaussian(variance_range=(1.0, 3.0)),
    }
    return data.SCM(**model, latents=hidden) if hidden else data.SCM(**model)


def arcs_of(marks: np.ndarray) -> np.ndarray:
    """The ``(u, v)`` pairs of a DAG's endpoint marks with ``u -> v``."""
    from andrey.core import ARROW, TAIL

    marks = np.asarray(marks)
    return np.argwhere((marks == TAIL) & (marks.T == ARROW))


def _degrees(arcs: np.ndarray, nodes: int) -> list[tuple[int, int]]:
    """The sorted (in, out) degree pairs: the same for a graph under any node labels."""
    into = np.bincount(arcs[:, 1], minlength=nodes)
    out = np.bincount(arcs[:, 0], minlength=nodes)
    return sorted(zip(into.tolist(), out.tolist()))


def dataset(cell: Cell, seed: int) -> tuple[np.ndarray, dict]:
    """One dataset of ``cell``: the measured columns and the truth.

    The truth holds the DAG over every node (``arcs``, the measured nodes first, in the data's
    column order, then the hidden ones), ``observed``, ``latents_hidden``, ``candidates`` (the nodes
    with two or more children of the draw with nothing hidden), the PAG of the measured variables
    (``pag``), and the empty graph's SHD against it. Raises ``ValueError`` when the forward graph
    does not have ``degree * d / 2`` edges or changes when nodes are hidden.
    """
    import andrey.data as data
    from andrey.core import GraphStructure
    from andrey.data.latent import marginal
    from andrey.metrics import shd

    candidates = None
    if cell.generator == "sample_scm":
        ds = data.sample_scm(
            d=cell.d + cell.latents,
            n=cell.n,
            seed=seed,
            density=float(cell.degree),
            latents=cell.latents,
        )
        hidden = cell.latents
    else:
        total = cell.d + cell.latents
        plain = _forward(cell, 0).sample(n=cell.n, seed=seed, scale="raw")
        drawn = arcs_of(plain.graph.to_numpy())
        if len(drawn) != cell.degree * cell.d // 2:
            raise ValueError(f"{cell.name} seed {seed}: {len(drawn)} edges, not degree * d / 2")
        candidates = int((np.bincount(drawn[:, 0], minlength=total) >= 2).sum())
        hidden = min(cell.latents, candidates)
        ds = _forward(cell, hidden).sample(n=cell.n, seed=seed, scale="raw") if hidden else plain
        if _degrees(arcs_of(ds.graph.to_numpy()), total) != _degrees(drawn, total):
            raise ValueError(f"{cell.name} seed {seed}: hiding nodes changed the graph")
    full = ds.graph.to_numpy()
    kinds = ds.graph.node_types  # None when nothing is hidden
    types = np.zeros(len(full), dtype=int) if kinds is None else np.asarray(kinds)
    observed = int((types == 0).sum())
    X = np.ascontiguousarray(ds.data, dtype=np.float64)
    if X.shape[1] != observed or (types[:observed] != 0).any():
        raise ValueError(f"{cell.name} seed {seed}: the measured nodes are not the first columns")
    pag = marginal(ds.graph, "pag").to_numpy().astype(int)
    empty = GraphStructure.from_numpy(np.zeros_like(pag, dtype=np.int8), kind="pag")
    truth = {
        "nodes": int(full.shape[0]),
        "observed": observed,
        "latents_hidden": int(hidden),
        "candidates": candidates,
        "arcs": arcs_of(full).tolist(),
        "pag": pag.tolist(),
        "empty_shd": int(
            shd(empty, GraphStructure.from_numpy(pag, kind="pag"), endpoint_aware=True)
        ),
    }
    return X, truth


# ------------------------------------------------------------------------------------------------
# scores
# ------------------------------------------------------------------------------------------------


def _ratio(hits: int, total: int) -> float | None:
    return hits / total if total else None


def _rates(found: set, true: set, prefix: str) -> dict[str, float | None]:
    """Precision, recall, and F1 of ``found`` against ``true``; ``None`` where undefined."""
    hits = len(found & true)
    return {
        f"{prefix}_precision": _ratio(hits, len(found)),
        f"{prefix}_recall": _ratio(hits, len(true)),
        f"{prefix}_f1": _ratio(2 * hits, len(found) + len(true)),
    }


def _reach(arcs: np.ndarray, nodes: int) -> np.ndarray:
    """``reach[u, v]``: a directed path of one or more edges runs from ``u`` to ``v``."""
    step = np.zeros((nodes, nodes), dtype=bool)
    if len(arcs):
        step[arcs[:, 0], arcs[:, 1]] = True
    reach = step.copy()
    while True:
        wider = reach | ((reach.astype(np.int64) @ step.astype(np.int64)) > 0)
        if (wider == reach).all():
            return reach
        reach = wider


def pag_scores(marks, truth: dict) -> dict[str, float | int | None]:
    """Score an estimated PAG's endpoint marks against the dataset's ``truth``.

    ``marks[i, j]`` is the mark at ``i`` on edge ``i-j``. ``shd`` is the endpoint SHD against the
    true PAG. ``adjacency_*`` read the pairs joined; ``arrowhead_*`` and ``tail_*`` the endpoints
    with that mark, as ordered pairs; ``arrowhead_common_*`` the arrowheads on pairs both graphs
    join. The paper's path metrics read the DAG: ``arrow_path_precision``, the share of estimated
    ``x *-> y`` with no directed path from ``y`` to ``x``; ``tail_path_precision``, of ``x -> y``
    with a directed path from ``x`` to ``y``; ``bidirected_latent_precision``, of ``x <-> y`` with
    a hidden node that is a parent of both.
    """
    from andrey.core import ARROW, TAIL, GraphStructure
    from andrey.metrics import shd

    est, pag = np.asarray(marks), np.asarray(truth["pag"])
    shd_value = shd(
        GraphStructure.from_numpy(est, kind="pag"),
        GraphStructure.from_numpy(pag, kind="pag"),
        endpoint_aware=True,
    )
    out: dict[str, float | int | None] = {"shd": int(shd_value), "empty_shd": truth["empty_shd"]}
    joined_est = (est != 0) | (est.T != 0)
    joined_true = (pag != 0) | (pag.T != 0)
    pairs_est = {(int(i), int(j)) for i, j in np.argwhere(np.triu(joined_est, 1))}
    pairs_true = {(int(i), int(j)) for i, j in np.argwhere(np.triu(joined_true, 1))}
    out["edges"] = len(pairs_est)
    rates = _rates(pairs_est, pairs_true, "adjacency")
    out |= {k: v for k, v in rates.items() if not k.endswith("_f1")}
    for name, mark in (("arrowhead", ARROW), ("tail", TAIL)):
        found = {(int(i), int(j)) for i, j in np.argwhere(est == mark)}
        true = {(int(i), int(j)) for i, j in np.argwhere(pag == mark)}
        out |= _rates(found, true, name)
        if name == "arrowhead":
            both = joined_est & joined_true
            common = _rates(
                {p for p in found if both[p]}, {p for p in true if both[p]}, "arrowhead_common"
            )
            out |= {k: v for k, v in common.items() if not k.endswith("_f1")}
    arcs = np.asarray(truth["arcs"], dtype=np.int64).reshape(-1, 2)
    reach = _reach(arcs, truth["nodes"])
    into = [(int(x), int(y)) for y, x in np.argwhere(est == ARROW)]  # x *-> y
    out["arrow_path_precision"] = _ratio(sum(not reach[y, x] for x, y in into), len(into))
    directed = [(x, y) for x, y in into if est[x, y] == TAIL]
    out["tail_path_precision"] = _ratio(sum(bool(reach[x, y]) for x, y in directed), len(directed))
    observed = truth["observed"]
    parents = [set() for _ in range(truth["nodes"])]
    for u, v in arcs:
        parents[v].add(int(u))
    both_ways = [(x, y) for x, y in into if x < y and est[x, y] == ARROW]
    shared = [
        any(u >= observed for u in parents[x] & parents[y]) for x, y in both_ways
    ]  # x <-> y with a hidden common parent
    out["bidirected_latent_precision"] = _ratio(sum(shared), len(both_ways))
    return out


# ------------------------------------------------------------------------------------------------
# arms
# ------------------------------------------------------------------------------------------------


def _params(cell: Cell) -> dict:
    """The settings both packages run in ``cell``, in the harness's names."""
    return {
        **SETTINGS[cell.setting],
        "depth": cell.depth,
        "indep_test": "fisherz",
        "score_func": "local_score_BIC",
    }


def andrey_fits(cell: Cell) -> list[dict]:
    """Andrey's fits on each dataset of ``cell``, one draw each (seed 0)."""
    params = _params(cell)

    def fit(algorithm, config=cell.setting, trace=None, **extra):
        return {
            "algorithm": algorithm,
            "config": config,
            "params": {**params, **extra},
            "trace": trace,
            "draws": [0],
        }

    fits = [fit("boss_fci"), fit("grasp_fci")]
    if cell.block == "P20" and cell.setting == "paper":
        fits += [
            fit("boss_fci", "paper-no-possible-dsep", "no_possible_dsep"),
            fit("grasp_fci", "paper-no-possible-dsep", "no_possible_dsep"),
            fit("grasp_fci", f"paper-grasp-depth-{cell.d}", grasp_depth=cell.d),
            fit("gfci"),
            fit("fci"),
        ]
    if cell.block in ("P100", "O1"):
        fits.append(fit("grasp_fci", "paper-grasp-depth-7", grasp_depth=7))
    return fits


def tetrad_fits(cell: Cell, package: str) -> list[dict]:
    """``package``'s fits on each dataset of ``cell`` (module docstring)."""
    params = _params(cell)

    def fit(solution, algorithm, draws):
        return {
            "solution": solution,
            "algorithm": algorithm,
            "config": cell.setting,
            "params": params,
            "draws": list(range(draws)),
        }

    if package == "tetrad-7.6.3":
        draws = 1 if cell.block == "O2" else DRAWS
        fits = [fit(GATED[a], a, draws) for a in GATED]
        if cell.block == "P20" and cell.setting == "paper":
            fits += [fit("tetrad.gfci", "gfci", 1), fit("tetrad.cmd.fci.fisherz", "fci", 1)]
        return fits
    if package == "tetrad-7.6.10" and cell.setting == "paper" and cell.block in ("P20", "P100"):
        return [
            fit("tetrad.boss_fci.bic", "boss_fci", 1),
            fit("tetrad.grasp_fci.bic", "grasp_fci", 1),
        ]
    return []


def tetrad_solutions() -> dict:
    """Every Tetrad solution the lane runs, by name."""
    from andrey_bench.adapters import tetrad_gfci, tetrad_pag, tetrad_search

    adapters = (
        tetrad_pag.TetradBFCI(),
        tetrad_pag.TetradCmdGraspFCI(),
        tetrad_pag.TetradCmdFCI(),
        tetrad_gfci.TetradGFCI(),
        tetrad_search.TetradBossFCI(),
        tetrad_search.TetradGraspFCI(),
    )
    return {a.name: a for a in adapters}


def tetrad_child(request_path: str, out_path: str) -> int:
    """Run a request's Tetrad fits, one JVM per draw with one warm-up search; append a line each.

    A line's ``wall_s`` is Tetrad's search inside the JVM and ``call_s`` the whole call.
    """
    request = np.load(request_path)
    X = np.asarray(request["X"], dtype=np.float64)
    solutions = tetrad_solutions()
    with open(out_path, "a") as sink:
        for fit in json.loads(str(request["fits"])):
            adapter = solutions[fit["solution"]]
            adapter.warmup = 1
            known = adapter.params()
            params = {
                **known,
                **{k: v for k, v in fit["params"].items() if k in known or k == "depth"},
            }
            for draw in fit["draws"]:
                line = {"algorithm": fit["algorithm"], "config": fit["config"], "draw": draw}
                line["solution"] = adapter.name
                t0 = time.perf_counter()
                try:
                    marks = adapter.fit(X, params)
                    line.update(
                        status="ok",
                        wall_s=adapter.search_s,
                        call_s=time.perf_counter() - t0,
                        marks=np.asarray(marks).astype(int).tolist(),
                    )
                except Exception:
                    line.update(status="error", error_text=traceback.format_exc()[-2000:])
                line["versions"] = {"tetrad": adapter.package_version}
                sink.write(json.dumps(line) + "\n")
                sink.flush()
    return 0


# ------------------------------------------------------------------------------------------------
# run
# ------------------------------------------------------------------------------------------------


def _read(path: Path) -> list[dict]:
    return (
        [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    )


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def run_dataset(cell: Cell, seed: int, packages: list[str], args, cpu: int | None) -> tuple:
    """Fit every package on one dataset; return its truth line and its records."""
    X, truth = dataset(cell, seed)
    truth_line = {"cell": cell.name, "seed": seed, "rows": int(X.shape[0]), **truth}
    records = []
    for package in packages:
        if package == "andrey":
            fits, script = andrey_fits(cell), ANDREY_CHILD
            runs = sum(len(f["draws"]) + 1 for f in fits)  # each fit's warm-up and draws
        else:
            fits, script = tetrad_fits(cell, package), Path(__file__)
            runs = sum(2 * len(f["draws"]) for f in fits)  # one warm-up search per draw
        if args.smoke:
            fits = [{**f, "draws": f["draws"][:1]} for f in fits]
        if not fits:
            continue
        request = {
            "X": X,
            "fits": json.dumps(fits),
            "warmup_rows": max(50, 2 * X.shape[1]),
        }
        lines, killed, code, err = run_child(
            args.python, script, request, runs * cell.cap_draw_s + 300, cpu, args.workdir
        )
        found = {(x["algorithm"], x["config"], x["draw"]): x for x in lines}
        for fit in fits:
            for draw in fit["draws"]:
                line = found.get((fit["algorithm"], fit["config"], draw))
                record = {
                    "cell": cell.name,
                    "block": cell.block,
                    "setting": cell.setting,
                    "claim": cell.claim,
                    "seed": seed,
                    "package": package,
                    "algorithm": fit["algorithm"],
                    "config": fit["config"],
                    "fit": {k: v for k, v in fit.items() if k != "draws"},
                    "draw": draw,
                    "cap_draw_s": cell.cap_draw_s,
                    "machine_id": args.machine_id,
                    "cpu": cpu,
                }
                if line is None:
                    record.update(
                        status="timeout" if killed else "error",
                        error_text=None if killed else f"exit {code}: {err}",
                    )
                elif line["status"] == "ok" and line["wall_s"] > cell.cap_draw_s:
                    record.update({**line, "status": "timeout"})
                    record.pop("marks", None)
                else:
                    record.update(line)
                if record["status"] == "ok":
                    record.update(pag_scores(record["marks"], truth))
                records.append(record)
    return truth_line, records


def select_cells(args) -> list[Cell]:
    """The cells ``--blocks``, ``--settings``, and ``--cells`` name, all by default."""
    cells = grid()
    if args.blocks:
        cells = [c for c in cells if c.block in args.blocks.split(",")]
    if args.settings:
        cells = [c for c in cells if c.setting in args.settings.split(",")]
    if args.degrees:
        cells = [c for c in cells if str(c.degree) in args.degrees.split(",")]
    if args.latents:
        cells = [c for c in cells if str(c.latents) in args.latents.split(",")]
    if args.cells:
        wanted = set(args.cells.split(";"))
        cells = [c for c in cells if c.name in wanted]
    if not cells:
        raise SystemExit("no cell matches")
    return cells


def run(args) -> int:
    cells = select_cells(args)
    packages = args.packages.split(",")
    if unknown := set(packages) - set(PACKAGES):
        raise SystemExit(f"unknown packages {sorted(unknown)}; choose from {PACKAGES}")
    args.python = os.environ.get("ANDREY_BENCH_PYTHON") or sys.executable
    for variable in ("ANDREY_BENCH_TETRAD_JAR", "ANDREY_BENCH_TETRAD_GIN_JAR"):
        if any(p.startswith("tetrad") for p in packages) and not os.environ.get(variable):
            raise SystemExit(f"set {variable} (README: Tetrad on continuous data)")
    units = [
        (cell, seed)
        for cell in cells
        for seed in (args.seeds or (cell.seeds[:1] if args.smoke else cell.seeds))
    ]
    out = Path(args.out)
    if (out / RECORDS).exists() and not args.resume:
        raise SystemExit(f"{out} already holds a run; pass --resume to continue it")
    out.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "driver": Path(__file__).name,
        "sampler": "andrey.data, scale='raw'",
        "cells": [asdict(c) | {"name": c.name} for c in cells],
        "seeds": sorted({s for _, s in units}),
        "packages": packages,
        "settings": SETTINGS,
        "draws": DRAWS,
        "andrey_fits": {c.name: andrey_fits(c) for c in cells},
        "tetrad_fits": {c.name: {p: tetrad_fits(c, p) for p in PACKAGES[1:]} for c in cells},
        "smoke": args.smoke,
        "workers": args.workers,
        "machine_id": args.machine_id,
    }
    meta_path = out / "run_meta.json"
    if (
        args.resume
        and meta_path.exists()
        and not same_run(json.loads(meta_path.read_text()), run_meta)
    ):
        raise SystemExit(f"--resume: {meta_path} differs from these flags; pass the first run's")
    meta_path.write_text(json.dumps(run_meta, indent=2))
    from andrey_bench import rundir

    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())
    _log(f"=== BOSS-FCI / GRaSP-FCI lane, {len(units)} datasets -> {out} (git {prov['git_sha']})")

    done = {(r["cell"], r["seed"], r["package"]) for r in _read(out / RECORDS)}
    todo = [
        (cell, seed, [p for p in packages if (cell.name, seed, p) not in done])
        for cell, seed in units
    ]
    todo = [t for t in todo if t[2]]
    cpus = sorted(os.sched_getaffinity(0))
    free = list(cpus[: args.workers]) if len(cpus) >= args.workers else [None] * args.workers
    lock = threading.Lock()
    start = time.monotonic()
    skipped, failed = [], []

    with tempfile.TemporaryDirectory(prefix="bgf_") as tmp:
        args.workdir = Path(tmp)

        def task(item):
            cell, seed, pkgs = item
            if args.deadline_s and time.monotonic() - start > args.deadline_s:
                with lock:
                    skipped.append((cell.name, seed))
                return
            with lock:
                cpu = free.pop()
            try:
                truth_line, records = run_dataset(cell, seed, pkgs, args, cpu)
            except Exception:
                with lock:
                    failed.append((cell.name, seed, traceback.format_exc()[-2000:]))
                    _log(f"[{cell.name} seed {seed}] FAILED\n{failed[-1][2]}")
                return
            finally:
                with lock:
                    free.append(cpu)
            with lock:
                with (out / DATASETS).open("a") as sink:
                    sink.write(json.dumps(truth_line) + "\n")
                with (out / RECORDS).open("a") as sink:
                    sink.writelines(json.dumps(r) + "\n" for r in records)
                ok = sum(r["status"] == "ok" for r in records)
                elapsed = time.monotonic() - start
                _log(f"[{cell.name} seed {seed}] {ok}/{len(records)} ok | {elapsed:.0f}s")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(task, todo))

    records = _read(out / RECORDS)
    if skipped:
        _log(f"deadline: {len(skipped)} datasets left for --resume")
        return 3
    counts = {s: sum(r["status"] == s for r in records) for s in ("ok", "timeout", "error")}
    summary = {"datasets": len(units), "records": len(records), **counts, "failed": len(failed)}
    (out / DONE).write_text(json.dumps(summary, indent=2))
    _log(f"done: {summary}")
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------------
# truth check against pcalg
# ------------------------------------------------------------------------------------------------

#: pcalg's ``dag2pag`` on the DAG renumbered in a topological order (its ``ancTS`` reads the
#: parents among lower-numbered nodes only), its rows mapped back to the measured nodes, the first
#: ``k``. ``amat[i, j]`` is the mark at ``j``: 1 circle, 2 arrowhead, 3 tail.
_DAG2PAG = """
andrey_dag2pag <- function(A, k) {
    n <- nrow(A)
    dimnames(A) <- list(paste0("x", seq_len(n) - 1), paste0("x", seq_len(n) - 1))
    g <- methods::as(A, "graphNEL")
    perm <- match(RBGL::tsort(g), graph::nodes(g))
    A2 <- A[perm, perm]
    g2 <- methods::as(A2, "graphNEL")
    fit <- suppressWarnings(pcalg::dag2pag(list(g = g2, jp = RBGL::johnson.all.pairs.sp(g2)),
                                           indepTest = pcalg::dsepTest, graph = g2,
                                           L = which(perm > k), alpha = 0.5))
    obs <- perm[perm <= k]
    labels <- paste0("V", seq_len(k) - 1)
    amat <- matrix(0, k, k, dimnames = list(labels, labels))
    amat[obs, obs] <- fit@amat
    amat
}
"""


def check_truth(args) -> int:
    """Compare each distinct DAG's stored PAG with pcalg's ``dag2pag`` on at most
    ``--max-observed`` measured variables; write ``truth-check.json``."""
    from andrey_bench import rsession
    from andrey_bench.adapters.r_convert import marks_from_amat_pag

    lines = [x for run_dir in args.runs for x in _read(Path(run_dir) / DATASETS)]
    graphs: dict[str, dict] = {}
    for line in lines:
        if line["observed"] <= args.max_observed:
            key = json.dumps([line["nodes"], line["observed"], sorted(line["arcs"])])
            graphs.setdefault(key, {**line, "cells": []})["cells"].append(
                [line["cell"], line["seed"]]
            )
    session = rsession.start("graph", "RBGL", "pcalg")
    session.define(_DAG2PAG)
    dag2pag = session.function("andrey_dag2pag")
    disagree = []
    for line in graphs.values():
        A = np.zeros((line["nodes"], line["nodes"]))
        for u, v in line["arcs"]:
            A[u, v] = 1.0
        result = dag2pag(session.matrix(A), line["observed"])
        amat = session.square_matrix(result, names=rsession.columns(line["observed"]))
        theirs = marks_from_amat_pag(amat).astype(int)
        mine = np.asarray(line["pag"])
        if not np.array_equal(mine, theirs):
            pairs = np.argwhere(np.triu((mine != theirs) | (mine.T != theirs.T), 1)).tolist()
            disagree.append({"cells": line["cells"], "pairs": pairs})
    result = {
        "pcalg": rsession.package_version("pcalg"),
        "max_observed": args.max_observed,
        "graphs": len(graphs),
        "datasets": sum(len(g["cells"]) for g in graphs.values()),
        "agree": len(graphs) - len(disagree),
        "disagree": disagree,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "truth-check.json").write_text(json.dumps(result, indent=2))
    _log(f"truth check: {result['agree']} of {result['graphs']} graphs agree with pcalg")
    return 1 if disagree else 0


# ------------------------------------------------------------------------------------------------
# summarize
# ------------------------------------------------------------------------------------------------

#: Every score :func:`per_dataset` averages over a dataset's draws.
METRICS = (
    "shd",
    "empty_shd",
    "edges",
    "adjacency_precision",
    "adjacency_recall",
    "arrowhead_precision",
    "arrowhead_recall",
    "arrowhead_f1",
    "arrowhead_common_precision",
    "arrowhead_common_recall",
    "tail_precision",
    "tail_recall",
    "tail_f1",
    "arrow_path_precision",
    "tail_path_precision",
    "bidirected_latent_precision",
)


def per_dataset(records: list[dict]) -> dict[tuple, dict]:
    """Each (cell, seed, package, algorithm, config): whether every draw finished, the means of
    :data:`METRICS` over its draws, the median times, the distinct graphs over its draws, and the
    largest possible-d-separation candidate set."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        key = (r["cell"], r["seed"], r["package"], r["algorithm"], r["config"])
        groups.setdefault(key, []).append(r)
    out = {}
    for key, rows in groups.items():
        ok = [r for r in rows if r["status"] == "ok"]
        entry = {"complete": len(ok) == len(rows), "draws": len(rows), "timeouts": 0}
        entry["timeouts"] = sum(r["status"] == "timeout" for r in rows)
        if entry["complete"]:
            for metric in METRICS:
                values = [r.get(metric) for r in ok]
                entry[metric] = None if None in values else float(np.mean(values))
            entry["wall_s"] = statistics.median(r["wall_s"] for r in ok)
            calls = [r["call_s"] for r in ok if r.get("call_s") is not None]
            entry["call_s"] = statistics.median(calls) if calls else None
            entry["distinct"] = len({json.dumps(r["marks"]) for r in ok})
            dsep = [r["possible_dsep_max"] for r in ok if r.get("possible_dsep_max") is not None]
            entry["possible_dsep_max"] = max(dsep) if dsep else None
        out[key] = entry
    return out


#: Each comparison :func:`summarize` writes: Andrey's (algorithm, config suffix) against a
#: (package, algorithm, config suffix), its comparator name, and its role.
COMPARISONS = (
    (("boss_fci", ""), ("tetrad-7.6.3", "boss_fci", ""), "tetrad-cmd-1.12.0.bfci", "gated"),
    (("grasp_fci", ""), ("tetrad-7.6.3", "grasp_fci", ""), "tetrad-cmd-1.12.0.grasp-fci", "gated"),
    (("boss_fci", ""), ("tetrad-7.6.10", "boss_fci", ""), "tetrad-lib-7.6.10.BossFci", "context"),
    (
        ("grasp_fci", ""),
        ("tetrad-7.6.10", "grasp_fci", ""),
        "tetrad-lib-7.6.10.GraspFci",
        "context",
    ),
    (
        ("boss_fci", "-no-possible-dsep"),
        ("tetrad-7.6.3", "boss_fci", ""),
        "tetrad-cmd-1.12.0.bfci@andrey-no-possible-dsep",
        "trace",
    ),
    (
        ("grasp_fci", "-no-possible-dsep"),
        ("tetrad-7.6.3", "grasp_fci", ""),
        "tetrad-cmd-1.12.0.grasp-fci@andrey-no-possible-dsep",
        "trace",
    ),
    (
        ("boss_fci", "-no-possible-dsep"),
        ("andrey", "boss_fci", ""),
        "andrey.boss_fci@no-possible-dsep",
        "trace",
    ),
    (
        ("grasp_fci", "-no-possible-dsep"),
        ("andrey", "grasp_fci", ""),
        "andrey.grasp_fci@no-possible-dsep",
        "trace",
    ),
    (("gfci", ""), ("tetrad-7.6.3", "gfci", ""), "tetrad-cmd-1.12.0.gfci", "context"),
    (("fci", ""), ("tetrad-7.6.3", "fci", ""), "tetrad-cmd-1.12.0.fci", "context"),
)


def summarize(records: list[dict], truths: list[dict]) -> dict:
    """The paired comparisons of finished runs, one entry per (regime, comparison), one row per
    ``n``; the reference's draws and the largest possible-d-separation candidate sets."""
    by = per_dataset(records)
    cells = {c.name: c for c in grid()}
    arms: dict[tuple, dict[int, dict]] = {}
    for (cell, seed, package, algorithm, config), entry in by.items():
        arms.setdefault((cell, package, algorithm, config), {})[seed] = entry
    hidden: dict[str, list[int]] = {}
    for line in truths:
        hidden.setdefault(line["cell"], []).append(line["latents_hidden"])
    entries: dict[tuple, dict] = {}
    for name, cell in cells.items():
        for (algorithm, mine_suffix), (package, other, suffix), comparator, role in COMPARISONS:
            mine = arms.get((name, "andrey", algorithm, cell.setting + mine_suffix))
            theirs = arms.get((name, package, other, cell.setting + suffix))
            if not mine or not theirs:
                continue
            entry = entries.setdefault(
                (cell.regime, algorithm, comparator),
                {
                    "method": algorithm,
                    "comparator": comparator,
                    "role": role,
                    "regime": cell.regime,
                    "metric": "shd",
                    "claim": cell.claim,
                    "rows": [],
                },
            )
            row = {"mean_degree": cell.degree, "d": cell.d, "latents": cell.latents, "n": cell.n}
            if hidden.get(name):
                row["latents_hidden_mean"] = float(np.mean(hidden[name]))
            fresh = [s for s in set(mine) | set(theirs) if s not in cell.seeds]
            if fresh:
                rerun = replication(
                    {s: e for s, e in mine.items() if s in fresh},
                    {s: e for s, e in theirs.items() if s in fresh},
                    "shd",
                    len(fresh),
                )
                mine = {s: e for s, e in mine.items() if s in cell.seeds}
                theirs = {s: e for s, e in theirs.items() if s in cell.seeds}
            row.update(compare(mine, theirs, "shd", len(cell.seeds)))
            if fresh and rerun is not None:
                row["replication"] = rerun
            andrey_floor = floor_interval(mine, "shd", "empty_shd")
            other_floor = floor_interval(theirs, "shd", "empty_shd")
            if andrey_floor is not None and other_floor is not None:
                row["floor"] = {
                    "applies": other_floor[1] < 0,
                    "andrey_minus_empty": {"bootstrap_95": andrey_floor},
                    "reference_minus_empty": {"bootstrap_95": other_floor},
                }
            for metric in METRICS[2:]:
                for side, arm in (("andrey", mine), ("comparator", theirs)):
                    values = [e[metric] for e in arm.values() if e.get(metric) is not None]
                    row[f"{side}_{metric}_mean"] = float(np.mean(values)) if values else None
            entry["rows"].append(row)
    draws = {}
    dsep = {}
    for (name, package, algorithm, config), arm in sorted(arms.items()):
        done = [e for e in arm.values() if e["complete"]]
        if package != "andrey":
            draws[f"{name}|{package}|{algorithm}"] = {
                "datasets": len(done),
                "draws": max((e["draws"] for e in done), default=0),
                "datasets_with_distinct_draws": sum(e["distinct"] > 1 for e in done),
            }
        sizes = [e["possible_dsep_max"] for e in done if e.get("possible_dsep_max") is not None]
        if package == "andrey" and done:
            dsep[f"{name}|{algorithm}|{config}"] = {
                "datasets": len(done),
                "max": max(sizes) if sizes else None,
                "median": statistics.median(sizes) if sizes else None,
            }
    return {
        "parity_rule": "paired bootstrap",
        "methods": list(entries.values()),
        "reference_draws": draws,
        "possible_dsep_candidates": dsep,
    }


def _run_dirs(paths: list[str]) -> list[Path]:
    return [Path(p) for p in paths if (Path(p) / RECORDS).exists()]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0].endswith(".npz"):
        return tetrad_child(*argv)
    p = argparse.ArgumentParser(description="BOSS-FCI and GRaSP-FCI against Tetrad")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="fit every package on the cells' datasets")
    r.add_argument("--out", required=True)
    r.add_argument("--blocks", help="comma list: P20, P100, O1, O2, O3")
    r.add_argument("--settings", help="comma list: paper, default")
    r.add_argument("--degrees", help="comma list of mean degrees")
    r.add_argument("--latents", help="comma list of hidden-node counts")
    r.add_argument("--cells", help="cell names, separated by ';'")
    r.add_argument(
        "--seeds",
        type=parse_seeds,
        help="'0-1' or '100-119': the seeds of every cell, in place of its own",
    )
    r.add_argument("--packages", default=",".join(PACKAGES))
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--deadline-s", type=float, default=None)
    r.add_argument("--machine-id", default=socket.gethostname())
    r.add_argument("--smoke", action="store_true", help="one seed and one draw per cell")
    r.add_argument("--resume", action="store_true")
    t = sub.add_parser("check-truth", help="the stored truth against pcalg's dag2pag")
    t.add_argument("--runs", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--max-observed", type=int, default=30)
    s = sub.add_parser("summarize", help="the paired comparisons of finished runs")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.command == "run":
        return run(args)
    if args.command == "check-truth":
        return check_truth(args)
    dirs = _run_dirs(args.runs)
    records = [x for d in dirs for x in _read(d / RECORDS)]
    truths = [x for d in dirs for x in _read(d / DATASETS)]
    result = summarize(records, truths)
    Path(args.out).write_text(json.dumps(result, indent=2))
    _log(f"summary: {len(result['methods'])} entries -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
