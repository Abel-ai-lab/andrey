#!/usr/bin/env python
"""Benchmark IGSP on CPU: Andrey against the authors' library, on interventional data.

Every dataset comes from ``andrey.data`` (:func:`dataset`) with an observational environment and
``K`` settings of ``n`` rows each, in the generator's own units (``scale="raw"``), its columns in
an order drawn from the seed. The targets are column positions, drawn by ``default_rng([seed,
655])``. :func:`grid` lists the cells:

- **W**, Wang et al. (2017, Figure 2): an Erdos-Renyi DAG with mean degree 1.5 on 10 or 20
  variables, weights uniform on ``[0.1, 1]`` in absolute value with a random sign, standard normal
  noise; ``K = 1`` setting of 5 (10) targets or ``K = 2`` disjoint settings of 4 (8), each target
  cut from its causes and given the noise ``N(0, 0.2)``; ``n`` 1e3, 1e4, 1e5.
- **S**, Wang et al. (2017, Figure 3): the 7-node DAG of their Figure 1 (left), weights on ``[c,
  1]`` for ``c`` 0.1, 0.25, 0.5, one setting on ``{4}``, ``{5}``, or ``{4, 5}`` (the paper's node
  names, mapped to positions through ``provenance["column_order"]``), Wang's intervention.
- **Y**, Yang, Katcoff, and Uhler (2018, Figures 5, 6, 8): weights on ``[0.25, 1]``; perfect
  (cut from the causes, noise ``N(0, 1)`` kept), inhibiting (incoming weights divided by 10), or
  imperfect (perfect on a random half of the rows: each target succeeds on each row with
  probability 1/2, independently) interventions; Y6 ``k`` single-target settings on ``p = 10``, Y5
  every variable of ``p = 20``, Y8 every pair of ``p = 10``; ``n`` 1e3 and 1e4 (the lane's choice).
- **Q**, the confirmation family of the defaults (Y perfect, ``p = 20``, ``k = 8``, mean degree 3,
  weights on ``[0.5, 2]``), and the out-of-claim cells K and O1-O9.

A target whose intervention leaves its distribution unchanged is null: under Yang's three kinds,
every target without parents. Each setting's effective targets are its targets less the null
ones. Scores (:func:`score`): every arm's output is reduced to a DAG (a returned DAG as is, an
interventional essential graph through ``pdag2dag``) and read as its interventional essential
graph under the effective targets, the truth that of the generating DAG under the same targets;
``shd`` counts the variable pairs whose endpoint marks differ.

Per dataset, each package runs in one child process that fits every configuration and draw
(:mod:`andrey_bench.adapters.andrey_interventional`,
:mod:`andrey_bench.adapters.gml_interventional`), pinned to one core with one BLAS thread. The
reference runs in ``ANDREY_BENCH_UTIGSP_PYTHON``; its copy with Definition 4.2's I-contradicting
test (``patches/gml-igsp-def42.patch``) in ``ANDREY_BENCH_GML_DEF42_PYTHON``. ``run`` writes
``records.jsonl``, ``datasets.jsonl``, ``run_meta.json``, ``provenance.json``, and ``done.json``;
``summarize`` the summary entries ``qa/reference_parity.py`` reads; ``select`` the rule that
chooses Andrey's defaults; ``validate-patch`` the checks of the patched reference under oracle
tests.

    benchmarks/start_utigsp.sh
    export ANDREY_BENCH_UTIGSP_PYTHON=$PWD/benchmarks/.venv-utigsp/bin/python
    export ANDREY_BENCH_GML_DEF42_PYTHON=$PWD/benchmarks/.venv-gml-def42/bin/python
    $ANDREY_BENCH_PYTHON benchmarks/igsp_cpu.py validate-patch --out outputs/igsp-patch
    $ANDREY_BENCH_PYTHON benchmarks/igsp_cpu.py run --out outputs/igsp-W --arm W
    $ANDREY_BENCH_PYTHON benchmarks/igsp_cpu.py summarize --runs outputs/igsp-W \\
        --out outputs/igsp-summary.json
"""

from __future__ import annotations

import argparse
import functools
import inspect
import itertools
import json
import os
import socket
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ut_igsp_cpu import (
    ANDREY_CHILD,
    GML_CHILD,
    REFERENCE_PYTHON,
    compare,
    dag_from_arcs,
    essential,
    floor_interval,
    pairs_differ,
    parse_seeds,
    replication,
    run_child,
    same_run,
    skeleton_differ,
)

DEF42_PYTHON = "ANDREY_BENCH_GML_DEF42_PYTHON"
RECORDS, DATASETS, DONE = "records.jsonl", "datasets.jsonl", "done.json"
SAMPLER = "andrey.data"

#: The gate's levels, inside Wang et al.'s sweep, and the candidates for Andrey's default.
ALPHAS = (1e-4, 1e-3, 1e-2)
CANDIDATES = (1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2)
#: Andrey's candidate numbers of starts; the reference library's number of runs.
STARTS, NRUNS = (1, 5, 10), 5
#: The level of the context arm UT-IGSP with every target known: the UT-IGSP lane's.
UT_ALPHA = 1e-5
#: Draws per dataset of a randomized method in a judged cell; one elsewhere.
DRAWS = 5
#: First SCM seed of each block: A chooses Andrey's defaults, B compares, C replicates, N is the
#: random-ranking null, Q confirms the defaults outside the grid.
BLOCKS = {"A": 1000, "B": 0, "C": 100, "N": 2000, "Q": 3000}
#: Seconds a draw may take in claim and out of it.
CAP_DRAW_S, CAP_DRAW_LONG_S = 1800.0, 3600.0

#: Wang et al. (2017, Figure 1, left): 5 is the collider of 3 and 7.
S_NODES = ("1", "2", "3", "4", "5", "6", "7")
S_ARCS = (("1", "4"), ("2", "4"), ("3", "4"), ("3", "5"), ("7", "5"), ("4", "6"))
#: Block S's target families by name: one setting on 4, on 5, on 4 and 5, or two settings.
S_FAMILIES = {"4": [["4"]], "5": [["5"]], "45": [["4", "5"]], "4-5": [["4"], ["5"]]}

#: Each kind's intervention: ``(constructor, arguments)`` from ``andrey.data.interventions``.
KINDS: dict[str, tuple[str, dict]] = {
    "wang": ("hard", {"value": 0.0, "sd": 0.2**0.5}),
    "perfect": ("hard", {"value": 0.0, "sd": 1.0}),
    "inhibiting": ("soft", {"weight_scale": 0.1}),
    "imperfect": ("hard", {"value": 0.0, "sd": 1.0}),
    "shift": ("soft", {"shift": 1.0}),
    "discrete": ("hard", {}),
}
#: The kinds under which a target without parents keeps its observational law.
NULL_ON_ROOTS = ("perfect", "inhibiting", "imperfect")


@dataclass(frozen=True)
class Cell:
    """One cell of the grid: a setting, its datasets, and whether it is in the method's claim."""

    arm: str
    kind: str
    n: int
    p: int = 10
    settings: int = 1
    size: int = 1
    family: str = ""
    c: float = 0.25
    hi: float = 1.0
    degree: float = 1.5
    obs_ratio: int = 1
    off: int = 0
    functional: str = "linear"
    noise: str = "gaussian"
    scale: str = "raw"
    claim: str = "in"
    block: str = "B"
    datasets: int = 20

    @property
    def group(self) -> str:
        """The summary regime without its alpha: every axis but the row keys ``d`` and ``n``."""
        if self.arm == "W":
            return f"W|K={self.settings}"
        if self.family:
            parts = [self.arm, f"family={family_text(self.family)}", f"c={self.c:g}"]
        elif self.arm in ("Y5", "Y8"):
            parts = [self.arm, self.kind]
        else:
            parts = [self.arm, self.kind, f"k={self.settings}"]
        if self.arm in ("K", "O5", "O7"):
            parts[1] = {"K": "kci", "O5": "fisherz", "O7": "gsq"}[self.arm]
        if self.noise != "gaussian":
            parts.append(self.noise)
        if self.off:
            parts.append(f"off={self.off}")
        if self.obs_ratio != 1:
            parts.append(f"C={self.obs_ratio}")
        return "|".join(parts)

    @property
    def name(self) -> str:
        """The cell's identifier in records and on the command line."""
        text = self.group.replace("|", "_")
        for mark in "=,{}":
            text = text.replace(mark, "")
        text += f"_p{self.p}_n{self.n}"
        return text + (f"_deg{self.degree:g}" if self.degree != 1.5 else "")

    @property
    def seeds(self) -> list[int]:
        return list(range(BLOCKS[self.block], BLOCKS[self.block] + self.datasets))

    @property
    def multi_target(self) -> bool:
        """Whether a setting holds two or more targets: the cells judged against both copies."""
        return self.size > 1 or self.family == "45"

    @property
    def cap_draw_s(self) -> float:
        return CAP_DRAW_S if self.claim == "in" else CAP_DRAW_LONG_S

    @property
    def draws(self) -> int:
        return DRAWS if self.claim == "in" and self.block in ("B", "C") else 1


def family_text(code: str) -> str:
    """``"4"``, ``"5"``, ``"4,5"`` (one setting), or ``"{4},{5}"`` (two settings)."""
    return ",".join(code) if "-" not in code else ",".join(f"{{{v}}}" for v in code.split("-"))


def grid() -> list[Cell]:
    """Every cell: the in-claim blocks W, S, and Y (87 cells), Q, and the out-of-claim cells."""
    sizes = {(10, 1): 5, (10, 2): 4, (20, 1): 10, (20, 2): 8}
    cells = [
        Cell("W", "wang", n, p=p, settings=K, size=sizes[p, K], c=0.1)
        for p in (10, 20)
        for K in (1, 2)
        for n in (1000, 10_000, 100_000)
    ]
    cells += [
        Cell("S", "wang", n, p=7, family=f, c=c)
        for f in ("4", "5", "45")
        for c in (0.1, 0.25, 0.5)
        for n in (1000, 10_000, 100_000)
    ]
    kinds = ("perfect", "inhibiting", "imperfect")
    cells += [
        Cell("Y6", kind, n, settings=k)
        for kind in kinds
        for k in (1, 2, 4, 6, 8, 10)
        for n in (1000, 10_000)
    ]
    cells += [Cell("Y5", kind, n, p=20, settings=20) for kind in kinds for n in (1000, 10_000)]
    cells += [Cell("Y8", kind, n, settings=45, size=2) for kind in kinds for n in (1000, 10_000)]
    out = {"claim": "out"}
    cells += [
        Cell("Q", "perfect", n, p=20, settings=8, c=0.5, hi=2.0, degree=3.0, block="Q", **out)
        for n in (1000, 10_000)
    ]
    standardized = {"functional": "additive", "scale": "standardize", "settings": 4, **out}
    cells += [Cell(arm, "wang", n, **standardized) for arm in ("K", "O5") for n in (200, 500)]
    cells += [
        Cell("O1", "perfect", 1000, p=p, settings=10, datasets=10, **out)
        for p in (50, 100, 200, 400)
    ]
    cells += [
        Cell("O2", "perfect", n, p=p, settings=k, **out)
        for p, k in ((10, 4), (20, 8))
        for n in (100, 300)
    ]
    cells += [Cell("O3", "perfect", 10_000, p=20, settings=8, degree=d, **out) for d in (3.0, 4.0)]
    cells += [Cell("O4", kind, 5000, p=20, settings=5, off=1, **out) for kind in ("wang", "shift")]
    cells += [
        Cell("O7", "discrete", n, settings=4, functional="discrete", **out) for n in (1000, 10_000)
    ]
    cells += [
        Cell("O8", "wang", n, settings=4, noise=noise, **out)
        for noise in ("laplace", "uniform")
        for n in (1000, 10_000)
    ]
    cells += [
        Cell("O9", "wang", n, p=7, family="4-5", obs_ratio=C, **out)
        for n in (1000, 10_000)
        for C in (1, 3, 10)
    ]
    cells += [Cell("O9", "perfect", 1000, settings=4, obs_ratio=C, **out) for C in (3, 10)]
    return cells


def cell_named(name: str) -> Cell:
    for cell in grid():
        if cell.name == name:
            return cell
    raise KeyError(f"no cell named {name!r}")


def alpha_text(alpha: float) -> str:
    """``1e-3``, ``5e-4``: a level as the summary regimes write it."""
    mantissa, exponent = f"{alpha:e}".split("e")
    return f"{float(mantissa):g}e{int(exponent)}"


# ------------------------------------------------------------------------------------------------
# Data and truth
# ------------------------------------------------------------------------------------------------


def _intervention(kind: str):
    from andrey.data.interventions import hard, soft

    maker, kwargs = KINDS[kind]
    if maker == "soft":
        return soft(**kwargs)
    kwargs = dict(kwargs)
    return hard(kwargs.pop("value", None), **kwargs)


def _scm(cell: Cell):
    import andrey.data as data

    if cell.family:
        graph = data.graphs.FixedDAG(nodes=S_NODES, arcs=S_ARCS)
    else:
        graph = data.graphs.erdos_renyi(d=cell.p, avg_degree=cell.degree)
    if cell.functional == "additive":
        functional = data.functional.additive_noise()
    else:
        functional = data.functional.linear(weight_range=(cell.c, cell.hi))
    noise = getattr(data.noise, cell.noise)()
    return data.SCM(graph=graph, functional=functional, noise=noise)


def named_positions(column_order: list[int]) -> dict[str, int]:
    """Each named node's column: column ``j`` holds node ``column_order[j]`` of :data:`S_NODES`."""
    return {S_NODES[node]: j for j, node in enumerate(column_order)}


def planned_targets(cell: Cell, seed: int, column_order: list[int]) -> list[list[int]]:
    """Each setting's targets, as column positions, before any null target is dropped."""
    rng = np.random.default_rng([seed, 655])
    if cell.family:
        at = named_positions(column_order)
        return [sorted(at[v] for v in names) for names in S_FAMILIES[cell.family]]
    if cell.arm == "W":
        order = rng.permutation(cell.p)
        return [
            sorted(int(v) for v in order[k * cell.size : (k + 1) * cell.size])
            for k in range(cell.settings)
        ]
    if cell.arm == "Y5":
        return [[v] for v in range(cell.p)]
    if cell.arm == "Y8":
        return [[i, j] for i, j in itertools.combinations(range(cell.p), 2)]
    designed = [[int(v)] for v in rng.choice(cell.p, cell.settings, replace=False)]
    for k, targets in enumerate(designed):
        others = [v for v in range(cell.p) if v != targets[0]]
        off = np.random.default_rng([seed, 655, k]).choice(others, cell.off, replace=False)
        targets += sorted(int(v) for v in off)
    return designed


def environments(cell: Cell, seed: int, planned: list[list[int]]):
    """The sampler's environments, their row counts, and each one's setting.

    An imperfect setting is several environments: one per subset of its targets, each target
    succeeding on each row with probability 1/2 independently, so the subsets' rows are a
    multinomial split of ``n`` (an empty subset draws observational rows); all of them are the
    setting's rows.
    """
    spec = _intervention(cell.kind)
    envs: list[dict] = [{}]
    rows, setting = [cell.n * cell.obs_ratio], [0]
    rng = np.random.default_rng([seed, 655, 1])
    for k, targets in enumerate(planned, start=1):
        if cell.kind != "imperfect":
            envs.append({v: spec for v in targets})
            rows.append(cell.n)
            setting.append(k)
            continue
        subsets = [
            s for r in range(len(targets), -1, -1) for s in itertools.combinations(targets, r)
        ]
        counts = rng.multinomial(cell.n, [1 / len(subsets)] * len(subsets))
        for subset, count in zip(subsets, counts, strict=True):
            if count:
                envs.append({v: spec for v in subset})
                rows.append(int(count))
                setting.append(k)
    return envs, rows, np.asarray(setting, dtype=np.int64)


def dataset(cell: Cell, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(X, env, truth)`` for one dataset of ``cell``.

    ``truth`` holds the generating DAG's endpoint marks (``dag``), each setting's targets
    (``nominal``), its effective targets (``effective``), the targets the fit is given (``given``:
    the nominal ones less any off-target), the number of null (setting, target) pairs
    (``null_targets``), and the true arcs out of a null target (``null_arcs``).
    """
    from andrey.core import ARROW

    if cell.functional == "discrete":
        from andrey.data import sample_discrete

        planned = planned_targets(cell, seed, [])
        envs, rows, setting = environments(cell, seed, planned)
        ds = sample_discrete(d=cell.p, n=rows, seed=seed, envs=envs)
    else:
        scm = _scm(cell)
        bare = scm.sample(n=2, seed=seed, scale=cell.scale)
        planned = planned_targets(cell, seed, bare.provenance["column_order"])
        envs, rows, setting = environments(cell, seed, planned)
        ds = scm.sample(n=rows, seed=seed, scale=cell.scale, envs=envs)
        if not np.array_equal(ds.graph.to_numpy(), bare.graph.to_numpy()):
            raise RuntimeError(f"{cell.name} seed {seed}: the graph depends on the rows")
    for e, targets in ds.targets.items():
        expected = planned[setting[e] - 1] if e else []
        if cell.kind == "imperfect" and set(targets) <= set(expected):
            continue
        if sorted(targets) != sorted(expected):
            raise RuntimeError(f"{cell.name} seed {seed}: the sampler moved the targets")
    dag = np.asarray(ds.graph.to_numpy(), dtype=int)
    roots = ~(dag == ARROW).any(axis=1)
    null = set()
    if cell.kind in NULL_ON_ROOTS:
        null = {v for t in planned for v in t if roots[v]}
    given = [t[:1] if cell.off else t for t in planned]
    truth = {
        "dag": dag.tolist(),
        "nominal": planned,
        "effective": [sorted(set(t) - null) for t in planned],
        "given": given,
        "null_targets": sum(len(set(t) & null) for t in planned),
        "null_arcs": [
            [int(r), int(c)] for r in sorted(null) for c in np.flatnonzero(dag[:, r] == ARROW)
        ],
    }
    return np.asarray(ds.data, dtype=np.float64), setting[np.asarray(ds.env)], truth


def score(line: dict, package: str, truth: dict) -> dict:
    """Every metric of one finished draw against ``truth``.

    The estimate is the arm's DAG, the reference's as returned and an Andrey essential graph's
    consistent extension, read under the effective targets (``shd``, ``skeleton_shd``,
    ``orientation_errors``) and under the nominal ones (``shd_nominal``); Andrey's ``grasp`` on the
    observational rows is scored as returned. ``null_reversed`` and ``null_kept`` count the true
    arcs out of a null target that the estimate under the nominal targets directs against or along
    the truth; ``dag_hamming`` compares a returned DAG with the true one. ``consistent`` is whether
    Andrey's essential graph equals the one the scorer rebuilds from it under the given targets.
    """
    from andrey.core import ARROW, TAIL
    from andrey.core.orient import pdag2dag

    true_dag = np.asarray(truth["dag"], dtype=np.int8)
    p = true_dag.shape[0]
    truth_eff = essential(true_dag, truth["effective"])
    truth_nom = essential(true_dag, truth["nominal"])
    algorithm = line["algorithm"]
    consistent, dag = None, None
    if package != "andrey":
        dag = dag_from_arcs(line["arcs"], p)
        est_eff, est_nom = essential(dag, truth["effective"]), essential(dag, truth["nominal"])
    elif algorithm == "grasp":
        est_eff = est_nom = np.asarray(line["marks"], dtype=np.int8)
    else:
        marks = np.asarray(line["marks"], dtype=np.int8)
        member = pdag2dag(marks)
        est_eff, est_nom = (
            essential(member, truth["effective"]),
            essential(member, truth["nominal"]),
        )
        if algorithm == "igsp":
            consistent = bool(np.array_equal(essential(member, truth["given"]), marks))
    shd = pairs_differ(est_eff, truth_eff)
    skeleton = skeleton_differ(est_eff, truth_eff)
    arcs = truth["null_arcs"]
    out = {
        "shd": shd,
        "shd_nominal": pairs_differ(est_nom, truth_nom),
        "skeleton_shd": skeleton,
        "orientation_errors": shd - skeleton,
        "iess_exact": shd == 0,
        "empty_shd": int(np.count_nonzero(np.triu(truth_eff != 0))),
        "null_targets": truth["null_targets"],
        "null_arcs": len(arcs),
        "null_reversed": sum(est_nom[c, r] == TAIL and est_nom[r, c] == ARROW for r, c in arcs),
        "null_kept": sum(est_nom[r, c] == TAIL and est_nom[c, r] == ARROW for r, c in arcs),
        "dag_hamming": None if dag is None else pairs_differ(dag, true_dag),
        "consistent": consistent,
    }
    out["null_reversed"], out["null_kept"] = int(out["null_reversed"]), int(out["null_kept"])
    return out


# ------------------------------------------------------------------------------------------------
# Fits
# ------------------------------------------------------------------------------------------------


def config_name(alpha: float, starts: int) -> str:
    """``a<alpha>_s<n_starts>`` (Andrey) or ``a<alpha>_r<nruns>`` (the reference)."""
    return f"a{alpha:g}_s{starts}"


def andrey_defaults() -> dict:
    """``igsp``'s shipped ``alpha`` and ``n_starts``."""
    import andrey

    params = inspect.signature(andrey.igsp).parameters
    return {"alpha": params["alpha"].default, "n_starts": params["n_starts"].default}


def gate_alphas() -> list[float]:
    """The gate's levels and Andrey's default when it is a fourth."""
    return sorted({*ALPHAS, andrey_defaults()["alpha"]})


def judged_starts() -> list[int]:
    """Andrey's numbers of starts in the judged rows: the reference's number of runs (J1, matched)
    and ``igsp``'s shipped ``n_starts`` (J2) when it differs."""
    return sorted({NRUNS, andrey_defaults()["n_starts"]})


def _igsp_fit(alpha: float, starts: int, draws: list[int], **kwargs) -> dict:
    return {
        "algorithm": "igsp",
        "config": config_name(alpha, starts),
        "kwargs": {"alpha": alpha, "n_starts": starts, **kwargs},
        "draws": draws,
    }


def andrey_fits(cell: Cell, step: int = 1, winner: str | None = None) -> list[dict]:
    """Andrey's fits on one dataset of ``cell``.

    Blocks A and Q (Andrey's defaults, draw 0): step 1 runs every candidate alpha at 5 starts,
    step 2 the winner's alpha at 1 and 10 starts. Elsewhere ``igsp`` at each of
    :func:`judged_starts` and every gate level (in claim) or its shipped ``alpha`` (out of claim),
    and the context arms the design lists for the cell.
    """
    if cell.block == "A" or cell.arm == "Q":
        if step == 1:
            return [_igsp_fit(a, 5, [0]) for a in CANDIDATES]
        if not winner:
            raise SystemExit("--step 2 needs --winner, the step-1 winner a<alpha>_s5")
        alpha = float(winner[1:].split("_")[0])
        return [_igsp_fit(alpha, s, [0]) for s in STARTS if s != 5]
    defaults = andrey_defaults()
    draws = list(range(cell.draws))
    extra = {"K": {"indep_test": "kci"}, "O7": {"indep_test": "gsq"}}.get(cell.arm, {})
    alphas = gate_alphas() if cell.claim == "in" else [defaults["alpha"]]
    fits = [_igsp_fit(a, s, draws, **extra) for a in alphas for s in judged_starts()]
    one = {"config": "default", "kwargs": {}, "draws": [0]}
    if cell.claim == "in" or cell.arm in ("O1", "O2", "O3", "O4", "O7", "O8", "O9"):
        fits.append({"algorithm": "gies", **one})
    if cell.claim == "in":
        fits.append({"algorithm": "grasp", **one, "config": "observational"})
    if cell.claim == "in" or cell.arm == "O4":
        fits.append({"algorithm": "ut_igsp", **one, "config": "known_targets"})
    return fits


def reference_fits(cell: Cell, package: str) -> list[dict]:
    """The fits of ``package``, ``reference`` or ``def42``, on one dataset of ``cell``.

    The reference's ``igsp`` at Andrey's levels with 5 runs, in every cell but blocks A and Q,
    O4, O5, O7, and O1 past 100 variables; ``def42``, the patched copy, in the in-claim cells with
    a multi-target setting. In claim, both also run at Andrey's shipped ``alpha`` with its shipped
    ``n_starts`` as the number of runs when that is not 5 (context), and the reference runs
    ``unknown_target_igsp`` with every target known, at the UT-IGSP lane's level.
    """
    if cell.block == "A" or cell.arm in ("Q", "O4", "O5", "O7") or cell.p > 100:
        return []
    if package == "def42" and not (cell.claim == "in" and cell.multi_target):
        return []
    draws = list(range(cell.draws))
    tester = "kci" if cell.arm == "K" else "gauss"
    defaults = andrey_defaults()
    alphas = gate_alphas() if cell.claim == "in" else [defaults["alpha"]]
    runs = [(a, NRUNS) for a in alphas]
    if cell.claim == "in" and defaults["n_starts"] != NRUNS:
        runs.append((defaults["alpha"], defaults["n_starts"]))
    fits = [
        {
            "algorithm": "igsp",
            "config": config_name(a, r).replace("_s", "_r"),
            "alpha": a,
            "nruns": r,
            "tester": tester,
            "draws": draws,
        }
        for a, r in runs
    ]
    if package == "reference" and cell.claim == "in":
        fits.append(
            {
                "algorithm": "ut_igsp",
                "config": "known_targets",
                "alpha": UT_ALPHA,
                "nruns": NRUNS,
                "tester": "gauss",
                "draws": [0],
            }
        )
    return fits


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
    X, env, truth = dataset(cell, seed)
    known = np.zeros(X.shape, dtype=np.uint8)
    for k, given in enumerate(truth["given"], start=1):
        known[np.ix_(env == k, given)] = 1
    truth_line = {
        "cell": cell.name,
        "seed": seed,
        **truth,
        "empty_shd": int(
            np.count_nonzero(np.triu(essential(np.array(truth["dag"]), truth["effective"]) != 0))
        ),
        "rows": int(X.shape[0]),
        "moments": np.concatenate([X.mean(axis=0), X.std(axis=0)]).tolist(),
    }
    records = []
    for package in packages:
        if package == "andrey":
            fits = andrey_fits(cell, args.step, args.winner)
            python, script = args.andrey_python, ANDREY_CHILD
        else:
            fits = reference_fits(cell, package)
            python, script = args.pythons.get(package), GML_CHILD
        if args.smoke:
            fits = [{**f, "draws": f["draws"][:1]} for f in fits]
        if not fits:
            continue
        cap_draw = cell.cap_draw_s
        draws = sum(len(f["draws"]) for f in fits)
        request = {
            "mode": "fit",
            "X": X,
            "env": env,
            "known": known,
            "fits": json.dumps(fits),
            "warmup_rows": max(50, 2 * X.shape[1]),
        }
        lines, killed, code, err = run_child(
            python, script, request, (draws + 1) * cap_draw + 300, cpu, args.workdir
        )
        found = {(x["algorithm"], x["config"], x["draw"]): x for x in lines}
        for fit in fits:
            for draw in fit["draws"]:
                line = found.get((fit["algorithm"], fit["config"], draw))
                record = {
                    "cell": cell.name,
                    "block": cell.block,
                    "arm": cell.arm,
                    "claim": cell.claim,
                    "seed": seed,
                    "package": package,
                    "algorithm": fit["algorithm"],
                    "config": fit["config"],
                    "fit": {k: v for k, v in fit.items() if k != "draws"},
                    "draw": draw,
                    "cap_draw_s": cap_draw,
                    "machine_id": args.machine_id,
                }
                if line is None:
                    record.update(
                        status="timeout" if killed else "error",
                        error_text=None if killed else f"exit {code}: {err}",
                    )
                elif line["status"] == "ok" and line["wall_s"] > cap_draw:
                    record.update({**line, "status": "timeout"})
                    record.pop("marks", None)
                    record.pop("arcs", None)
                else:
                    record.update(line)
                if record["status"] == "ok":
                    record.update(score(record, package, truth))
                records.append(record)
    return truth_line, records


def run(args) -> int:
    cells = [c for c in grid() if c.arm == args.arm]
    if args.block:
        cells = [replace(c, block=args.block) for c in cells if c.claim == "in" or args.arm == "Q"]
    if args.cells:
        wanted = set(args.cells.split(","))
        cells = [c for c in cells if c.name in wanted]
        if missing := wanted - {c.name for c in cells}:
            raise SystemExit(f"no cells {sorted(missing)} in arm {args.arm}")
    if not cells:
        raise SystemExit(f"arm {args.arm} has no cells")
    packages = args.packages.split(",")
    args.andrey_python = os.environ.get("ANDREY_BENCH_PYTHON") or sys.executable
    args.pythons = {
        "reference": os.environ.get(REFERENCE_PYTHON),
        "def42": os.environ.get(DEF42_PYTHON),
    }
    for package, variable in (("reference", REFERENCE_PYTHON), ("def42", DEF42_PYTHON)):
        needed = package in packages and any(reference_fits(c, package) for c in cells)
        if needed and not args.pythons[package]:
            raise SystemExit(f"set {variable} to the {package} interpreter (start_utigsp.sh)")
    units = [
        (cell, seed)
        for cell in cells
        for seed in (cell.seeds[:1] if args.smoke else cell.seeds)
        if args.seeds is None or seed in args.seeds
    ]

    out = Path(args.out)
    if (out / RECORDS).exists() and not args.resume:
        raise SystemExit(f"{out} already holds a run; pass --resume to continue it")
    out.mkdir(parents=True, exist_ok=True)
    extra = json.loads(Path(args.provenance_extra).read_text()) if args.provenance_extra else {}
    run_meta = {
        "driver": Path(__file__).name,
        "sampler": SAMPLER,
        "arm": args.arm,
        "cells": [asdict(c) | {"name": c.name, "regime": c.group} for c in cells],
        "seeds": sorted({s for _, s in units}),
        "packages": packages,
        "step": args.step,
        "winner": args.winner,
        "andrey_fits": {c.name: andrey_fits(c, args.step, args.winner) for c in cells},
        "reference_fits": {c.name: reference_fits(c, "reference") for c in cells},
        "def42_fits": {c.name: reference_fits(c, "def42") for c in cells},
        "andrey_defaults": andrey_defaults(),
        "smoke": args.smoke,
        "workers": args.workers,
        "machine_id": args.machine_id,
        "extra": extra,
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

    venvs = {
        name: Path(python).absolute().parent.parent
        for name, python in (
            ("utigsp", args.pythons["reference"]),
            ("gml-def42", args.pythons["def42"]),
        )
        if python
    }
    prov = rundir.provenance_once(
        out, timestamp=datetime.now(timezone.utc).isoformat(), extra_venvs=venvs
    )
    _log(f"=== IGSP lane, arm {args.arm}, {len(units)} datasets -> {out} (git {prov['git_sha']})")

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
    skipped = []

    with tempfile.TemporaryDirectory(prefix="igsp_") as tmp:
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
            finally:
                with lock:
                    free.append(cpu)
            with lock:
                with (out / DATASETS).open("a") as sink:
                    sink.write(json.dumps(truth_line) + "\n")
                with (out / RECORDS).open("a") as sink:
                    sink.writelines(json.dumps(r) + "\n" for r in records)
                ok = sum(r["status"] == "ok" for r in records)
                _log(
                    f"[{cell.name} seed {seed}] {ok}/{len(records)} ok | {time.monotonic() - start:.0f}s"
                )

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(task, todo))

    records = _read(out / RECORDS)
    if skipped:
        _log(f"deadline: {len(skipped)} datasets left for --resume")
        return 3
    counts = {s: sum(r["status"] == s for r in records) for s in ("ok", "timeout", "error")}
    (out / DONE).write_text(
        json.dumps({"datasets": len(units), "records": len(records), **counts}, indent=2)
    )
    _log(f"done: {counts}")
    return 0


# ------------------------------------------------------------------------------------------------
# summarize and select
# ------------------------------------------------------------------------------------------------

#: Every per-dataset mean :func:`per_dataset` keeps.
METRICS = (
    "shd",
    "shd_nominal",
    "skeleton_shd",
    "orientation_errors",
    "iess_exact",
    "empty_shd",
    "null_targets",
    "null_arcs",
    "null_reversed",
    "null_kept",
    "dag_hamming",
)


def per_dataset(records: list[dict]) -> dict[tuple, dict]:
    """Each (cell, seed, package, algorithm, config): whether every draw finished, the mean over
    its draws of each metric and of its time, and the distinct answers over its draws."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault(
            (r["cell"], r["seed"], r["package"], r["algorithm"], r["config"]), []
        ).append(r)
    out = {}
    for key, rows in groups.items():
        ok = [r for r in rows if r["status"] == "ok"]
        entry = {
            "complete": len(ok) == len(rows),
            "draws": len(rows),
            "timeouts": len(rows) - len(ok),
        }
        if entry["complete"]:
            for metric in METRICS:
                values = [r.get(metric) for r in ok]
                entry[metric] = None if None in values else float(np.mean(values))
            entry["wall_s"] = float(np.mean([r["wall_s"] for r in ok]))
            entry["distinct"] = len({json.dumps(r.get("marks") or r.get("arcs")) for r in ok})
        out[key] = entry
    return out


def _arms(records: list[dict]) -> dict[tuple, dict[int, dict]]:
    """:func:`per_dataset` keyed by (cell, package, algorithm, config), then by seed."""
    arms: dict[tuple, dict[int, dict]] = {}
    for (cell, seed, package, algorithm, config), entry in per_dataset(records).items():
        arms.setdefault((cell, package, algorithm, config), {})[seed] = entry
    return arms


#: The gated comparators by package, and the context arms (each at Andrey's default level).
GATED = {"reference": "graphical-model-learning", "def42": "graphical-model-learning+def42"}
CONTEXT = {
    ("andrey", "gies", "default"): "andrey.gies",
    ("andrey", "grasp", "observational"): "andrey.grasp@observational-rows",
    ("andrey", "ut_igsp", "known_targets"): "andrey.ut_igsp@known-targets",
    (
        "reference",
        "ut_igsp",
        "known_targets",
    ): "graphical-model-learning.unknown_target_igsp@known-targets",
}
#: Context means reported beside each row, per side.
REPORTED = (
    "shd_nominal",
    "skeleton_shd",
    "orientation_errors",
    "iess_exact",
    "null_reversed",
    "null_kept",
    "null_arcs",
    "dag_hamming",
)


def _means(arm: dict[int, dict], seeds: list[int]) -> dict:
    done = [arm[s] for s in seeds if arm.get(s, {}).get("complete")]
    out = {}
    for metric in REPORTED:
        values = [e[metric] for e in done if e.get(metric) is not None]
        out[metric] = float(np.mean(values)) if values else None
    return out


def _row(cell: Cell, mine: dict, theirs: dict) -> dict:
    """One summary row: the paired comparison on ``shd``, its coverage, and the context means."""
    row: dict = {"mean_degree": cell.degree, "d": cell.p, "n": cell.n}
    row.update(compare(mine, theirs, "shd", cell.datasets))
    row["coverage"]["reference_short"] = sum(
        1 for s, e in mine.items() if e["complete"] and not theirs.get(s, {}).get("complete")
    )
    row["excess_timeouts"] = row["coverage"]["andrey_short"] - row["coverage"]["reference_short"]
    seeds = sorted(set(mine) | set(theirs))
    row["context"] = {"andrey": _means(mine, seeds), "comparator": _means(theirs, seeds)}
    return row


def summarize(records: list[dict], replications: list[dict] | None = None) -> dict:
    """The summary entries of finished runs (design 4a-4c).

    One entry per (regime, comparator): a gated comparator (the reference with 5 runs, and its
    copy with Definition 4.2 in the cells with a multi-target setting) at each level against
    Andrey at each of :func:`judged_starts` (J1 at 5 starts, J2 at the shipped ``n_starts``; the
    regime ends in ``n_starts=<starts>``), the reference at Andrey's shipped number of runs
    against Andrey's J2 (context), and each other context arm against Andrey's shipped defaults;
    one row per (``mean_degree``, ``d``, ``n``). A gated row whose reference does not beat the
    empty graph (the interval of the reference's ``shd`` minus the empty graph's not wholly below
    0) is marked ``claim: "out"`` with that reason. A gated row that passes this filter carries
    ``floor_95``, Andrey against the empty graph, except the unpatched reference's rows in
    multi-target cells, whose floor is the patched copy's row. ``replications`` are the records
    of block C's reruns.
    """
    reruns = _arms(replications or [])
    arms = _arms(records)
    cells = {c.name: c for c in grid()}
    defaults = andrey_defaults()
    mine_default = config_name(defaults["alpha"], defaults["n_starts"])
    entries: dict[tuple, dict] = {}

    def entry(regime: str, comparator: str, cell: Cell, context: bool) -> dict:
        return entries.setdefault(
            (regime, comparator),
            {
                "method": "igsp",
                "comparator": comparator,
                "regime": regime,
                "metric": "shd",
                "claim": cell.claim,
                **({"role": "context"} if context else {}),
                "rows": [],
            },
        )

    for (name, package, algorithm, config), theirs in sorted(arms.items()):
        cell = cells.get(name)
        if cell is None or (package == "andrey" and algorithm == "igsp"):
            continue
        if algorithm == "igsp":
            level, runs = config[1:].split("_r")
            alpha, runs = float(level), int(runs)
            if runs != NRUNS:
                mine = arms.get((name, "andrey", "igsp", config_name(alpha, runs)))
                if mine is not None:
                    regime = f"{cell.group}|alpha={alpha_text(alpha)}|n_starts={runs}"
                    comparator = f"{GATED[package]}@nruns={runs}"
                    entry(regime, comparator, cell, context=True)["rows"].append(
                        _row(cell, mine, theirs)
                    )
                continue
            for starts in judged_starts():
                mine = arms.get((name, "andrey", "igsp", config_name(alpha, starts)))
                if mine is None:
                    continue
                row = _row(cell, mine, theirs)
                reference_floor = floor_interval(theirs, "shd", "empty_shd")
                row["reference_floor_95"] = reference_floor
                if not any(e["complete"] for e in theirs.values()):
                    row.update(claim="out", reason="reference finished no dataset")
                elif reference_floor is None or reference_floor[1] >= 0:
                    row.update(claim="out", reason="reference not better than the empty graph")
                elif package == "def42" or not cell.multi_target:
                    row["floor_95"] = floor_interval(mine, "shd", "empty_shd")
                rerun_mine = reruns.get((name, "andrey", "igsp", config_name(alpha, starts)))
                rerun_theirs = reruns.get((name, package, algorithm, config))
                if rerun_mine and rerun_theirs:
                    rerun = replication(rerun_mine, rerun_theirs, "shd", cell.datasets)
                    if rerun is not None:
                        row["replication"] = rerun
                regime = f"{cell.group}|alpha={alpha_text(alpha)}|n_starts={starts}"
                entry(regime, GATED[package], cell, context=False)["rows"].append(row)
        elif (package, algorithm, config) in CONTEXT:
            mine = arms.get((name, "andrey", "igsp", mine_default))
            if mine is None:
                continue
            comparator = CONTEXT[package, algorithm, config]
            entry(cell.group, comparator, cell, context=True)["rows"].append(
                _row(cell, mine, theirs)
            )
    rows = [
        (r, r.get("claim", e["claim"]))
        for e in entries.values()
        if "role" not in e
        for r in e["rows"]
    ]
    judged = [r for r, claim in rows if claim == "in"]
    return {
        "parity_rule": "paired bootstrap",
        "andrey_defaults": defaults,
        "methods": list(entries.values()),
        "counts": {
            verdict: sum(r.get("verdict") == verdict for r in judged)
            for verdict in ("better", "tie", "worse")
        },
        "filtered": sum(r.get("reason") is not None for r, _ in rows),
    }


def _verdicts(arms: dict, cells: list[str], configs: list[str]) -> dict:
    """Each candidate's paired verdict against every other one, cell by cell, on ``shd``."""
    datasets = {c.name: c.datasets for c in grid()}
    table = {}
    for a, b in itertools.permutations(configs, 2):
        for cell in cells:
            mine, theirs = arms.get((cell, a)), arms.get((cell, b))
            if mine is None or theirs is None:
                raise ValueError(f"cell {cell} lacks {a if mine is None else b}")
            row = compare(mine, theirs, "shd", datasets[cell])
            table[a, b, cell] = {
                "verdict": row.get("verdict", "unpaired"),
                "mean_D": row.get("multi_draw", {}).get("mean_D"),
                "bootstrap_95": row.get("multi_draw", {}).get("bootstrap_95"),
                "paired": row["paired"],
            }
    return table


def _choose(table: dict, configs: list[str], cells: list[str], speed: dict[str, float]):
    """Section 4d's rule: the candidate never worse and better somewhere; else the most cell wins
    minus losses; speed breaks a tie only when the tied candidates tie each other everywhere.
    Returns the winner (``None`` when the rule leaves a tie) and each candidate's counts."""
    counts = {
        a: {
            "wins": sum(
                table[a, b, c]["verdict"] == "better" for b in configs if b != a for c in cells
            ),
            "losses": sum(
                table[a, b, c]["verdict"] == "worse" for b in configs if b != a for c in cells
            ),
        }
        for a in configs
    }
    for c in counts.values():
        c["eligible"] = c["losses"] == 0 and c["wins"] > 0
    pool = [a for a in configs if counts[a]["eligible"]] or list(configs)
    best = max(counts[a]["wins"] - counts[a]["losses"] for a in pool)
    top = [a for a in pool if counts[a]["wins"] - counts[a]["losses"] == best]
    if len(top) == 1:
        return top[0], counts
    if all(table[a, b, c]["verdict"] == "tie" for a in top for b in top if a != b for c in cells):
        return min(top, key=lambda a: speed[a]), counts
    return None, counts


def select(records: list[dict], step: int) -> dict:
    """Andrey's defaults (design 4d) from blocks A and Q.

    Step 1 chooses ``alpha`` among :data:`CANDIDATES` at 5 starts over the in-claim cells of block
    A, then checks it on Q: confirmed when no candidate's paired interval against it lies wholly
    below 0 in either Q cell. Step 2 chooses ``n_starts`` among 1, 5, and 10 at that ``alpha`` by
    the same rule, 5 unless another is eligible. ``escalate`` is true when the rule leaves a tie or
    the confirmation fails: both tables then go to the user before the gate.
    """
    andrey = [r for r in records if r["package"] == "andrey" and r["algorithm"] == "igsp"]
    arms: dict[tuple[str, str], dict[int, dict]] = {}
    for (cell, seed, _, _, config), entry in per_dataset(andrey).items():
        arms.setdefault((cell, config), {})[seed] = entry
    in_claim = sorted({r["cell"] for r in andrey if r["block"] == "A" and r["claim"] == "in"})
    confirm = sorted({r["cell"] for r in andrey if r["arm"] == "Q"})
    present = sorted({config for _, config in arms})
    if step == 1:
        configs = [config_name(a, 5) for a in CANDIDATES if config_name(a, 5) in present]
    else:
        alphas = {c.split("_")[0] for c in present if not c.endswith("_s5")}
        if len(alphas) != 1:
            raise ValueError(f"step 2 needs one alpha at 1 and 10 starts, found {sorted(alphas)}")
        alpha = alphas.pop()
        configs = [f"{alpha}_s{s}" for s in STARTS]
    table = _verdicts(arms, in_claim, configs)
    speed = {
        a: statistics.median(
            e["wall_s"] for c in in_claim for e in arms[(c, a)].values() if e["complete"]
        )
        for a in configs
    }
    winner, counts = _choose(table, configs, in_claim, speed)
    if step == 2 and not any(c["eligible"] for c in counts.values()):
        winner = next(c for c in configs if c.endswith("_s5"))
    result = {
        "step": step,
        "cells": in_claim,
        "candidates": configs,
        "winner": winner,
        "counts": counts,
        "median_wall_s": speed,
        "coverage": {
            a: sum(e["complete"] for c in in_claim for e in arms[(c, a)].values()) for a in configs
        },
        "pairs": {f"{a} vs {b} | {c}": v for (a, b, c), v in table.items()},
    }
    escalate = winner is None
    if step == 1 and confirm and winner is not None:
        q = _verdicts(arms, confirm, configs)
        beaten = {
            c: [b for b in configs if b != winner and q[b, winner, c]["verdict"] == "better"]
            for c in confirm
        }
        result["confirmation"] = {
            "cells": confirm,
            "beaten_by": beaten,
            "confirmed": not any(beaten.values()),
            "pairs": {f"{a} vs {b} | {c}": v for (a, b, c), v in q.items() if b == winner},
        }
        escalate = escalate or not result["confirmation"]["confirmed"]
    result["escalate"] = escalate
    return result


# ------------------------------------------------------------------------------------------------
# validate-patch
# ------------------------------------------------------------------------------------------------


@functools.cache
def oracle_case(seed: int) -> tuple[int, list[list[int]], list[list[int]]]:
    """``(p, arcs, family)``: the 200 oracle cases of ``tests/unit/test_search_igsp.py``.

    A random DAG on 4-7 variables with mean degree 1.5 (even seeds) or 3 (odd ones), and 1-4
    settings of one target each, of 2-3 each, or of 1-3 each, in turn by ``seed // 2``.
    """
    rng = np.random.default_rng([seed, 655])
    p = int(rng.integers(4, 8))
    probability = min(1.0, (1.5, 3.0)[seed % 2] / (p - 1))
    order = rng.permutation(p)
    arcs = [
        [int(order[a]), int(order[b])]
        for a, b in itertools.combinations(range(p), 2)
        if rng.random() < probability
    ]
    sizes = ((1, 2), (2, 4), (1, 4))[(seed // 2) % 3]
    family = [
        sorted(int(v) for v in rng.choice(p, size=int(rng.integers(*sizes)), replace=False))
        for _ in range(int(rng.integers(1, 5)))
    ]
    return p, arcs, family


def patch_queries(p: int, arcs: list[list[int]], family: list[list[int]]) -> list[dict]:
    """The predicate checks of one case: no arc of the true DAG is I-contradicting (``expect``
    false), and the reversal of every covered arc that changes the class is (``expect`` true, in
    the reversed DAG: Lemma B.5 with one step)."""
    parents = {j: {i for i, jj in arcs if jj == j} for j in range(p)}
    truth = essential(dag_from_arcs(arcs, p), family)
    queries = [{"arcs": arcs, "arc": arc, "expect": False} for arc in arcs]
    for i, j in arcs:
        if parents[j] != parents[i] | {i}:
            continue
        flipped = [a for a in arcs if a != [i, j]] + [[j, i]]
        if not np.array_equal(essential(dag_from_arcs(flipped, p), family), truth):
            queries.append({"arcs": flipped, "arc": [j, i], "expect": True})
    return queries


def andrey_oracle_found(p: int, arcs: list[list[int]], family: list[list[int]], seed: int):
    """The arcs of Andrey's engine (``n_starts=5``) with every test answered by d-separation."""
    from andrey.core import ARROW, TAIL
    from andrey.core.structure import GraphStructure
    from andrey.search import igsp

    K = len(family)
    augmented = np.zeros((p + K, p + K), dtype=np.int8)
    for i, j in [*arcs, *([p + k, v] for k, t in enumerate(family) for v in t)]:
        augmented[i, j], augmented[j, i] = TAIL, ARROW
    graph = GraphStructure.from_numpy(augmented, kind="dag")
    contexts = list(range(p, p + K))

    @functools.cache
    def independent(i, j, S):
        return graph.d_separated(i, j, [*S, *contexts])

    @functools.cache
    def invariant(j, S, k):
        return graph.d_separated(p + k, j, [*S, *(c for c in contexts if c != p + k)])

    rules = igsp.igsp_rules(invariant, [frozenset(t) for t in family])
    found, _ = igsp.igsp_search(p, independent, rules, n_starts=5, seed=seed)
    return [[i, j] for j, pa in enumerate(found) for i in range(p) if pa >> i & 1]


def probe(kind: str, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """The ``{0, 2}`` probe: ``0 -> 1`` with weight 0.8, ``2`` isolated, 2,000 rows observational
    and 2,000 with ``0`` and ``2`` intervened (Wang's hard intervention or a shift of 1), columns
    in node order. Returns the datasets stacked and their ``env``."""
    import andrey.data as data

    scm = data.SCM(
        graph=data.graphs.FixedDAG(nodes=("0", "1", "2"), arcs=(("0", "1"),)),
        functional=data.functional.linear(weight_range=(0.8, 0.8)),
        noise=data.noise.gaussian(),
    )
    spec = _intervention("wang" if kind == "hard" else "shift")
    Xs, env = [], None
    for seed in seeds:
        ds = scm.sample(
            n=[2000, 2000], seed=seed, scale="raw", shuffle=False, envs=[{}, {0: spec, 2: spec}]
        )
        Xs.append(np.asarray(ds.data, dtype=np.float64))
        env = np.asarray(ds.env, dtype=np.int64)
    return np.stack(Xs), env


def validate_patch(out: Path, cases: int = 200, seeds: int = 5) -> dict:
    """Design 1b's checks of the patched reference before any fit, under oracle tests: (i) no arc
    of the true DAG is I-contradicting, (ii) every covered arc whose reversal changes the class
    is, in the reversed DAG, (iii) the ``{0, 2}`` probe returns ``0 -> 1`` on 20 of 20 seeds, hard
    and shift. Also counts how often the patched package and Andrey's engine return the true
    class under the oracle (reported, not checked). Writes ``patch-validation.json``."""
    pythons = {"def42": os.environ.get(DEF42_PYTHON), "reference": os.environ.get(REFERENCE_PYTHON)}
    if not all(pythons.values()):
        raise SystemExit(f"set {DEF42_PYTHON} and {REFERENCE_PYTHON} (start_utigsp.sh)")
    payload = []
    for seed in range(cases):
        p, arcs, family = oracle_case(seed)
        payload.append(
            {
                "p": p,
                "arcs": arcs,
                "family": family,
                "queries": patch_queries(p, arcs, family),
                "seeds": list(range(seeds)),
            }
        )
    result: dict = {"cases": cases, "seeds": seeds}
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="igsp_patch_") as tmp:
        for package, python in pythons.items():
            lines, killed, code, err = run_child(
                python,
                GML_CHILD,
                {"mode": "oracle", "cases": json.dumps(payload)},
                3600,
                None,
                Path(tmp),
            )
            if killed or len(lines) != cases:
                raise SystemExit(
                    f"the {package} child answered {len(lines)} of {cases} (exit {code}): {err}"
                )
            wrong, recovered = [], 0
            for seed, (case, line) in enumerate(zip(payload, lines, strict=True)):
                for query, answer in zip(case["queries"], line["contradicting"], strict=True):
                    if answer != query["expect"]:
                        wrong.append({"case": seed, "arc": query["arc"], "expect": query["expect"]})
                truth = essential(dag_from_arcs(case["arcs"], case["p"]), case["family"])
                recovered += sum(
                    np.array_equal(essential(dag_from_arcs(f, case["p"]), case["family"]), truth)
                    for f in line["found"]
                )
            result[package] = {
                "queries": sum(len(c["queries"]) for c in payload),
                "wrong": wrong,
                "recovered": int(recovered),
            }
            result[package]["probe"] = {}
            for kind in ("hard", "shift"):
                Xs, env = probe(kind, list(range(20)))
                request = {
                    "mode": "probe",
                    "Xs": Xs,
                    "env": env,
                    "known": json.dumps([[0, 2]]),
                    "seeds": json.dumps(list(range(20))),
                    "fits": json.dumps(
                        [{"algorithm": "igsp", "alpha": 1e-3, "nruns": NRUNS, "tester": "gauss"}]
                    ),
                }
                lines, killed, code, err = run_child(
                    python, GML_CHILD, request, 1800, None, Path(tmp)
                )
                if killed or len(lines) != 20:
                    raise SystemExit(
                        f"the {package} probe answered {len(lines)} of 20 (exit {code}): {err}"
                    )
                result[package]["probe"][kind] = {
                    "0->1": sum([0, 1] in line["arcs"] for line in lines),
                    "1->0": sum([1, 0] in line["arcs"] for line in lines),
                }
    andrey_recovered = 0
    for seed in range(cases):
        p, arcs, family = oracle_case(seed)
        truth = essential(dag_from_arcs(arcs, p), family)
        andrey_recovered += sum(
            np.array_equal(
                essential(dag_from_arcs(andrey_oracle_found(p, arcs, family, s), p), family), truth
            )
            for s in range(seeds)
        )
    result["andrey"] = {"recovered": int(andrey_recovered)}
    result["passed"] = not result["def42"]["wrong"] and all(
        v["0->1"] == 20 for v in result["def42"]["probe"].values()
    )
    (out / "patch-validation.json").write_text(json.dumps(result, indent=2))
    return result


def _records(dirs: list[str]) -> list[dict]:
    records = []
    for d in dirs:
        records += _read(Path(d) / RECORDS)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IGSP, Andrey against the authors' library, CPU")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="fit one arm's cells")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument("--arm", required=True, help="W, S, Y6, Y5, Y8, Q, K, or O1-O9")
    p.add_argument("--cells", default=None, help="only these cells of the arm, comma-separated")
    p.add_argument(
        "--seeds", type=parse_seeds, default=None, help='only these seeds, e.g. "1000-1009"'
    )
    p.add_argument("--packages", default="andrey,reference,def42", help="andrey, reference, def42")
    p.add_argument(
        "--block",
        choices=sorted(BLOCKS),
        default=None,
        help="the in-claim cells on this block's seeds",
    )
    p.add_argument("--step", type=int, default=1, choices=(1, 2), help="blocks A and Q: the step")
    p.add_argument("--winner", default=None, help="step 2: the step-1 winner, a<alpha>_s5")
    p.add_argument("--workers", type=int, default=1, help="datasets fitted at once, one core each")
    p.add_argument(
        "--deadline-s", type=float, default=None, help="start no dataset after this many seconds"
    )
    p.add_argument("--machine-id", default=socket.gethostname(), help="stamped on every record")
    p.add_argument("--provenance-extra", default=None, help="a JSON file recorded in run_meta.json")
    p.add_argument(
        "--smoke", action="store_true", help="the first dataset of each cell, one draw per fit"
    )
    p.add_argument("--resume", action="store_true", help="continue the run already in --out")
    s = sub.add_parser("summarize", help="summary entries of finished runs")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--replications", nargs="+", default=[], help="block C's runs")
    s.add_argument("--out", required=True)
    c = sub.add_parser("select", help="Andrey's defaults from blocks A and Q")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--step", type=int, default=1, choices=(1, 2))
    c.add_argument("--out", required=True)
    v = sub.add_parser("validate-patch", help="check the patched reference under oracle tests")
    v.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    if args.command == "summarize":
        summary = summarize(_records(args.runs), _records(args.replications))
        Path(args.out).write_text(json.dumps(summary, indent=2))
        _log(f"{len(summary['methods'])} entries -> {args.out}; judged rows {summary['counts']}")
        return 0
    if args.command == "validate-patch":
        result = validate_patch(Path(args.out))
        _log(json.dumps({k: v for k, v in result.items() if k not in ("def42", "reference")}))
        _log(f"def42 wrong {len(result['def42']['wrong'])}, probe {result['def42']['probe']}")
        return 0 if result["passed"] else 1
    result = select(_records(args.runs), args.step)
    Path(args.out).write_text(json.dumps(result, indent=2))
    _log(f"winner {result['winner']}; escalate {result['escalate']}")
    for config, counts in result["counts"].items():
        _log(f"  {config:16s} {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
