#!/usr/bin/env python
"""Benchmark GnIES on CPU: Andrey against gnies 0.3.3, on noise-intervention data.

Each cell of :data:`CELLS` is a regime of Gamella et al. (2022, Sec. 5.1 and App. H): a linear
SCM with Gaussian noise on ``d`` variables, an Erdos-Renyi DAG at a mean degree, and environments
that each set the noise variance of a few targets. :func:`draw` samples a cell's dataset for a seed
with ``andrey.data``: weights ``U[0.5, 1]``, positive as the paper's code draws them
(:class:`PaperLinear`), noise variances ``U[1, 2]`` (:class:`PaperGaussian`), and a target's
variance in its environment set to ``v`` from the cell's range. The S cells use the stock signed
weights and unit noise, and set the ratio ``v / omega`` with ``omega ~ U[1, 2]``. Targets and
variances come from ``default_rng([d, cell_id, seed])``, so cells do not share them; every
environment draws its own noise. The rows are standardized on the pooled sample, as the paper does.
Cells marked ``claim="out"`` break the model (hard interventions, a mean shift, a weight change,
Laplace noise, additive nonlinear functions) or exceed its scale (``d`` above 10).

Every arm fits the same rows, back to back on one CPU:

- Andrey's ``gnies`` at its defaults (the gated arm), at the standard BIC (``ebic_gamma=0``), with
  the true targets at BIC and at the defaults, and with ties broken by column position
  (:data:`andrey_bench.adapters.andrey_gnies.ARMS`);
- gnies 0.3.3 at every default (``gnies.fit(data)``, the gated reference) and with the true
  targets fixed (``known_targets=I*, phases=[]``, one inner search), in its own interpreter
  (:mod:`andrey_bench.adapters.gnies_gnies`);
- Andrey's ``ges`` on the pooled rows, Andrey's ``gies`` with the true targets, and the empty graph.

A fit past its cap is a ``timeout`` with no scores, never an empty graph. :func:`scores` computes
every metric in this process from the returned marks and targets: ``shd`` against the true
interventional essential graph (the true DAG with every edge at a true target oriented), and
``target_error``, the size of the symmetric difference of the learned and true targets, then target
precision and recall, skeleton and orientation errors, the paper's TDP and FDP over the members of
both classes, and the score gap to the truth in deviance units (lower is better), under one common
scorer (Andrey's GnIES score at the standard BIC) and under the fit's own objective.
:func:`model_check` flags an in-claim SCM that fails a numerical form of the paper's Assumption 2
(and, with several targets per environment, Assumption 1) at its drawn parameters.

One JSON line per (cell, seed, arm) goes to ``<out>/records.jsonl``; ``--resume`` skips the lines
on disk, ``--rescore`` recomputes the scores from the stored marks. ``<out>/provenance.json``
records the builds and the machine.

    ANDREY_BENCH_GNIES_PYTHON=... python benchmarks/gnies_cpu.py --out outputs/gnies --cells A100
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import signal
import socket
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from andrey.core import ARROW, TAIL
from andrey.data.functional import Linear
from andrey.data.noise import Gaussian
from andrey_bench.adapters import andrey_gnies, gnies_gnies


@dataclass(frozen=True, kw_only=True)
class PaperLinear(Linear):
    """Linear equations with positive weights: the absolute value of the stock signed draw."""

    name: str = "linear_positive"

    def draw_weights(self, draw, rng):
        return np.abs(super().draw_weights(draw, rng))


@dataclass(frozen=True, kw_only=True)
class PaperGaussian(Gaussian):
    """Gaussian noise with variances ``U[1, 2]``."""

    name: str = "gaussian_u12"

    def draw_scales(self, d, rng):
        return np.sqrt(rng.uniform(1.0, 2.0, d))


@dataclass(frozen=True)
class Cell:
    """One regime: rows per environment (the observational one first when there is one)."""

    name: str
    regime: str
    sizes: tuple[int, ...]
    claim: str = "in"
    observational: bool = True
    per_env: int = 1
    degree: float = 2.7
    d: int = 10
    variance: tuple[float, float] = (5.0, 10.0)
    weights: str = "positive"  # "positive" or "signed"
    noise: str = "u12"  # "u12", "unit", or "laplace"
    functional: str = "linear"  # "linear" or "additive"
    intervention: str = "noise"  # "noise", "hard", "shift", or "weight"

    @property
    def n(self) -> int:
        """Rows in the smallest environment."""
        return min(self.sizes)


def _cells() -> tuple[Cell, ...]:
    cells = []
    for regime, changes in (("A", {}), ("B", {"degree": 4.0}), ("C", {"per_env": 3})):
        cells += [Cell(f"{regime}{n}", regime, (n,) * 4, **changes) for n in (10, 100, 1000)]
    cells += [Cell(f"D{n}", "D", (n,) * 4, variance=(3.0, 4.0)) for n in (10, 100, 1000)]
    cells += [Cell(f"E{n}", "E", (n,) * 4, observational=False) for n in (100, 1000)]
    cells += [Cell(f"F{n}", "F", (n,) * 10) for n in (10, 100)]
    cells += [Cell("G", "G", (1000, 100, 100, 100))]
    cells += [Cell(f"H{k}", f"H{k}", (1000,) * (k + 1)) for k in (1, 2)]
    cells += [Cell(f"S{n}", "S", (n,) * 4, weights="signed", noise="unit") for n in (100, 1000)]
    out = {"claim": "out"}
    cells += [Cell(f"O1-{n}", "O1", (n,) * 4, intervention="hard", **out) for n in (100, 1000)]
    cells += [
        Cell("O2", "O2", (1000,) * 4, intervention="shift", **out),
        Cell("O3", "O3", (1000,) * 4, intervention="weight", **out),
        Cell("O4", "O4", (1000,) * 4, noise="laplace", **out),
        Cell("O5", "O5", (1000,) * 4, noise="unit", functional="additive", **out),
    ]
    cells += [Cell(f"O6-{d}", "O6", (1000,) * 4, d=d, **out) for d in (20, 30, 50, 100)]
    return tuple(cells)


#: Every cell, in-claim first; a cell's position is its ``cell_id`` in the target draws.
CELLS = _cells()
BY_NAME = {cell.name: cell for cell in CELLS}

#: The reference's gated call and its known-target ceiling.
REFERENCE, CEILING = "gnies.fit", "gnies.fit.known"
#: Every arm, in the order a seed runs them.
ARMS = (*andrey_gnies.ARMS, REFERENCE, CEILING, "andrey.ges", "andrey.gies.true_targets", "empty")

#: Classes with more members than this get no TDP or FDP, and SCMs no Assumption 2 check.
MEMBER_CAP = 5000

#: An SCM is flagged when a diagonal entry of ``M`` is this small, or two target ratios this close.
MODEL_TOL = 1e-6

#: Smallest run that touches every arm: two small datasets.
SMOKE = {"cells": ["A100", "C10"], "seeds": [0], "cap_wall_s": 300.0, "cap_ceiling_s": 300.0}

RECORDS = "records.jsonl"


@dataclass
class Draw:
    """One dataset: the standardized rows, their environment codes, and the truth.

    ``targets[k]`` lists environment ``k``'s targets; ``dag`` holds the true DAG's endpoint marks,
    ``weights[i, j]`` the weight of ``i -> j``, and ``omegas[k, j]`` variable ``j``'s noise
    variance in environment ``k`` (noise interventions on a linear model only, else ``None``).
    """

    cell: Cell
    seed: int
    X: np.ndarray
    env: np.ndarray
    targets: list[list[int]]
    dag: np.ndarray
    weights: np.ndarray
    omegas: np.ndarray | None
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self.sha256 = hashlib.sha256(np.ascontiguousarray(self.X, np.float64).tobytes()).hexdigest()

    @property
    def true_targets(self) -> list[int]:
        return sorted({j for t in self.targets for j in t})


def _scm(cell: Cell):
    from andrey.data import SCM, functional, graphs, noise

    if cell.functional == "additive":
        equations = functional.additive_noise(weight_range=(0.5, 1.0))
    elif cell.weights == "signed":
        equations = functional.linear(weight_range=(0.5, 1.0))
    else:
        equations = PaperLinear(weight_range=(0.5, 1.0))
    errors = {"u12": PaperGaussian(), "unit": noise.gaussian(), "laplace": noise.laplace()}
    return SCM(
        graph=graphs.erdos_renyi(cell.d, avg_degree=cell.degree),
        functional=equations,
        noise=errors[cell.noise],
    )


def draw(cell: Cell, seed: int, cell_id: int | None = None) -> Draw:
    """Sample one dataset of ``cell``; the same ``(cell, seed)`` gives the same bytes.

    ``cell_id`` keys the target draws; it defaults to the cell's position in :data:`CELLS`, so a
    cell outside that list (as ``qa/gnies_order.py`` builds) passes its own.
    """
    from andrey.data.interventions import hard, soft

    scm = _scm(cell)
    # The noise scales are the first draw of the noise stream, so they do not depend on n.
    sigma = scm.sample(n=10, seed=seed, scale="raw").params.noise_scales
    cell_id = CELLS.index(cell) if cell_id is None else cell_id
    rng = np.random.default_rng([cell.d, cell_id, seed])
    k = len(cell.sizes) - cell.observational
    chosen = rng.choice(cell.d, size=(k, cell.per_env), replace=False)
    envs: list[dict] = [{}] if cell.observational else []
    for row in chosen:
        env = {}
        for j in (int(j) for j in row):
            v = rng.uniform(*cell.variance)
            if cell.intervention == "hard":
                env[j] = hard(0.0, sd=float(np.sqrt(v)))
            elif cell.intervention == "shift":
                env[j] = soft(shift=2.0)
            elif cell.intervention == "weight":
                env[j] = soft(weight_scale=0.5)
            elif cell.noise == "u12":
                env[j] = soft(noise_scale=float(np.sqrt(v) / sigma[j]))
            else:
                env[j] = soft(noise_scale=float(np.sqrt(v / rng.uniform(1.0, 2.0))))
        envs.append(env)
    scale = "standardize" if cell.intervention == "shift" else "raw"
    ds = scm.sample(n=list(cell.sizes), seed=seed, envs=envs, scale=scale)
    X = (ds.data - ds.data.mean(axis=0)) / ds.data.std(axis=0)
    weights = np.zeros((cell.d, cell.d))
    edges = np.asarray(ds.params.edges)
    if cell.functional == "linear":
        weights[edges[:, 0], edges[:, 1]] = ds.params.weights
    omegas = None
    if cell.intervention == "noise" and cell.functional == "linear":
        omegas = np.tile(np.asarray(ds.params.noise_scales) ** 2, (len(envs), 1))
        for code, env in enumerate(envs):
            for j, intervention in env.items():
                omegas[code, j] *= intervention.noise_scale**2
    targets = [sorted(int(j) for j in env) for env in envs]
    assert ds.targets is not None and all(sorted(ds.targets[k]) == targets[k] for k in ds.targets)
    dag = np.asarray(ds.graph.to_numpy(), dtype=np.int8)
    return Draw(cell, seed, X, np.asarray(ds.env, dtype=np.intp), targets, dag, weights, omegas)


def directed(marks: np.ndarray) -> np.ndarray:
    """``A[i, j]`` is True for ``i -> j``."""
    return (marks == TAIL) & (marks.T == ARROW)


def essential_graph(dag: np.ndarray, cut: np.ndarray) -> np.ndarray:
    """The class of ``dag`` (endpoint marks) whose edges in ``cut`` keep their orientation."""
    from andrey.search.gies import replace_unprotected

    return replace_unprotected(dag.copy(), cut)


def class_members(marks: np.ndarray, cap: int = MEMBER_CAP) -> list[np.ndarray] | None:
    """Every DAG of the class ``marks`` draws, as 0/1 matrices with ``A[i, j] = 1`` for ``i -> j``.

    A member keeps every directed edge and orients every undirected one, so that it is acyclic and
    has no v-structure the marks lack. ``None`` when there are more than ``cap``.
    """
    A = directed(marks)
    adjacent = marks != 0
    lines = [
        (int(i), int(j)) for i, j in zip(*np.nonzero(np.triu((marks == TAIL) & (marks.T == TAIL))))
    ]
    members: list[np.ndarray] = []

    def reaches(src: int, dst: int) -> bool:
        stack, seen = [src], {src}
        while stack:
            u = stack.pop()
            if u == dst:
                return True
            for w in np.flatnonzero(A[u]):
                if int(w) not in seen:
                    seen.add(int(w))
                    stack.append(int(w))
        return False

    def extend(k: int) -> None:
        if len(members) > cap:
            return
        if k == len(lines):
            members.append(A.astype(np.int8))
            return
        i, j = lines[k]
        for a, b in ((i, j), (j, i)):
            parents = np.flatnonzero(A[:, b])
            if all(adjacent[a, c] for c in parents) and not reaches(b, a):
                A[a, b] = True
                extend(k + 1)
                A[a, b] = False

    if any(reaches(int(j), int(i)) for i, j in zip(*np.nonzero(A))):
        return []
    extend(0)
    return None if len(members) > cap else members


def discovery(estimate: list[np.ndarray], truth: list[np.ndarray]) -> tuple[float, float]:
    """The paper's ``(TDP, FDP)`` between two classes (Definition 4, ``metrics.py``).

    FDP: over the estimate's members, the largest share of edges absent from the closest true
    member. TDP: one minus, over the true members, the largest share of edges absent from the
    closest estimated member.
    """
    E = np.array([m.ravel() for m in estimate], dtype=float)
    T = np.array([m.ravel() for m in truth], dtype=float)
    both = E @ T.T
    e, t = E.sum(axis=1)[:, None], T.sum(axis=1)[None, :]
    false = np.divide(e - both, e, out=np.zeros_like(both), where=e > 0)
    missed = np.divide(t - both, t, out=np.zeros_like(both), where=t > 0)
    return 1.0 - float(missed.min(axis=0).max()), float(false.min(axis=1).max())


def model_check(data: Draw, members: list[np.ndarray] | None) -> dict[str, Any] | None:
    """The paper's Assumptions 2 (and 1 with several targets per environment), numerically.

    For every member ``G'`` of the true DAG's Markov equivalence class and every environment,
    ``B'`` regresses each node on its ``G'``-parents in the population covariance and
    ``M = (I - B')(I - B)^-1``; the SCM fails when ``min |M_ii|`` is at most :data:`MODEL_TOL`.
    Assumption 1: within an environment, two targets' variance ratios differ by more than
    :data:`MODEL_TOL` relative. ``None`` for a cell out of claim.
    """
    if data.cell.claim != "in" or data.omegas is None:
        return None
    d = data.cell.d
    inv = np.linalg.inv(np.eye(d) - data.weights.T)
    smallest = math.inf
    for omega in data.omegas if members is not None else ():
        sigma = inv @ np.diag(omega) @ inv.T
        for member in members:
            fitted = np.zeros((d, d))
            for i in range(d):
                pa = np.flatnonzero(member[:, i])
                if pa.size:
                    fitted[i, pa] = np.linalg.solve(sigma[np.ix_(pa, pa)], sigma[pa, i])
            smallest = min(smallest, float(np.abs(np.diag((np.eye(d) - fitted) @ inv)).min()))
    distinct = True
    base = data.omegas[0] if data.cell.observational else None
    for code, targets in enumerate(data.targets):
        if len(targets) > 1 and base is not None:
            ratios = sorted(data.omegas[code, targets] / base[targets])
            distinct &= all(b - a > MODEL_TOL * b for a, b in zip(ratios, ratios[1:]))
    passed = members is None or smallest > MODEL_TOL
    return {
        "assumption2_min": None if members is None else smallest,
        "assumption1": distinct,
        "flagged": not (passed and distinct),
    }


class Truth:
    """What every arm of one dataset is scored against, built once."""

    def __init__(self, data: Draw) -> None:
        from andrey.search.gies import cut_pairs
        from andrey.search.gnies import singleton_cut

        d = data.cell.d
        self.data = data
        self.targets = data.true_targets
        self.marks = essential_graph(data.dag, singleton_cut(self.targets, d))
        self.observational = essential_graph(data.dag, np.zeros((d, d), dtype=bool))
        mask = np.zeros((len(data.targets), d), dtype=bool)
        for code, targets in enumerate(data.targets):
            mask[code, targets] = True
        self.hauser_buhlmann = essential_graph(data.dag, cut_pairs(mask))
        self.members = class_members(self.marks)
        self.model = model_check(data, class_members(self.observational))
        self._totals: dict[tuple[float, float], float] = {}

    def total(self, settings: tuple[float, float]) -> float:
        """The true DAG's total under Andrey's score at ``(penalty_discount, ebic_gamma)``."""
        if settings not in self._totals:
            self._totals[settings] = class_total(self.data, self.data.dag, self.targets, settings)
        return self._totals[settings]


def class_total(
    data: Draw, marks: np.ndarray, targets: list[int], settings: tuple[float, float]
) -> float:
    """A class's total under Andrey's GnIES score, through any DAG of the class."""
    from andrey.core.orient import pdag2dag
    from andrey.search.gnies import NoiseInterventionScore

    dag = pdag2dag(np.asarray(marks, dtype=np.int8))
    score = NoiseInterventionScore(
        data.X, data.env, penalty_discount=settings[0], ebic_gamma=settings[1]
    ).view(targets)
    return math.fsum(score.score(j, np.flatnonzero(dag[j] == ARROW)) for j in range(len(dag)))


#: The common scorer of the score gap: Andrey's GnIES score at the standard BIC.
COMMON = (1.0, 0.0)


def scores(marks: np.ndarray, targets: list[int], truth: Truth) -> dict[str, Any]:
    """Every metric of one result against ``truth``; see the module docstring."""
    from andrey import GraphStructure
    from andrey.metrics import shd

    def to_truth(other: np.ndarray) -> int:
        return int(
            shd(
                GraphStructure.from_numpy(marks, kind="cpdag"),
                GraphStructure.from_numpy(other, kind="cpdag"),
            )
        )

    learned, true = set(targets), set(truth.targets)
    found = len(learned & true)
    adjacent, true_adjacent = marks != 0, truth.marks != 0
    shared = np.triu(adjacent & true_adjacent, 1)
    out: dict[str, Any] = {
        "shd": to_truth(truth.marks),
        "target_error": len(learned ^ true),
        "target_precision": found / len(learned) if learned else 1.0,
        "target_recall": found / len(true) if true else 1.0,
        "target_exact": learned == true,
        "class_equal": bool(np.array_equal(marks, truth.marks)) and learned == true,
        "skeleton_shd": int(np.count_nonzero(np.triu(adjacent != true_adjacent, 1))),
        "orientation_errors": int(
            np.count_nonzero(shared & ((marks != truth.marks) | (marks.T != truth.marks.T)))
        ),
        "shd_observational": to_truth(truth.observational),
        "shd_hauser_buhlmann": to_truth(truth.hauser_buhlmann),
        "tdp": None,
        "fdp": None,
        "gap_common": None,
    }
    estimate = class_members(marks)
    if estimate and truth.members:
        out["tdp"], out["fdp"] = discovery(estimate, truth.members)
    with contextlib.suppress(Exception):
        total = class_total(truth.data, marks, sorted(learned), COMMON)
        out["gap_common"] = total - truth.total(COMMON)
    return out


class FitTimeout(Exception):
    """A fit in this process passed its cap."""


@contextlib.contextmanager
def capped(seconds: float):
    """Raise :class:`FitTimeout` in the body after ``seconds`` of wall time."""

    def stop(signum, frame):
        raise FitTimeout

    previous = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run_arm(arm: str, data: Draw, truth: Truth, caps: dict[str, float]) -> dict[str, Any]:
    """Fit one arm and return ``status``, ``marks``, ``targets``, ``seconds``, and its extras."""
    import time

    import andrey

    if arm == "empty":
        d = data.cell.d
        return {"status": "ok", "marks": np.zeros((d, d), np.int8), "targets": [], "seconds": 0.0}
    if arm in (REFERENCE, CEILING):
        fields = {"A_true": directed(data.dag).astype(np.int8), "I_true": np.array(truth.targets)}
        if arm == CEILING:
            fields["I"] = np.array(truth.targets, dtype=int)
        mode = "fit" if arm == REFERENCE else "known"
        answer = gnies_gnies.request(data.X, data.env, mode, cap_s=caps[arm], **fields)
        if answer["status"] == "ok":
            answer["marks"] = gnies_gnies.marks(answer.pop("icpdag"))
        return answer
    try:
        with capped(caps[arm]):
            if arm in andrey_gnies.ARMS:
                return {"status": "ok", **andrey_gnies.fit(data.X, data.env, arm, data.targets)}
            start = time.perf_counter()
            if arm == "andrey.ges":
                out, targets = andrey.ges(data.X), []
            else:
                mapping = {k: t for k, t in enumerate(data.targets)}
                out, targets = andrey.gies(data.X, data.env, targets=mapping), truth.targets
            seconds = time.perf_counter() - start
            marks = np.asarray(out.structure.to_numpy(), dtype=np.int8)
            return {"status": "ok", "marks": marks, "targets": targets, "seconds": seconds}
    except FitTimeout:
        return {"status": "timeout", "error_text": f"past the {caps[arm]:g} s cap"}
    except Exception as exc:  # an error is a result of the arm, recorded and counted
        return {"status": "error", "error_text": f"{type(exc).__name__}: {exc}"}


def own_gap(arm: str, answer: dict[str, Any], truth: Truth) -> float | None:
    """The result's total minus the truth's, under the arm's own objective."""
    if answer.get("deviance") is None:
        return None
    if arm in andrey_gnies.ARMS:
        chosen = andrey_gnies.settings(arm)
        return answer["deviance"] - truth.total((chosen["penalty_discount"], chosen["ebic_gamma"]))
    if answer.get("truth_deviance") is not None:
        return answer["deviance"] - answer["truth_deviance"]
    return None


def record(arm: str, data: Draw, truth: Truth, answer: dict[str, Any], meta: dict) -> dict:
    """One JSON line: the dataset's keys, the fit's status and outputs, and its scores."""
    cell = data.cell
    row: dict[str, Any] = {
        "cell": cell.name,
        "regime": cell.regime,
        "claim": cell.claim,
        "d": cell.d,
        "mean_degree": cell.degree,
        "n": cell.n,
        "sizes": list(cell.sizes),
        "seed": data.seed,
        "arm": arm,
        "package": "gnies" if arm in (REFERENCE, CEILING) else "andrey",
        "package_version": meta["versions"]["gnies" if arm in (REFERENCE, CEILING) else "andrey"],
        "machine_id": meta["machine_id"],
        "data_sha256": data.sha256,
        "true_targets": truth.targets,
        "model_check": truth.model,
        "empty_shd": int(np.count_nonzero(np.triu(truth.marks != 0, 1))),
        "status": answer["status"],
        "error_text": answer.get("error_text"),
        "wall_s": answer.get("seconds"),
        "marks": None,
        "targets": None,
        "deviance": answer.get("deviance"),
        "truth_deviance": answer.get("truth_deviance"),
        "counts": answer.get("counts"),
    }
    if answer["status"] == "ok":
        marks = np.asarray(answer["marks"], dtype=np.int8)
        row["marks"] = marks.tolist()
        row["targets"] = [int(j) for j in answer["targets"]]
        row.update(scores(marks, row["targets"], truth))
        row["gap_own"] = own_gap(arm, answer, truth)
    return row


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _read(out: Path) -> list[dict]:
    path = out / RECORDS
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def rescore(out: Path) -> list[dict]:
    """Recompute every record's scores from its stored marks and rewrite the records."""
    records = _read(out)
    datasets: dict[tuple[str, int], tuple[Draw, Truth]] = {}
    for row in records:
        key = (row["cell"], row["seed"])
        if key not in datasets:
            data = draw(BY_NAME[row["cell"]], row["seed"])
            datasets[key] = (data, Truth(data))
        data, truth = datasets[key]
        if data.sha256 != row["data_sha256"]:
            raise SystemExit(f"--rescore: {key} draws other data than the run recorded")
        if row["marks"] is not None:
            row.update(scores(np.asarray(row["marks"], dtype=np.int8), row["targets"], truth))
    (out / RECORDS).write_text("".join(json.dumps(r) + "\n" for r in records))
    return records


def _warm_up() -> None:
    """One small untimed fit, so the first timed one does not pay for lazy imports."""
    data = draw(BY_NAME["A100"], 0)
    andrey_gnies.fit(data.X[::4], data.env[::4], "andrey.gnies")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GnIES, Andrey against gnies 0.3.3, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--cells",
        nargs="+",
        default=[c.name for c in CELLS if c.claim == "in"],
        help="cell names (default: every in-claim cell)",
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(20)), help="seeds (default: 0-19)"
    )
    p.add_argument("--arms", nargs="+", default=list(ARMS), help="arms (default: every arm)")
    p.add_argument("--cap-wall-s", type=float, default=1800.0, help="cap per fit, seconds")
    p.add_argument(
        "--cap-ceiling-s", type=float, default=300.0, help="cap of the reference's ceiling call"
    )
    p.add_argument("--machine-id", default=socket.gethostname(), help="machine id on every record")
    p.add_argument("--smoke", action="store_true", help="A100 and C10, seed 0, a 300 s cap")
    p.add_argument("--resume", action="store_true", help="skip the fits already in --out")
    p.add_argument(
        "--rescore", action="store_true", help="fit nothing: rescore the stored marks in --out"
    )
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)
    unknown = sorted(set(args.cells) - set(BY_NAME)) + sorted(set(args.arms) - set(ARMS))
    if unknown:
        p.error(f"unknown cells or arms {unknown}; cells {list(BY_NAME)}, arms {list(ARMS)}")

    out = Path(args.out)
    if args.rescore:
        records = rescore(out)
        _log(f"rescored    : {len(records)} records -> {out / RECORDS}")
        _summarize(records)
        return 0
    if (out / RECORDS).exists() and not args.resume:
        raise SystemExit(f"{out} already holds a run; pass --resume to continue it")
    out.mkdir(parents=True, exist_ok=True)

    from andrey_bench import provenance

    reference = REFERENCE in args.arms or CEILING in args.arms
    python = gnies_gnies.interpreter() if reference else ""
    versions = {"andrey": andrey_gnies.VERSION, **(gnies_gnies.versions(python) if python else {})}
    versions.setdefault("gnies", "")
    meta = {
        "driver": Path(__file__).name,
        "sampler": "andrey.data, scale='raw', pooled standardization",
        "cells": args.cells,
        "seeds": args.seeds,
        "arms": args.arms,
        "cap_wall_s": args.cap_wall_s,
        "cap_ceiling_s": args.cap_ceiling_s,
        "member_cap": MEMBER_CAP,
        "versions": versions,
        "machine_id": args.machine_id,
    }
    meta_path = out / "run_meta.json"
    if args.resume and meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise SystemExit(f"--resume: {meta_path} differs from these flags; pass the first run's")
    meta_path.write_text(json.dumps(meta, indent=2))
    if not (out / "provenance.json").exists():
        extra = {"gnies": str(Path(python).parents[1])} if python else None
        stamp = datetime.now(timezone.utc).isoformat()
        provenance.capture(out, timestamp=stamp, extra_venvs=extra)
    _log(f"=== GnIES, Andrey and gnies -> {out} ===")
    for name, value in meta.items():
        _log(f"{name:14s}: {value}")

    caps = {arm: args.cap_wall_s for arm in ARMS} | {CEILING: args.cap_ceiling_s}
    done = {(r["cell"], r["seed"], r["arm"]) for r in _read(out)}
    _warm_up()
    with (out / RECORDS).open("a") as sink:
        for name in args.cells:
            for seed in args.seeds:
                todo = [a for a in args.arms if (name, seed, a) not in done]
                if not todo:
                    continue
                data = draw(BY_NAME[name], seed)
                truth = Truth(data)
                for arm in todo:
                    row = record(arm, data, truth, run_arm(arm, data, truth, caps), meta)
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
                _log(f"[{name} seed {seed}] {len(todo)} arms")
    _summarize(_read(out))
    return 0


def _summarize(records: list[dict]) -> None:
    """Print, per cell and arm, finished fits, median time, mean SHD, mean target error."""
    if not records:
        return
    _log(
        f"\n{'cell':8s} {'arm':26s} {'ok/all':>7s} {'wall_s':>8s} {'shd':>6s} {'t_err':>6s} "
        f"{'gap':>8s}"
    )
    for cell in dict.fromkeys(r["cell"] for r in records):
        for arm in ARMS:
            mine = [r for r in records if r["cell"] == cell and r["arm"] == arm]
            fit = [r for r in mine if r["status"] == "ok"]
            if not mine:
                continue

            def mean(key: str, rows: list[dict] = fit) -> float:
                values = [r[key] for r in rows if r.get(key) is not None]
                return statistics.fmean(values) if values else float("nan")

            wall = statistics.median(r["wall_s"] for r in fit) if fit else float("nan")
            _log(
                f"{cell:8s} {arm:26s} {len(fit):3d}/{len(mine):<3d} {wall:8.3g} "
                f"{mean('shd'):6.2f} {mean('target_error'):6.2f} {mean('gap_common'):8.3g}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
