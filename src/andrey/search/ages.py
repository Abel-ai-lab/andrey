"""Aggregative greedy equivalence search, the engine behind :func:`andrey.ages`.

GES with the linear-Gaussian BIC charges every edge one penalty, so the best Insert or Delete on a
graph is the same under every penalty, and only the point where a phase stops moves with it. The
CPDAGs GES returns over all penalties at or above a minimum, its solution path, therefore come
from one forward phase at the minimum penalty and one backward phase per distinct forward result
(:func:`solution_path`), scanned and applied by the GES engine (:mod:`andrey.search._parallel_ges`).
:func:`aggregate` overlays the orientations of the path's CPDAGs on the CPDAG of the minimum
penalty. The algorithm, its settings, and its reference are described on :func:`andrey.ages`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np

from andrey.core import GraphStructure, backend
from andrey.core.orient import meek, pdag2dag, to_structure
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore
from andrey.core.structure import ARROW, NULL, TAIL
from andrey.core.ties import EPS_IMPROVE, tie_mask, tie_ranks
from andrey.search import operators as ops
from andrey.search._parallel_ges import _ForwardScan, backward_chunk_best, neighborhood_state
from andrey.search.ges import recompute_objective


def _forward(
    score: BICScore, n: int, ranks: list[int]
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """The forward phase of GES under ``score`` from the empty graph, move by move.

    Returns the CPDAG after 0, 1, ... Inserts, the total score of each, and each Insert's limit:
    the ``penalty_discount`` below which the Insert lowers the score. No gate limits a node's
    directed parents, unlike :func:`andrey.search.ges.ges` without ``max_parents``.
    """
    scan = _ForwardScan(range(n), DeltaBICScore(score), n, None, ranks)
    graphs = [np.zeros((n, n), dtype=np.int8)]
    totals = [recompute_objective(graphs[0], score, n)]
    limits: list[float] = []
    while (best := scan.best(graphs[-1])) is not None:
        _, x, y, T = best
        graphs.append(ops.insert_cpdag(graphs[-1].copy(), x, y, T))
        totals.append(recompute_objective(graphs[-1], score, n))
        limits.append(score.penalty_discount + (totals[-2] - totals[-1]) / score.log_n)
    return graphs, totals, limits


def _backward(
    adj: np.ndarray, score: BICScore, n: int, ranks: list[int]
) -> Iterator[tuple[np.ndarray, float]]:
    """The backward phase of GES under ``score`` from the CPDAG ``adj``, move by move.

    Yields the CPDAG after each Delete and the Delete's limit: the ``penalty_discount`` above which
    the Delete lowers the score.
    """
    engine = DeltaBICScore(score)
    kept: dict[tuple[int, tuple[int, ...], int], float] = {}
    total = recompute_objective(adj, score, n)
    while True:
        state = neighborhood_state(adj)
        best = backward_chunk_best(range(n), adj, engine, n, state, ranks, kept)
        if best is None:
            return
        _, x, y, H = best
        adj = ops.delete_cpdag(adj.copy(), x, y, H)
        after = recompute_objective(adj, score, n)
        yield adj, score.penalty_discount + (after - total) / score.log_n
        total = after


def solution_path(
    data: np.ndarray, *, penalty_discount: float = 1.0
) -> list[tuple[float, np.ndarray]]:
    """The CPDAGs GES returns under every penalty from ``penalty_discount`` up, by rising penalty.

    Each entry is ``(penalty, cpdag)``: the CPDAG of the forward then backward phase under the
    standard linear-Gaussian BIC (``ebic_gamma=0.0``) with that ``penalty_discount`` and no gate
    on a node's directed parents, the lowest one at or above the minimum that gives the CPDAG.
    The first entry is the CPDAG of the minimum
    penalty; a CPDAG equal to the entry before it is left out.

    The forward phase runs once, at the minimum penalty: under a higher penalty it makes the same
    Inserts and stops at the first whose limit the penalty reaches, so the forward results are the
    graphs before an Insert whose limit is below every earlier one, and the last graph. From each,
    the backward phase runs at the lowest penalty that stops the forward phase there, then at the
    highest: a graph after a Delete is a result when the next Delete's limit is above every
    earlier one. Two limits within :data:`~andrey.core.ties.EPS_IMPROVE` score units are equal.
    """
    X = np.asarray(data, dtype=np.float64)
    n = X.shape[1]
    base = BICScore(X, penalty_discount=penalty_discount)
    ranks = tie_ranks(X, tie_mask("bic", None, n)).tolist()
    tol = EPS_IMPROVE / base.log_n

    def at(penalty: float) -> BICScore:
        return BICScore.from_cov(base.cov, base.n, penalty_discount=penalty)

    path: list[tuple[float, np.ndarray]] = []

    def add(penalty: float, cpdag: np.ndarray) -> None:
        if not path or not np.array_equal(path[-1][1], cpdag):
            path.append((penalty, cpdag))

    with backend.one_blas_thread():
        graphs, _, limits = _forward(base, n, ranks)
        for k in range(len(limits), -1, -1):
            low = limits[k] if k < len(limits) else base.penalty_discount
            high = min(limits[:k], default=math.inf)
            if not low < high - tol:
                continue  # no penalty stops the forward phase after k Inserts
            cpdag = graphs[k]
            for cpdag, _ in _backward(cpdag, at(low), n, ranks):  # noqa: B007
                pass
            add(low, cpdag)
            if math.isinf(high):
                continue  # the empty graph
            deletes = list(_backward(cpdag, at(high), n, ranks))
            top = low
            for j, (after, limit) in enumerate(deletes):
                top = max(top, limit)
                if j + 1 == len(deletes) or deletes[j + 1][1] > top + tol:
                    add(top, after)
    return path


def aggregate(cpdags: Sequence[np.ndarray]) -> tuple[np.ndarray, list[bool]]:
    """Overlay the orientations of ``cpdags`` on the first and close the result under Meek's rules.

    ``cpdags`` are a solution path, by rising penalty. The graph starts as the first CPDAG. Each
    later CPDAG whose skeleton lies within the first's directs, as it does, every edge it directs
    that the graph leaves undirected, when the graph so directed has a consistent DAG extension
    (:func:`~andrey.core.orient.pdag2dag`); otherwise none of its orientations is taken. Meek's
    rules R1 to R4 then direct what follows. Returns the aggregated graph and, per CPDAG, whether
    its orientations were taken.
    """
    graph = np.array(cpdags[0], dtype=np.int8)
    skeleton = graph != NULL
    taken = [True]
    for cpdag in cpdags[1:]:
        fits = not ((cpdag != NULL) & ~skeleton).any()
        if fits:
            directed = (cpdag == TAIL) & (cpdag.T == ARROW)  # [i, j]: i -> j
            new = directed & (graph == TAIL) & (graph.T == TAIL)
            merged = graph.copy()
            merged.T[new] = ARROW
            try:
                pdag2dag(merged)
            except ValueError:
                fits = False
            else:
                graph = merged
        taken.append(fits)
    return meek(graph, rule4=True), taken


def ages(
    data: np.ndarray, *, penalty_discount: float = 1.0
) -> tuple[GraphStructure, list[float], list[bool]]:
    """Aggregate the GES solution path of ``data`` from ``penalty_discount`` up.

    Returns the aggregated graph (``kind="cpdag"``, a maximally oriented partially directed graph
    with the skeleton of the CPDAG of the minimum penalty), the penalties of the path's CPDAGs
    (:func:`solution_path`), and whether each CPDAG's orientations were taken (:func:`aggregate`).
    """
    path = solution_path(data, penalty_discount=penalty_discount)
    graph, taken = aggregate([cpdag for _, cpdag in path])
    return to_structure(graph, kind="cpdag"), [penalty for penalty, _ in path], taken
