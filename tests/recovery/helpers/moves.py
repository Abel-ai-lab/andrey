"""Where a score search stops: no single move of its own improves the score.

Hill climbing stops at a DAG from which no legal edge addition, removal, or reversal lowers the
score, and tabu search returns the best DAG it visited, which no legal move improves either. Both
report the DAG's CPDAG, so :func:`class_gain` checks every DAG of that class and returns the
smallest best gain: some DAG of the class, the one the search stopped at, gains nothing.
:func:`boss_gain` and :func:`grasp_gain` rerun the single start of :func:`andrey.boss` and
:func:`andrey.grasp` and return the best gain of one more move of their own: moving one variable to
another place in the order, or one tuck.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from andrey.core import ARROW, TAIL, GraphStructure
from andrey.core.score import LocalScore
from andrey.core.seeding import start_rng
from andrey.metrics import to_cpdag
from andrey.search import boss as boss_search
from andrey.search import grasp as grasp_search
from andrey.search._parallel import _KEY_DP


def grid(n_samples: int) -> float:
    """One step of the grid hill climbing and tabu search round ``delta / n_samples`` to, in score
    units: a smaller gain is rounding."""
    return n_samples * 10.0**-_KEY_DP


def _dag(adj: np.ndarray) -> GraphStructure:
    marks = np.zeros(adj.shape, dtype=np.int8)
    marks[adj] = TAIL
    marks[adj.T] = ARROW
    return GraphStructure.from_numpy(marks, kind="dag")


def _acyclic(adj: np.ndarray) -> bool:
    left = adj.copy()
    alive = np.ones(len(adj), dtype=bool)
    while alive.any():
        sources = alive & ~left[alive].any(axis=0)
        if not sources.any():
            return False
        alive &= ~sources
        left[sources] = False
    return True


def class_dags(cpdag: GraphStructure, forbidden: np.ndarray | None = None) -> list[np.ndarray]:
    """Every DAG of ``cpdag``'s class with no edge ``u -> v`` that ``forbidden[u, v]`` bars, as a
    boolean adjacency matrix (``adj[u, v]``: ``u -> v``)."""
    marks = cpdag.to_numpy()
    directed = (marks == TAIL) & (marks.T == ARROW)
    undirected = list(zip(*np.nonzero(np.triu((marks == TAIL) & (marks.T == TAIL)))))
    members = []
    for flips in itertools.product((False, True), repeat=len(undirected)):
        adj = directed.copy()
        for (u, v), flip in zip(undirected, flips, strict=True):
            adj[(v, u) if flip else (u, v)] = True
        if forbidden is not None and (adj & forbidden).any():
            continue
        if _acyclic(adj) and np.array_equal(to_cpdag(_dag(adj)).to_numpy(), marks):
            members.append(adj)
    return members


def _reaches(adj: np.ndarray, a: int, b: int) -> bool:
    seen, todo = {a}, [a]
    while todo:
        for c in np.flatnonzero(adj[todo.pop()]).tolist():
            if c == b:
                return True
            if c not in seen:
                seen.add(c)
                todo.append(c)
    return False


def move_gain(
    adj: np.ndarray,
    score: LocalScore,
    allowed: np.ndarray | None = None,
    forbidden: np.ndarray | None = None,
) -> float:
    """The most one legal edge addition, removal, or reversal lowers the total score of ``adj``.

    An addition or a reversal keeps the graph acyclic, joins only a pair ``allowed`` holds, when it
    is given, and adds no edge ``u -> v`` that ``forbidden[u, v]`` bars. Lower scores are better,
    so a positive gain is an improving move.
    """
    d = len(adj)
    parents = [np.flatnonzero(adj[:, v]).tolist() for v in range(d)]
    local = [score.score(v, parents[v]) for v in range(d)]
    gain = -math.inf

    def without(v: int, u: int) -> list[int]:
        return [p for p in parents[v] if p != u]

    for u, v in itertools.permutations(range(d), 2):
        barred = forbidden is not None and forbidden[v, u]
        if adj[u, v]:
            gain = max(gain, local[v] - score.score(v, without(v, u)))
            adj[u, v] = False
            if not barred and not _reaches(adj, u, v):  # the reverse ``v -> u`` closes no cycle
                change = local[v] - score.score(v, without(v, u))
                change += local[u] - score.score(u, sorted([*parents[u], v]))
                gain = max(gain, change)
            adj[u, v] = True
        elif (
            not adj[v, u]
            and (allowed is None or allowed[u, v])
            and not (forbidden is not None and forbidden[u, v])
            and not _reaches(adj, v, u)
        ):
            gain = max(gain, local[v] - score.score(v, sorted([*parents[v], u])))
    return gain


def class_gain(
    cpdag: GraphStructure,
    score: LocalScore,
    allowed: np.ndarray | None = None,
    forbidden: np.ndarray | None = None,
) -> float:
    """The smallest :func:`move_gain` over the DAGs of ``cpdag``'s class that keep ``forbidden``."""
    return min(move_gain(adj, score, allowed, forbidden) for adj in class_dags(cpdag, forbidden))


def _cpdag(parents: dict[int, list[int]], ranks: np.ndarray) -> np.ndarray:
    """The CPDAG of the search's parent sets, over the data's columns, as the searches return it."""
    adj = np.zeros((len(ranks),) * 2, dtype=bool)
    for child, pa in parents.items():
        adj[pa, child] = True
    return to_cpdag(_dag(adj[np.ix_(ranks, ranks)])).to_numpy()


def boss_gain(data: np.ndarray, score_func: str, seed: int = 0) -> tuple[float, np.ndarray]:
    """Rerun the single start of ``andrey.boss(data, score_func=score_func, seed=seed)``; return the
    most that moving one variable to another place in its end order lowers the order's score, and
    the CPDAG it returns."""
    data = np.asarray(data, dtype=np.float64)
    score, ranks, trees, _ = boss_search._search_state(
        data, score_func, 1.0, 1.0, 1.0, None, None, None
    )
    order = list(range(data.shape[1]))
    parents, total = boss_search._search_from(order, start_rng(seed, 0), trees, score, None)

    def order_score(candidate: list[int]) -> float:
        prefix: set[int] = set()
        total = 0.0
        for v in candidate:
            pa: list[int] = []
            trees[v].trace(prefix, pa)
            total += score.score(v, pa)
            prefix.add(v)
        return total

    gain = -math.inf
    for v in order:
        rest = [u for u in order if u != v]
        for place in range(len(order)):
            gain = max(gain, total - order_score([*rest[:place], v, *rest[place:]]))
    return gain, _cpdag(parents, ranks)


def grasp_gain(
    data: np.ndarray, score_func: str, seed: int = 0, depth: int = 3
) -> tuple[float, np.ndarray]:
    """Rerun the single start of ``andrey.grasp(data, score_func=score_func, seed=seed)``; return
    the most one tuck of an edge of its end DAG raises the order's score, which GRaSP raises, and
    the CPDAG it returns."""
    data = np.asarray(data, dtype=np.float64)
    _, ranks, trees, _ = grasp_search._search_state(
        data, score_func, 1.0, 1.0, 1.0, None, None, None
    )
    order = grasp_search._search_from(data.shape[1], start_rng(seed, 0), trees, depth, None)
    gain = -math.inf
    for i in range(order.len()):
        y = order.get(i)
        for x in list(order.get_parents(y)):
            j = order.index(x)
            saved = grasp_search._snapshot(order, j, i)
            grasp_search._tuck(i, j, order)
            gain = max(gain, grasp_search._update(i, j, order, trees)[1])
            grasp_search._restore(order, j, saved)
    parents = {y: list(order.get_parents(y)) for y in order.order}
    return gain, _cpdag(parents, ranks)
