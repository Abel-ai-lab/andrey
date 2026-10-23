"""XGES with every operator rescanned and the CPDAG re-formed globally after each move.

The reference the queue-based search in ``andrey.search.xges`` must agree with: it rescans every
Delete, Reverse, and Insert from :mod:`andrey.search.operators` after each move and rebuilds the
whole CPDAG, where the search updates its queues and completes the CPDAG locally.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable

import numpy as np

from andrey.core import NULL, GraphStructure, backend
from andrey.core.knowledge import Constraints, admits, mpdag
from andrey.core.orient import to_structure
from andrey.core.score import BICScore
from andrey.search import operators as ops
from andrey.search._parallel import _EPS_IMPROVE, _KEY_DP
from andrey.search.ges import _reachable, _recompute_objective, _reform_cpdag, required_dag
from andrey.search.hc import tie_ranks
from andrey.search.xges import (
    _ACCEPT_PER_SAMPLE,
    _extension_scores,
    _local,
    _pinned,
)


def _move_key(
    delta: float, n: int, x: int, y: int, subset: Collection[int], ranks: list[int]
) -> tuple:
    """The order of a move of edge ``x``-``y`` with ``subset`` (``T`` or ``C``); lower goes first.

    ``(round(delta / n, _KEY_DP), ranks[x], ranks[y], sorted ranks of subset)`` for ``n``
    samples and ``ranks`` from :func:`~andrey.search.hc.tie_ranks`: moves whose score changes per
    sample agree to :data:`~andrey.search._parallel._KEY_DP` decimals are ordered by ranks that
    follow the columns.
    """
    ordered = tuple(sorted([ranks[v] for v in subset])) if subset else ()
    return (round(delta / n, _KEY_DP), ranks[x], ranks[y], ordered)


def _sets(adj: np.ndarray, v: int) -> tuple[set[int], set[int], set[int]]:
    """Parents, undirected neighbors, and adjacent nodes of ``v`` in a mark matrix."""
    return (
        set(ops.parents(adj, v).tolist()),
        set(ops.undirected_neighbors(adj, v).tolist()),
        set(ops.adjacent(adj, v).tolist()),
    )


def _all_deletes(
    adj: np.ndarray, score: BICScore, constraints: Constraints | None = None
) -> list[tuple]:
    """Every Delete of a CPDAG as ``(delta, x, y, C, E)``, ``C`` the neighbors kept undirected.

    With ``constraints``, no required pair is deleted and ``C`` keeps every :func:`_pinned` node.
    """
    out = []
    for y in range(adj.shape[0]):
        pa_y, ne_y, _ = _sets(adj, y)
        for x in sorted(pa_y | ne_y):
            h0 = sorted(ne_y & set(ops.adjacent(adj, x).tolist()))
            candidates = ops.clique_residuals(adj, h0)
            if constraints is not None:
                if constraints.required[x, y] or constraints.required[y, x]:
                    continue
                pinned = _pinned(h0, x, y, constraints)
                candidates = ((i, H) for i, H in candidates if pinned.isdisjoint(H))
            for _, H in candidates:
                C = frozenset(h0) - set(H)
                E = frozenset(C | pa_y | {x})
                delta = _local(score, y, E - {x}) - _local(score, y, E)
                out.append((delta, x, y, C, E))
    return out


def _free(ne_y: set[int], y: int, constraints: Constraints | None) -> set[int]:
    """The undirected neighbors of ``y`` an Insert into ``y`` may orient into it."""
    if constraints is None:
        return ne_y
    return {t for t in ne_y if not constraints.forbidden[t, y]}


def _all_inserts(
    adj: np.ndarray,
    score: BICScore,
    forbidden: set[tuple[int, int]],
    reachable: Callable,
    constraints: Constraints | None = None,
) -> list[tuple]:
    """Every valid Insert ``x -> y`` of a CPDAG as ``(delta, x, y, T)``."""
    out = []
    d = adj.shape[0]
    for y in range(d):
        pa_y, ne_y, ad_y = _sets(adj, y)
        free = _free(ne_y, y, constraints)
        for x in range(d):
            if x == y or x in ad_y or (x, y) in forbidden:
                continue
            if constraints is not None and constraints.forbidden[x, y]:
                continue
            ad_x = set(ops.adjacent(adj, x).tolist())
            na = ne_y & ad_x
            for _, T in ops.clique_extensions(adj, na, sorted(free - ad_x)):
                if reachable(y, x, na | set(T)):
                    continue
                E = pa_y | na | set(T)
                out.append((_local(score, y, E | {x}) - _local(score, y, E), x, y, frozenset(T)))
    return out


def _all_reverses(
    adj: np.ndarray,
    score: BICScore,
    forbidden: set[tuple[int, int]],
    constraints: Constraints | None = None,
) -> list[tuple]:
    """Every valid Reverse of a ``y -> x`` in a CPDAG as ``(delta, x, y, T)``."""
    out = []
    for x in range(adj.shape[0]):
        pa_x, ne_x, ad_x = _sets(adj, x)
        for y in sorted(pa_x):
            if (x, y) in forbidden:
                continue
            if constraints is not None and constraints.forbidden[x, y]:
                continue
            cut = adj.copy()
            cut[x, y] = cut[y, x] = NULL
            succ = ops.tail_successor_lists(cut)
            pa_y, ne_y, _ = _sets(adj, y)
            na = ne_y & ad_x
            drop = _local(score, x, pa_x - {y}) - _local(score, x, pa_x)
            for _, T in ops.clique_extensions(adj, na, sorted(_free(ne_y, y, constraints) - ad_x)):
                if not ops.blocks_semi_directed_paths(succ, y, x, na | set(T) | ne_x):
                    continue
                E = pa_y | na | set(T)
                delta = _local(score, y, E | {x}) - _local(score, y, E) + drop
                out.append((delta, x, y, frozenset(T)))
    return out


def _ordered(candidates: Iterable[tuple], n: int, ranks: list[int]) -> list[tuple]:
    """``(delta, x, y, subset, ...)`` candidates by :func:`_move_key`, for ``n`` samples."""
    return sorted(candidates, key=lambda c: _move_key(c[0], n, c[1], c[2], c[3], ranks))


def _improving(candidates: list[tuple], n: int, ranks: list[int]) -> list[tuple]:
    """The improving candidates in :func:`_ordered` order."""
    return _ordered((c for c in candidates if c[0] < -_EPS_IMPROVE), n, ranks)


def _delete_marks(adj: np.ndarray, x: int, y: int, C: frozenset[int]) -> np.ndarray:
    """Apply Delete ``x - y`` keeping ``C`` to a mark matrix and re-form the CPDAG."""
    adj = adj.copy()
    _, ne_y, _ = _sets(adj, y)
    H = (ne_y & set(ops.adjacent(adj, x).tolist())) - C
    ops.apply_delete(adj, x, y, H)
    return _reform_cpdag(adj)


def _insert_marks(adj: np.ndarray, x: int, y: int, T: frozenset[int]) -> np.ndarray:
    """Apply Insert ``x -> y`` with ``T`` to a copy of a mark matrix and re-form the CPDAG."""
    adj = adj.copy()
    ops.apply_insert(adj, x, y, T)
    return _reform_cpdag(adj)


def _rescan_moves(
    adj: np.ndarray,
    score: BICScore,
    forbidden: set[tuple[int, int]],
    ranks: list[int],
    constraints: Constraints | None = None,
) -> Iterable[np.ndarray]:
    """The graphs reached by improving base-search moves on ``adj``, in search order."""
    for _, x, y, C, _E in _improving(_all_deletes(adj, score, constraints), score.n, ranks):
        yield _delete_marks(adj, x, y, C)
    reverses = _all_reverses(adj, score, forbidden, constraints)
    for _, x, y, T in _improving(reverses, score.n, ranks):
        cut = adj.copy()
        cut[x, y] = cut[y, x] = NULL
        yield _insert_marks(cut, x, y, T)
    inserts = _all_inserts(adj, score, forbidden, _reachable(adj), constraints)
    for _, x, y, T in _improving(inserts, score.n, ranks):
        yield _insert_marks(adj, x, y, T)


def _rescan_base_search(
    adj: np.ndarray,
    score: BICScore,
    forbidden: set[tuple[int, int]],
    ranks: list[int],
    constraints: Constraints | None = None,
) -> np.ndarray:
    """The base search with every operator rescanned after each move, skipping visited graphs and,
    with ``constraints``, graphs whose class holds no DAG that satisfies them."""
    seen = {adj.tobytes()}
    while True:
        for moved in _rescan_moves(adj, score, forbidden, ranks, constraints):
            if constraints is not None and not admits(moved, constraints):
                continue
            if moved.tobytes() not in seen:
                seen.add(moved.tobytes())
                adj = moved
                break
        else:
            return adj


def _marks_total(adj: np.ndarray, score: BICScore) -> float:
    """The total of :func:`_extension_scores` for a CPDAG given as a mark matrix."""
    d = adj.shape[0]
    sets = [_sets(adj, v) for v in range(d)]
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = _extension_scores([s[0] for s in sets], [s[1] for s in sets], range(d), score)
    return sum(scores[v] for v in range(d))


def _rescan_search(
    score: BICScore,
    d: int,
    extended: bool,
    ranks: list[int],
    constraints: Constraints | None = None,
) -> np.ndarray:
    """XGES with every operator rescanned and the CPDAG re-formed globally after each move."""
    start = np.zeros((d, d), dtype=np.int8)
    if constraints is not None and constraints.required.any():
        start = _reform_cpdag(required_dag(constraints))
    adj = _rescan_base_search(start, score, set(), ranks, constraints)
    if not extended:
        return adj
    tolerance = _ACCEPT_PER_SAMPLE * score.n
    total = _marks_total(adj, score)
    forbidden: set[tuple[int, int]] = set()
    pending: list[tuple] = []
    fresh = False
    while pending or not fresh:
        if not pending:
            pending = _ordered(_all_deletes(adj, score, constraints), score.n, ranks)
            if not pending:
                break
            fresh = True
        _, x, y, C, E = pending.pop(0)
        pa_y, ne_y, _ = _sets(adj, y)
        ad_x = set(ops.adjacent(adj, x).tolist())
        valid = (x in pa_y or x in ne_y) and C <= ne_y & ad_x and E == C | pa_y | {x}
        if constraints is not None:
            valid = valid and _pinned(ne_y & ad_x, x, y, constraints) <= C
        if not (valid and ops.is_clique(adj, C)):
            continue
        deleted = _delete_marks(adj, x, y, C)
        if constraints is not None and not admits(deleted, constraints):
            continue
        trial_forbidden = forbidden | {(x, y)}
        trial = _rescan_base_search(deleted, score, trial_forbidden, ranks, constraints)
        trial_total = _marks_total(trial, score)
        if trial_total < total - tolerance:
            adj, total, forbidden = trial, trial_total, trial_forbidden
            fresh = False
    return adj


def xges_rescan(
    data: np.ndarray,
    *,
    lambda_value: float = 1.0,
    extended_search: bool = True,
    constraints: Constraints | None = None,
) -> tuple[GraphStructure, float]:
    """:func:`andrey.search.xges.xges` with the rescanning search: the CPDAG and its total score."""
    X = np.asarray(data, dtype=np.float64)
    d = X.shape[1]
    score = BICScore(X, lambda_value=lambda_value)
    ranks = tie_ranks(X).tolist()
    with backend.one_blas_thread():
        adj = _rescan_search(score, d, extended_search, ranks, constraints)
        total = _recompute_objective(adj, score, d)
    if constraints is not None:
        adj = mpdag(adj, constraints)
    return to_structure(adj, kind="cpdag"), total
