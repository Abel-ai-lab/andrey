"""Markov blankets and parents-and-children sets, the engines behind :func:`andrey.iamb`,
:func:`andrey.inter_iamb`, and :func:`andrey.mmpc`.

:func:`markov_blanket` runs IAMB or Inter-IAMB for one target and :func:`parents_children` the
forward and backward phases of MMPC. :func:`blanket_graph` and :func:`mmpc_skeleton` run them for
every variable and keep a pair when each lies in the other's set (the AND rule), which gives the
moral graph and the skeleton that :func:`andrey.calm`, :func:`andrey.mmhc`, and
:func:`andrey.rsmax2` search within. Each takes any :class:`~andrey.core.ci.CITest`.

Candidates are compared by the strength of their dependence on the target: the absolute partial
correlation for a test that reports one (``statistic_pvalue``), else the p-value. Strengths that
agree to ``_KEY_DP`` decimals go to the candidate with the lower rank, from
:func:`~andrey.search.hc.tie_ranks`, so the sets do not depend on the column order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations

import numpy as np

from andrey.constraint._skeleton import scan_conditioning_sets
from andrey.core.ci import CITest
from andrey.search._parallel import _KEY_DP

# ``(p-value, key)`` of ``x`` against the target given a conditioning set; a lower key is a
# stronger dependence.
_Dependence = Callable[[int, int, Sequence[int]], tuple[float, float]]


def _dependence(test: CITest) -> _Dependence:
    """The p-value and strength key of a query, read from ``statistic_pvalue`` where it exists."""
    statistic_pvalue = getattr(test, "statistic_pvalue", None)
    if statistic_pvalue is not None:

        def by_statistic(target: int, x: int, cond: Sequence[int]) -> tuple[float, float]:
            r, p = statistic_pvalue(target, x, cond)
            return p, -round(abs(r), _KEY_DP)

        return by_statistic

    def by_pvalue(target: int, x: int, cond: Sequence[int]) -> tuple[float, float]:
        p = test(target, x, cond)
        return p, round(p, _KEY_DP)

    return by_pvalue


def markov_blanket(
    test: CITest,
    target: int,
    d: int,
    alpha: float,
    ranks: Sequence[int],
    *,
    interleaved: bool = False,
) -> list[int]:
    """Estimate ``target``'s Markov blanket with IAMB, or with Inter-IAMB when ``interleaved``.

    Grow: add the candidate most dependent on ``target`` given the current blanket, among those
    whose p-value is at or below ``alpha``, until none is. Shrink: visit the members in the order
    they joined and drop each one the test finds independent of ``target`` (a p-value above
    ``alpha``) given the other current members. IAMB grows to the end and then shrinks once;
    Inter-IAMB shrinks after every addition and stops when no candidate qualifies or the blanket
    repeats one it held after an earlier shrink, so it always ends. Returns the members in the order
    they joined.
    """
    dependence = _dependence(test)
    blanket: list[int] = []
    seen: set[frozenset[int]] = set()
    while True:
        best_key, best_x = None, None
        for x in range(d):
            if x == target or x in blanket:
                continue
            p, strength = dependence(target, x, blanket)
            key = (strength, ranks[x])
            if p <= alpha and (best_key is None or key < best_key):
                best_key, best_x = key, x
        if best_x is None:
            break
        blanket.append(best_x)
        if interleaved:
            _shrink(test, target, alpha, blanket)
            state = frozenset(blanket)
            if state in seen:
                break
            seen.add(state)
    if not interleaved:
        _shrink(test, target, alpha, blanket)
    return blanket


def _shrink(test: CITest, target: int, alpha: float, blanket: list[int]) -> None:
    """Drop, in place and in joining order, each member independent of ``target`` given the rest."""
    for x in list(blanket):
        rest = [z for z in blanket if z != x]
        if test(target, x, rest) > alpha:
            blanket.remove(x)


def blanket_graph(
    test: CITest, d: int, alpha: float, ranks: Sequence[int], *, interleaved: bool = False
) -> np.ndarray:
    """The moral graph: ``[i, j]`` when each of ``i`` and ``j`` is in the other's Markov blanket.

    Runs :func:`markov_blanket` for every variable; the result is a symmetric boolean ``(d, d)``
    matrix with a ``False`` diagonal.
    """
    raw = np.zeros((d, d), dtype=bool)
    for i in range(d):
        for j in markov_blanket(test, i, d, alpha, ranks, interleaved=interleaved):
            raw[i, j] = True
    return raw & raw.T


def _weakest(
    test: CITest, target: int, xs: list[int], conds: list[tuple[int, ...]]
) -> list[tuple[float, float]]:
    """Each candidate's weakest association with ``target`` over the conditioning sets ``conds``.

    Returns ``(largest p-value, smallest strength)`` per entry of ``xs``. The strength is the
    absolute partial correlation for a test with ``statistic_pvalues``, evaluated for every query
    of one conditioning-set size in one call, else the negated p-value from ``batched_call`` or the
    scalar test.
    """
    batch = getattr(test, "statistic_pvalues", None)
    if batch is None:
        batched = getattr(test, "batched_call", None)
        out = []
        for x in xs:
            ps = (
                batched(target, x, conds)
                if batched is not None
                else [test(target, x, c) for c in conds]
            )
            top = float(max(ps))
            out.append((top, -top))
        return out
    pmax = np.full(len(xs), -np.inf)
    rmin = np.full(len(xs), np.inf)
    by_size: dict[int, list[tuple[int, ...]]] = {}
    for cond in conds:
        by_size.setdefault(len(cond), []).append(cond)
    for size, group in by_size.items():
        cols = np.array(group, dtype=np.intp).reshape(len(group), size)
        x = np.repeat(np.asarray(xs, dtype=np.intp), len(group))
        r, p = batch(np.full(x.size, target, dtype=np.intp), x, np.tile(cols, (len(xs), 1)))
        pmax = np.maximum(pmax, np.asarray(p).reshape(len(xs), len(group)).max(axis=1))
        rmin = np.minimum(rmin, np.abs(np.asarray(r)).reshape(len(xs), len(group)).min(axis=1))
    return list(zip(pmax.tolist(), rmin.tolist(), strict=True))


def parents_children(
    test: CITest, target: int, d: int, alpha: float, ranks: Sequence[int]
) -> list[int]:
    """Estimate ``target``'s parents and children with the forward and backward phases of MMPC.

    A candidate's association with ``target`` is its weakest over the conditioning sets drawn from
    the current set (:func:`_weakest`): the largest p-value among them, then, where two candidates'
    largest p-values agree to ``_KEY_DP`` decimals, the smallest strength, then the rank. Forward:
    add the candidate whose weakest association is the strongest, until every candidate has a
    conditioning set with a p-value above ``alpha``; such a candidate is dropped for good, since
    adding members only weakens an association. Each addition tests every remaining candidate
    against the conditioning sets that hold the new member. Backward: visit the members in the
    order they joined and drop each one some subset of the other current members separates; the
    subsets of the members that joined before it were tested in the forward phase. Returns the
    members in the order they joined.

    This is MMPC without its symmetry check: :func:`mmpc_skeleton` keeps ``x`` in ``target``'s set
    only when ``target`` is in ``x``'s.
    """
    others = [x for x in range(d) if x != target]
    weakest = dict(zip(others, _weakest(test, target, others, [()]), strict=True))
    members: list[int] = []
    while True:
        open_ = {x: w for x, w in weakest.items() if w[0] <= alpha}
        if not open_:
            break
        best = min(
            open_,
            key=lambda x: (round(open_[x][0], _KEY_DP), -round(open_[x][1], _KEY_DP), ranks[x]),
        )
        del open_[best]
        conds = [
            tuple(sorted((*rest, best)))
            for size in range(len(members) + 1)
            for rest in combinations(members, size)
        ]
        xs = list(open_)
        for x, (p, strength) in zip(xs, _weakest(test, target, xs, conds), strict=True):
            open_[x] = (max(open_[x][0], p), min(open_[x][1], strength))
        members.append(best)
        weakest = open_
    for x in list(members):
        before = set(members[: members.index(x)])
        rest = sorted(z for z in members if z != x)
        for size in range(1, len(rest) + 1):
            if scan_conditioning_sets(test, target, x, rest, size, alpha, set(), before):
                members.remove(x)
                break
    return members


def mmpc_skeleton(test: CITest, d: int, alpha: float, ranks: Sequence[int]) -> np.ndarray:
    """MMPC's skeleton: ``[i, j]`` when each of ``i`` and ``j`` is in the other's parents and
    children (:func:`parents_children`), a symmetric boolean ``(d, d)`` matrix."""
    raw = np.zeros((d, d), dtype=bool)
    for i in range(d):
        raw[i, parents_children(test, i, d, alpha, ranks)] = True
    return raw & raw.T
