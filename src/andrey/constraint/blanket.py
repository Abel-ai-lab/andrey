"""Markov blankets and parents-and-children sets, the engines behind :func:`andrey.iamb`,
:func:`andrey.inter_iamb`, and :func:`andrey.mmpc`.

:func:`markov_blanket` runs IAMB or Inter-IAMB for one target and :func:`parents_children` the
forward and backward phases of MMPC. :func:`blanket_graph` and :func:`mmpc_skeleton` run them for
every variable and keep a pair when each lies in the other's set (the AND rule), which gives the
moral graph and the skeleton that :func:`andrey.calm`, :func:`andrey.mmhc`, and
:func:`andrey.rsmax2` search within. Each takes any :class:`~andrey.core.ci.CITest`, and
background knowledge as :class:`~andrey.core.knowledge.Constraints`: each end of a required edge
is a member of the other's set from the start and never leaves it, and a pair the knowledge makes
non-adjacent is never in a returned set.

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
from andrey.core.knowledge import Constraints
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


def _required(constraints: Constraints | None, target: int, ranks: Sequence[int]) -> list[int]:
    """The variables a required edge joins to ``target``, in rank order; none without
    ``constraints``."""
    if constraints is None:
        return []
    joined = constraints.required[target] | constraints.required[:, target]
    return sorted(np.flatnonzero(joined).tolist(), key=lambda x: ranks[x])


def markov_blanket(
    test: CITest,
    target: int,
    d: int,
    alpha: float,
    ranks: Sequence[int],
    *,
    interleaved: bool = False,
    max_depth: int | None = None,
    constraints: Constraints | None = None,
) -> list[int]:
    """Estimate ``target``'s Markov blanket with IAMB, or with Inter-IAMB when ``interleaved``.

    Grow: add the candidate most dependent on ``target`` given the current blanket, among those
    whose p-value is at or below ``alpha``, until none is. Shrink: visit the members in the order
    they joined and drop each one the test finds independent of ``target`` (a p-value above
    ``alpha``) given the other current members. IAMB grows to the end and then shrinks once;
    Inter-IAMB shrinks after every addition and stops when no candidate qualifies or the blanket
    repeats one it held after an earlier shrink, so it always ends. Every grow test conditions on
    the whole blanket, so with ``max_depth`` the grow phase also stops once the blanket holds more
    than ``max_depth`` members, and every test conditions on at most ``max_depth`` variables.
    Returns the members in the order they joined.

    With ``constraints``, each variable a required edge joins to ``target`` is a member from the
    start, in rank order, and no shrink tests it. A pair they make non-adjacent stays a candidate,
    since a child's other parent belongs to the blanket without being adjacent and the later tests
    condition on it, and is dropped from the returned set.
    """
    dependence = _dependence(test)
    required = _required(constraints, target, ranks)
    blanket: list[int] = list(required)
    seen: set[frozenset[int]] = set()
    while max_depth is None or len(blanket) <= max_depth:
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
            _shrink(test, target, alpha, blanket, required)
            state = frozenset(blanket)
            if state in seen:
                break
            seen.add(state)
    if not interleaved:
        _shrink(test, target, alpha, blanket, required)
    if constraints is None:
        return blanket
    support = constraints.support()[target]
    return [x for x in blanket if support[x]]


def _shrink(
    test: CITest, target: int, alpha: float, blanket: list[int], keep: Sequence[int] = ()
) -> None:
    """Drop, in place and in joining order, each member independent of ``target`` given the rest;
    the members in ``keep`` are not tested."""
    for x in list(blanket):
        if x in keep:
            continue
        rest = [z for z in blanket if z != x]
        if test(target, x, rest) > alpha:
            blanket.remove(x)


def blanket_graph(
    test: CITest,
    d: int,
    alpha: float,
    ranks: Sequence[int],
    *,
    interleaved: bool = False,
    max_depth: int | None = None,
    constraints: Constraints | None = None,
) -> np.ndarray:
    """The moral graph: ``[i, j]`` when each of ``i`` and ``j`` is in the other's Markov blanket.

    Runs :func:`markov_blanket` for every variable, with ``constraints``; the result is a symmetric
    boolean ``(d, d)`` matrix with a ``False`` diagonal.
    """
    raw = np.zeros((d, d), dtype=bool)
    for i in range(d):
        for j in markov_blanket(
            test,
            i,
            d,
            alpha,
            ranks,
            interleaved=interleaved,
            max_depth=max_depth,
            constraints=constraints,
        ):
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
    if not xs:
        return []
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
    test: CITest,
    target: int,
    d: int,
    alpha: float,
    ranks: Sequence[int],
    max_depth: int | None = None,
    constraints: Constraints | None = None,
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
    subsets of the members that joined before it were tested in the forward phase. With
    ``max_depth``, both phases test only the conditioning sets of at most ``max_depth`` variables;
    the set itself may grow larger. Returns the members in the order they joined.

    With ``constraints``, each variable a required edge joins to ``target`` is a member from the
    start, in rank order, and neither phase tests it; a pair they make non-adjacent is never a
    candidate, as it cannot be adjacent.

    This is MMPC without its symmetry check: :func:`mmpc_skeleton` keeps ``x`` in ``target``'s set
    only when ``target`` is in ``x``'s.
    """
    required = _required(constraints, target, ranks)
    allowed = None if constraints is None else constraints.support()[target]
    others = [
        x for x in range(d) if x != target and x not in required and (allowed is None or allowed[x])
    ]
    weakest = dict(zip(others, _weakest(test, target, others, [()]), strict=True))
    members: list[int] = []

    def admit(new: int, pool: dict[int, tuple[float, float]]) -> None:
        """Weaken each association in ``pool`` by the conditioning sets that hold ``new``, then
        add ``new`` to the members."""
        top = len(members) if max_depth is None else min(len(members), max_depth - 1)
        conds = [
            tuple(sorted((*rest, new)))
            for size in range(top + 1)
            for rest in combinations(members, size)
        ]
        xs = list(pool)
        if conds:  # none when max_depth is 0
            for x, (p, strength) in zip(xs, _weakest(test, target, xs, conds), strict=True):
                pool[x] = (max(pool[x][0], p), min(pool[x][1], strength))
        members.append(new)

    for new in required:
        weakest = {x: w for x, w in weakest.items() if w[0] <= alpha}
        admit(new, weakest)
    while True:
        open_ = {x: w for x, w in weakest.items() if w[0] <= alpha}
        if not open_:
            break
        best = min(
            open_,
            key=lambda x: (round(open_[x][0], _KEY_DP), -round(open_[x][1], _KEY_DP), ranks[x]),
        )
        del open_[best]
        admit(best, open_)
        weakest = open_
    for x in list(members):
        if x in required:
            continue
        before = set(members[: members.index(x)])
        rest = sorted(z for z in members if z != x)
        top = len(rest) if max_depth is None else min(len(rest), max_depth)
        for size in range(1, top + 1):
            if scan_conditioning_sets(test, target, x, rest, size, alpha, set(), before):
                members.remove(x)
                break
    return members


def mmpc_skeleton(
    test: CITest,
    d: int,
    alpha: float,
    ranks: Sequence[int],
    max_depth: int | None = None,
    constraints: Constraints | None = None,
) -> np.ndarray:
    """MMPC's skeleton: ``[i, j]`` when each of ``i`` and ``j`` is in the other's parents and
    children (:func:`parents_children`, with ``constraints``), a symmetric boolean ``(d, d)``
    matrix."""
    raw = np.zeros((d, d), dtype=bool)
    for i in range(d):
        raw[i, parents_children(test, i, d, alpha, ranks, max_depth, constraints)] = True
    return raw & raw.T
