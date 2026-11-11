"""Markov blankets and parents-and-children sets, the engines behind :func:`andrey.iamb`,
:func:`andrey.inter_iamb`, and :func:`andrey.mmpc`.

:func:`markov_blanket` runs IAMB or Inter-IAMB for one target and :func:`parents_children` the
forward and backward phases of MMPC. :func:`blanket_graph` and :func:`mmpc_sets` run them for
every variable. Keeping a pair when each lies in the other's set (the AND rule) gives the moral
graph and the skeleton that :func:`andrey.calm`, :func:`andrey.mmhc`, and :func:`andrey.rsmax2`
search within. Each takes any :class:`~andrey.core.ci.CITest`, and background knowledge as
:class:`~andrey.core.knowledge.Constraints`: each end of a required edge is a member of the other's
set from the start and never leaves it, and a pair the knowledge makes non-adjacent is never in a
returned set.

:data:`LEARNERS` names each learner and its kind: a ``"blanket"`` learner returns a Markov blanket,
a ``"pc"`` learner the parents and children before the symmetry check, and a ``"pc_spouses"``
learner runs a ``"pc"`` learner and adds the spouses (:meth:`LocalSearch.spouses`). A
:class:`LocalSearch` runs the learners over one test, each learner for each target at most once,
and gives a target's parents and children, Markov blanket, and separating sets
(:meth:`LocalSearch.local`); :func:`local_search` builds one over the data. Every candidate a
learner's test drops keeps the conditioning set that separated it. :func:`local_structure` and
:func:`local_set` run a learner for every variable or for one target.

The structure learner of Margaritis (2003) turns the sets into a CPDAG: :func:`prune_spouses`
removes from the moral graph each pair that a subset of the smaller of the two blankets separates,
which gives the skeleton, and :func:`blanket_cpdag` decides each unshielded triple by testing its
endpoints given the middle node and each subset of the smaller blanket
(:func:`blanket_colliders`), orients the colliders with :mod:`andrey.constraint.pc`'s code, and
completes the result by Meek's rules.

Candidates are compared by the strength of their dependence on the target: the absolute partial
correlation for a test that reports one (``statistic_pvalue``), else the p-value. Strengths that
agree to ``KEY_DP`` decimals go to the candidate with the lower rank, from
:func:`~andrey.core.ties.tie_ranks`, so the sets do not depend on the column order.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import partial
from itertools import combinations, islice
from typing import Literal

import numpy as np

from andrey.constraint._skeleton import CHUNK, adjacent_pairs, scan_conditioning_sets
from andrey.core import ARROW, NULL, TAIL, backend
from andrey.core.ci import CITest, make_indep_test, pins_blas
from andrey.core.ci_missing import TestwiseDeletion
from andrey.core.knowledge import Constraints, orient
from andrey.core.orient import meek
from andrey.core.ties import (
    KEY_DP,
    resolve_collider_order,
    resolve_in_column_order,
    tie_mask,
    tie_ranks,
)

# The sets of one size a collider test asks first; each later chunk is twice as large.
_FIRST_CHUNK = 32
# ``(p-value, key)`` of ``x`` against the target given a conditioning set; a lower key is a
# stronger dependence.
_Dependence = Callable[[int, int, Sequence[int]], tuple[float, float]]
#: A separating set for each candidate a learner dropped, ascending, keyed by the candidate.
Sepsets = dict[int, tuple[int, ...]]


def _tests_every_set(test: CITest) -> bool:
    """Whether the searches test every set they reach without stopping early: a test that deletes
    rows per test, on data with missing values, counts each test's rows, which the fit's
    ``metadata["rows_tested"]`` summarizes."""
    return isinstance(test, TestwiseDeletion)


def _dependence(test: CITest) -> _Dependence:
    """The p-value and strength key of a query, read from ``statistic_pvalue`` where it exists."""
    statistic_pvalue = getattr(test, "statistic_pvalue", None)
    if statistic_pvalue is not None:

        def by_statistic(target: int, x: int, cond: Sequence[int]) -> tuple[float, float]:
            r, p = statistic_pvalue(target, x, cond)
            return p, -round(abs(r), KEY_DP)

        return by_statistic

    def by_pvalue(target: int, x: int, cond: Sequence[int]) -> tuple[float, float]:
        p = test(target, x, cond)
        return p, round(p, KEY_DP)

    return by_pvalue


def _dependences(
    test: CITest,
) -> Callable[[int, Sequence[int], Sequence[int]], list[tuple[float, float]]]:
    """:func:`_dependence` of every candidate in ``xs`` against one conditioning set.

    A test with ``pvalues_given`` and no ``statistic_pvalue`` scans the candidates together, which
    gives the p-values of the calls one by one and the key :func:`_dependence` derives from them.
    """
    scan = getattr(test, "pvalues_given", None)
    if scan is not None and getattr(test, "statistic_pvalue", None) is None:

        def by_scan(
            target: int, xs: Sequence[int], cond: Sequence[int]
        ) -> list[tuple[float, float]]:
            return [(p, round(p, KEY_DP)) for p in scan(target, xs, cond).tolist()]

        return by_scan
    dependence = _dependence(test)

    def one_by_one(
        target: int, xs: Sequence[int], cond: Sequence[int]
    ) -> list[tuple[float, float]]:
        return [dependence(target, x, cond) for x in xs]

    return one_by_one


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
    sepsets: Sepsets | None = None,
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

    With ``sepsets``, each variable outside the final blanket that some test found independent of
    ``target`` is set in it, in place, to the conditioning set of the last such test: the blanket
    of the grow step that left it out, or the other members when the shrink dropped it.
    """
    dependences = _dependences(test)
    required = _required(constraints, target, ranks)
    blanket: list[int] = list(required)
    seen: set[frozenset[int]] = set()
    while max_depth is None or len(blanket) <= max_depth:
        best_key, best_x = None, None
        candidates = [x for x in range(d) if x != target and x not in blanket]
        given = tuple(sorted(blanket))
        for x, (p, strength) in zip(
            candidates, dependences(target, candidates, blanket), strict=True
        ):
            key = (strength, ranks[x])
            if p <= alpha and (best_key is None or key < best_key):
                best_key, best_x = key, x
            if sepsets is not None and p > alpha:
                sepsets[x] = given
        if best_x is None:
            break
        blanket.append(best_x)
        if interleaved:
            _shrink(test, target, alpha, blanket, required, sepsets)
            state = frozenset(blanket)
            if state in seen:
                break
            seen.add(state)
    if not interleaved:
        _shrink(test, target, alpha, blanket, required, sepsets)
    if sepsets is not None:
        for x in blanket:
            sepsets.pop(x, None)
    if constraints is None:
        return blanket
    support = constraints.support()[target]
    return [x for x in blanket if support[x]]


def _shrink(
    test: CITest,
    target: int,
    alpha: float,
    blanket: list[int],
    keep: Sequence[int] = (),
    sepsets: Sepsets | None = None,
) -> None:
    """Drop, in place and in joining order, each member independent of ``target`` given the rest;
    the members in ``keep`` are not tested. With ``sepsets``, a dropped member's entry is set to the
    rest."""
    for x in list(blanket):
        if x in keep:
            continue
        rest = [z for z in blanket if z != x]
        if test(target, x, rest) > alpha:
            blanket.remove(x)
            if sepsets is not None:
                sepsets[x] = tuple(sorted(rest))


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
    test: CITest,
    target: int,
    xs: list[int],
    conds: list[tuple[int, ...]],
    sets: list[tuple[int, ...]] | None = None,
    ranks: Sequence[int] = (),
) -> list[tuple[float, float]]:
    """Each candidate's weakest association with ``target`` over the conditioning sets ``conds``.

    Returns ``(largest p-value, smallest strength)`` per entry of ``xs``. The strength is the
    absolute partial correlation for a test with ``statistic_pvalues``, evaluated for every query
    of one conditioning-set size in one call, else the negated p-value from ``batched_call`` or the
    scalar test. With ``sets``, each candidate's set with the largest p-value is appended to it
    (:func:`_largest_sets`, which reads ``ranks``).
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
            if sets is not None:
                sets.extend(_largest_sets(np.asarray(ps, dtype=np.float64)[None], conds, ranks))
        return out
    pmax = np.full(len(xs), -np.inf)
    rmin = np.full(len(xs), np.inf)
    by_size: dict[int, list[tuple[int, ...]]] = {}
    for cond in conds:
        by_size.setdefault(len(cond), []).append(cond)
    pvalues = []
    for size, group in by_size.items():
        cols = np.array(group, dtype=np.intp).reshape(len(group), size)
        x = np.repeat(np.asarray(xs, dtype=np.intp), len(group))
        r, p = batch(np.full(x.size, target, dtype=np.intp), x, np.tile(cols, (len(xs), 1)))
        if sets is not None:
            pvalues.append(np.asarray(p, dtype=np.float64).reshape(len(xs), len(group)))
        pmax = np.maximum(pmax, np.asarray(p).reshape(len(xs), len(group)).max(axis=1))
        rmin = np.minimum(rmin, np.abs(np.asarray(r)).reshape(len(xs), len(group)).min(axis=1))
    if sets is not None:
        sets.extend(
            _largest_sets(np.hstack(pvalues), [c for g in by_size.values() for c in g], ranks)
        )
    return list(zip(pmax.tolist(), rmin.tolist(), strict=True))


def _largest_sets(
    pvalues: np.ndarray, conds: list[tuple[int, ...]], ranks: Sequence[int]
) -> list[tuple[int, ...]]:
    """For each row of ``pvalues``, one p-value per set of ``conds``, the set with the largest
    p-value at ``KEY_DP`` decimals; of sets that agree, the smaller, then the one whose ascending
    ranks come first, so the choice does not depend on the column order."""
    rounded = np.round(pvalues, KEY_DP)
    tied = rounded == rounded.max(axis=1, keepdims=True)
    # A row of NaN ties nothing and takes its first set; NaN is not above alpha, so none reads it.
    out = [conds[j] for j in tied.argmax(axis=1).tolist()]
    for i in np.flatnonzero(tied.sum(axis=1) > 1).tolist():
        out[i] = min(
            (conds[j] for j in np.flatnonzero(tied[i]).tolist()),
            key=lambda c: (len(c), sorted(ranks[v] for v in c)),
        )
    return out


def parents_children(
    test: CITest,
    target: int,
    d: int,
    alpha: float,
    ranks: Sequence[int],
    max_depth: int | None = None,
    constraints: Constraints | None = None,
    sepsets: Sepsets | None = None,
) -> list[int]:
    """Estimate ``target``'s parents and children with the forward and backward phases of MMPC.

    A candidate's association with ``target`` is its weakest over the conditioning sets drawn from
    the current set (:func:`_weakest`): the largest p-value among them, then, where two candidates'
    largest p-values agree to ``KEY_DP`` decimals, the smallest strength, then the rank. Forward:
    add the candidate whose weakest association is the strongest, until every candidate has a
    conditioning set with a p-value above ``alpha``; such a candidate is dropped for good, since
    adding members only weakens an association. Each addition tests every remaining candidate
    against the conditioning sets that hold the new member, by increasing size, and stops testing
    a candidate after the first size at which a set separates it. Backward: visit the members in the
    order they joined and drop each one some subset of the other current members separates; the
    subsets of the members that joined before it were tested in the forward phase. With
    ``max_depth``, both phases test only the conditioning sets of at most ``max_depth`` variables;
    the set itself may grow larger. A test that deletes rows per test keeps testing a separated
    candidate at every size (:func:`_tests_every_set`). Returns the members in the order they
    joined.

    With ``constraints``, each variable a required edge joins to ``target`` is a member from the
    start, in rank order, and neither phase tests it; a pair they make non-adjacent is never a
    candidate, as it cannot be adjacent.

    This is MMPC without its symmetry check, which keeps ``x`` in ``target``'s set only when
    ``target`` is in ``x``'s; :func:`local_structure` applies it to :func:`mmpc_sets`.

    With ``sepsets``, each candidate either phase drops is set in it, in place, to a set that
    separated it, taken at the first size at which a set did: in the forward phase the set with
    the largest p-value (:func:`_largest_sets`), in the backward phase the separating subset with
    the largest p-value, ties broken alike.
    """
    required = _required(constraints, target, ranks)
    every = _tests_every_set(test)
    allowed = None if constraints is None else constraints.support()[target]
    others = [
        x for x in range(d) if x != target and x not in required and (allowed is None or allowed[x])
    ]
    weakest = dict(zip(others, _weakest(test, target, others, [()]), strict=True))
    if sepsets is not None:
        sepsets.update({x: () for x, (p, _) in weakest.items() if p > alpha})
    members: list[int] = []

    def admit(new: int, pool: dict[int, tuple[float, float]]) -> None:
        """Weaken each association in ``pool`` by the conditioning sets that hold ``new``, size by
        size, then add ``new`` to the members. A candidate separated at one size is not tested at
        the next: it leaves the pool, and only the candidates that stay are ranked."""
        top = len(members) if max_depth is None else min(len(members), max_depth - 1)
        xs = list(pool)
        for size in range(top + 1):  # none when max_depth is 0
            conds = [tuple(sorted((*rest, new))) for rest in combinations(members, size)]
            largest: list[tuple[int, ...]] = []
            weakened = _weakest(
                test, target, xs, conds, None if sepsets is None else largest, ranks
            )
            for i, (x, (p, strength)) in enumerate(zip(xs, weakened, strict=True)):
                pool[x] = (max(pool[x][0], p), min(pool[x][1], strength))
                if sepsets is not None and pool[x][0] > alpha:
                    sepsets.setdefault(x, largest[i])
            if not every:
                xs = [x for x in xs if pool[x][0] <= alpha]
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
            key=lambda x: (round(open_[x][0], KEY_DP), -round(open_[x][1], KEY_DP), ranks[x]),
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
            trace: list[dict] | None = None if sepsets is None else []
            if scan_conditioning_sets(test, target, x, rest, size, alpha, set(), before, trace):
                members.remove(x)
                if sepsets is not None and trace is not None:
                    found = [t for t in trace if t["removed"]]
                    p = np.array([[t["p"] for t in found]])
                    [sepsets[x]] = _largest_sets(p, [t["S"] for t in found], ranks)
                break
    return members


def mmpc_sets(
    test: CITest,
    d: int,
    alpha: float,
    ranks: Sequence[int],
    max_depth: int | None = None,
    constraints: Constraints | None = None,
) -> np.ndarray:
    """Every variable's parents and children before the symmetry check: ``[i, j]`` when ``j`` is
    in ``i``'s set (:func:`parents_children`, with ``constraints``), a boolean ``(d, d)``
    matrix."""
    raw = np.zeros((d, d), dtype=bool)
    for i in range(d):
        raw[i, parents_children(test, i, d, alpha, ranks, max_depth, constraints)] = True
    return raw


def reach(sets: np.ndarray) -> np.ndarray:
    """Each variable's set together with the sets of its members, less the variable itself: the
    blanket MMPC's sets stand in for, which holds each child's other parents."""
    out = sets | (sets.astype(np.float32) @ sets.astype(np.float32) > 0)
    np.fill_diagonal(out, False)
    return out


def _smaller(blankets: np.ndarray, a: int, b: int, drop: Sequence[int] = ()) -> list[list[int]]:
    """The candidate sets of the pair ``a``, ``b``: the smaller of the two rows of ``blankets``
    less ``a``, ``b``, and ``drop``, ascending, or both when they hold as many members."""
    sides = [
        [v for v in np.flatnonzero(blankets[e]).tolist() if v not in (a, b, *drop)] for e in (a, b)
    ]
    if len(sides[0]) != len(sides[1]):
        return [min(sides, key=len)]
    return sides if sides[0] != sides[1] else sides[:1]


def prune_spouses(
    test: CITest,
    moral: np.ndarray,
    alpha: float,
    max_depth: int | None = None,
    constraints: Constraints | None = None,
) -> np.ndarray:
    """The skeleton: the ``moral`` graph less every pair that some subset of the smaller of the
    two blankets separates (Margaritis 2003).

    For each pair ``x``, ``y`` of ``moral``, the candidates are the smaller of ``moral[x]`` less
    ``y`` and ``moral[y]`` less ``x``, or both when they hold as many members; the pair leaves when
    ``x`` and ``y`` test independent (a p-value above ``alpha``) given some subset of them, the
    empty set included, by increasing size up to ``max_depth``. The blankets are the moral graph
    the search starts from, so each pair's verdict does not depend on another's. A pair a required
    edge of ``constraints`` joins is not tested. Returns a new symmetric boolean matrix.
    """
    skeleton = moral.copy()
    required = None if constraints is None else constraints.required | constraints.required.T
    for x, y in adjacent_pairs(moral):
        if required is not None and required[x, y]:
            continue
        sides = _smaller(moral, x, y)
        top = max(len(side) for side in sides)
        for size in range(top + 1 if max_depth is None else min(top, max_depth) + 1):
            seen: set[int] | None = None
            separated = False
            for side in sides:
                if size <= len(side) and scan_conditioning_sets(
                    test, x, y, side, size, alpha, set(), seen
                ):
                    separated = True
                    break
                seen = set(side)
            if separated:
                skeleton[x, y] = skeleton[y, x] = False
                break
    return skeleton


def _collider_sets(sides: list[list[int]], size: int, middle: int) -> Iterator[tuple[int, ...]]:
    """The conditioning sets of a collider test, each once: ``middle`` with every size-``size``
    subset of ``sides[0]``, then of the other side, less the subsets the first side holds."""
    first = set(sides[0])
    for n, side in enumerate(sides):
        for rest in combinations(side, size):
            if n == 0 or not first.issuperset(rest):
                yield tuple(sorted((*rest, middle)))


def blanket_colliders(
    test: CITest,
    skeleton: np.ndarray,
    blankets: np.ndarray,
    alpha: float,
    max_depth: int | None = None,
) -> tuple[
    dict[tuple[int, int], tuple[int, ...]],
    dict[tuple[int, int, int], float],
    frozenset[tuple[int, int, int]],
]:
    """Decide every unshielded triple of ``skeleton`` by the collider test of Margaritis (2003).

    For a triple ``a - b - c``, ``a < c``, with ``a`` and ``c`` not adjacent, the candidates are the
    smaller of ``blankets[a]`` and ``blankets[c]`` less ``a``, ``b``, and ``c``, or both when they
    hold as many members. ``b`` is a collider when ``a`` and ``c`` test dependent (a p-value at or
    below ``alpha``) given ``b`` together with every subset of them, and a non-collider when some
    such set separates them. Every conditioning set holds ``b``, so with ``max_depth`` the subsets
    hold at most ``max_depth - 1`` members, and with ``max_depth=0`` no set can be tested and the
    triple is ambiguous. The sets are tested by increasing size, and a non-collider stops at the
    chunk that holds its first separating set: the sets of one size are generated and tested in
    chunks of ``_FIRST_CHUNK`` sets, then twice as many each time, at most ``CHUNK``. A size of
    at most ``_FIRST_CHUNK`` sets takes one call, a larger one skips the chunks after the one that
    holds the first separating set, and the memory a triple takes does not grow with the number of
    its subsets. A test that deletes rows per test tests every set of each size, in chunks of
    ``CHUNK`` (:func:`_tests_every_set`).

    Returns the separating sets :func:`~andrey.constraint.pc.orient_colliders` reads, keyed
    ``(a, c)`` and holding the pair's non-collider middle nodes; each decided triple's rank, the
    weakest association over its tests (:func:`_weakest`: the smallest absolute partial correlation
    for a test that reports one, else the negated largest p-value), so the most clearly dependent
    collider comes first; and the ambiguous triples, keyed ``(a, b, c)``. A collider's rank covers
    every set; a non-collider's covers the sets it tested, and orients nothing.
    """
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {}
    strengths: dict[tuple[int, int, int], float] = {}
    ambiguous: set[tuple[int, int, int]] = set()
    every = _tests_every_set(test)
    for b in range(skeleton.shape[0]):
        for a, c in combinations(np.flatnonzero(skeleton[b]).tolist(), 2):
            if skeleton[a, c]:
                continue
            sepsets.setdefault((a, c), ())
            sides = _smaller(blankets, a, c, (b,))
            top = max(len(side) for side in sides)
            if max_depth is not None:
                top = min(top, max_depth - 1)
            if top < 0:
                ambiguous.add((a, b, c))
                continue
            p, strength = -np.inf, np.inf
            for size in range(top + 1):
                conds = _collider_sets(sides, size, b)
                width = CHUNK if every else min(_FIRST_CHUNK, CHUNK)
                while (every or p <= alpha) and (chunk := list(islice(conds, width))):
                    [(p_chunk, strength_chunk)] = _weakest(test, a, [c], chunk)
                    p, strength = max(p, p_chunk), min(strength, strength_chunk)
                    width = min(2 * width, CHUNK)
                if p > alpha:
                    sepsets[a, c] = (*sepsets[a, c], b)
                    break
            strengths[a, b, c] = strength
    return sepsets, strengths, frozenset(ambiguous)


def blanket_cpdag(
    test: CITest,
    skeleton: np.ndarray,
    blankets: np.ndarray,
    alpha: float,
    max_depth: int | None = None,
    constraints: Constraints | None = None,
) -> np.ndarray:
    """The CPDAG of the structure learner of Margaritis (2003), as an endpoint-mark matrix.

    The unshielded triples of the boolean ``skeleton`` are decided by :func:`blanket_colliders`
    against ``blankets``. As in :func:`andrey.constraint.pc.pc`, every edge ``constraints`` decide
    is directed first (:func:`~andrey.constraint.pc.check_knowledge_cycle` raises on a directed
    cycle), the colliders are oriented by
    :func:`~andrey.constraint.pc.orient_colliders` in the order ``andrey.config.tie_break``
    selects, Meek's rules complete the result, with R4 under ``constraints``, and an edge two
    colliders of equal rank dispute is left undirected.
    """
    # pc imports this module through andrey.search, so its import waits for the call.
    from andrey.constraint.pc import check_knowledge_cycle, orient_colliders, undirect_conflicts

    sepsets, strengths, ambiguous = blanket_colliders(test, skeleton, blankets, alpha, max_depth)
    marks = np.where(skeleton, TAIL, NULL).astype(np.int8)
    if constraints is not None:
        marks = orient(marks, constraints)
        check_knowledge_cycle((marks == TAIL) & (marks.T == ARROW))
    pdag = orient_colliders(
        marks,
        sepsets,
        {},
        order=resolve_collider_order(None),
        ambiguous=ambiguous,
        ranks=strengths,
    )
    cpdag = meek(
        pdag,
        rule4=constraints is not None,
        ambiguous=ambiguous,
        in_column_order=resolve_in_column_order(),
    )
    return undirect_conflicts(cpdag)


def _ranked_test(
    X: np.ndarray, indep_test: str, categorical: np.ndarray | None, seed: int | None
) -> tuple[CITest, list[int]]:
    """The ``indep_test`` test, seeded by ``seed``, over ``X`` and each column's
    :func:`~andrey.core.ties.tie_ranks`, the columns the test reads as categories ranked as
    categories (:func:`~andrey.core.ties.tie_mask`)."""
    test = make_indep_test(indep_test, X, categorical=categorical, seed=seed)
    return test, tie_ranks(X, tie_mask(indep_test, categorical, X.shape[1])).tolist()


#: Runs one learner for one target of a :class:`LocalSearch` and returns the target's set in the
#: order its members joined, adding to the :data:`Sepsets` each candidate a test dropped.
Search = Callable[["LocalSearch", int, Sepsets], list[int]]


@dataclass(frozen=True)
class Learner:
    """One row of :data:`LEARNERS`.

    ``kind`` is ``"blanket"`` for a learner whose ``search`` returns a Markov blanket, ``"pc"`` for
    one whose ``search`` returns the parents and children before the symmetry check, and
    ``"pc_spouses"`` for one that runs the ``"pc"`` learner named ``pc`` and adds the spouses
    (:meth:`LocalSearch.spouses`). A ``search`` records a separating set for every candidate it
    drops after a test, so that :meth:`LocalSearch.local` has one for each.
    """

    kind: Literal["blanket", "pc", "pc_spouses"]
    search: Search | None = None
    pc: str | None = None

    def __post_init__(self) -> None:
        spouses = self.kind == "pc_spouses"
        if spouses == (self.search is not None) or spouses != (self.pc is not None):
            raise ValueError("a 'pc_spouses' learner names its 'pc' learner; another has a search")


def _iamb(search: LocalSearch, target: int, sepsets: Sepsets, *, interleaved: bool) -> list[int]:
    return markov_blanket(
        search.test,
        target,
        search.d,
        search.alpha,
        search.ranks,
        interleaved=interleaved,
        max_depth=search.max_depth,
        constraints=search.constraints,
        sepsets=sepsets,
    )


def _mmpc(search: LocalSearch, target: int, sepsets: Sepsets) -> list[int]:
    return parents_children(
        search.test,
        target,
        search.d,
        search.alpha,
        search.ranks,
        search.max_depth,
        search.constraints,
        sepsets,
    )


#: Each learner by the name of the method that runs it.
LEARNERS: dict[str, Learner] = {
    "iamb": Learner("blanket", partial(_iamb, interleaved=False)),
    "inter_iamb": Learner("blanket", partial(_iamb, interleaved=True)),
    "mmpc": Learner("pc", _mmpc),
}


class LocalSearch:
    """The learners of :data:`LEARNERS` over one test, each run at most once per target.

    ``test`` answers every query of every run; ``ranks`` orders candidates whose strengths agree
    (:func:`~andrey.core.ties.tie_ranks`); ``alpha``, ``max_depth``, and ``constraints`` are as
    for :func:`markov_blanket` and :func:`parents_children`. ``levels`` holds each column's number
    of categories, ``0`` for a column the test reads as continuous, or is ``None`` when it reads
    none as categories, and ``n`` is the number of rows: what :func:`table_size` and a learner's
    sparse-table rule read. A run, and the spouses :meth:`local` finds, are kept for the life of
    the object, so the symmetry check, the spouse step, and a caller that asks for a target again
    read them without a test. A test with BLAS calls gives the same bits on every host inside
    :meth:`pinned`.
    """

    def __init__(
        self,
        test: CITest,
        ranks: Sequence[int],
        alpha: float,
        *,
        max_depth: int | None = None,
        constraints: Constraints | None = None,
        levels: np.ndarray | None = None,
        n: int = 0,
    ) -> None:
        self.test = test
        self.ranks = ranks
        self.d = len(ranks)
        self.alpha = alpha
        self.max_depth = max_depth
        self.constraints = constraints
        self.levels = levels
        self.n = n
        self._runs: dict[tuple[str, int], tuple[list[int], Sepsets]] = {}
        self._spouses: dict[tuple[str, int], list[int]] = {}

    def pinned(self) -> AbstractContextManager:
        """BLAS limited to one thread when the test needs it (:func:`~andrey.core.ci.pins_blas`),
        a block that does nothing otherwise."""
        return backend.one_blas_thread() if pins_blas(self.test) else nullcontext()

    def run(self, method: str, target: int) -> tuple[list[int], Sepsets]:
        """``target``'s set by the ``"blanket"`` or ``"pc"`` learner ``method``, in joining order,
        and the separating set of each candidate its tests dropped.

        The first call runs the search; a later one returns the same two objects, which the caller
        must not modify.
        """
        key = (method, target)
        if key not in self._runs:
            search = LEARNERS[method].search
            if search is None:
                raise ValueError(f"{method!r} has no search of its own; ask LocalSearch.local")
            sepsets: Sepsets = {}
            self._runs[key] = (search(self, target, sepsets), sepsets)
        return self._runs[key]

    def parents_children(self, method: str, target: int) -> list[int]:
        """``target``'s parents and children by the ``"pc"`` learner ``method``: the members of its
        set whose own set holds ``target`` (the symmetry check), in joining order."""
        members, _ = self.run(method, target)
        return [x for x in members if target in self.run(method, x)[0]]

    def local(self, method: str, target: int) -> tuple[list[int] | None, list[int] | None, Sepsets]:
        """``(pc, mb, sepsets)`` of ``target`` by the learner ``method``, each new.

        ``pc`` is the parents and children (:meth:`parents_children`), ``None`` for a
        ``"blanket"`` learner. ``mb`` is the Markov blanket: the set of a ``"blanket"`` learner,
        the parents and children followed by the spouses (:meth:`spouses`) for a ``"pc_spouses"``
        learner, and ``None`` for a ``"pc"`` learner. Both are in joining order. ``sepsets`` maps
        each variable outside ``pc``, or outside ``mb`` for a ``"blanket"`` learner, that a test
        separated from ``target`` to a set that did: the one ``target``'s run recorded or, for a
        member the symmetry check dropped, the one that member's run recorded for ``target``. A
        variable no test separated, as one the knowledge makes non-adjacent, has no entry.
        """
        learner = LEARNERS[method]
        if learner.kind == "blanket":
            members, sepsets = self.run(method, target)
            return None, list(members), dict(sepsets)
        name = method if learner.pc is None else learner.pc
        pc = self.parents_children(name, target)
        members, own = self.run(name, target)
        sepsets = dict(own)
        for x in members:
            if x not in pc:
                sepsets[x] = self.run(name, x)[1][target]
        if learner.kind == "pc":
            return pc, None, sepsets
        if (method, target) not in self._spouses:
            self._spouses[method, target] = self.spouses(name, target, pc, sepsets)
        return pc, pc + self._spouses[method, target], sepsets

    def spouses(self, method: str, target: int, pc: list[int], sepsets: Sepsets) -> list[int]:
        """The spouses of ``target``, in the order they join, given its parents and children
        ``pc`` and the separating sets ``sepsets`` of :meth:`local`.

        A candidate is a variable outside ``pc`` and other than ``target`` in the parents and
        children of some ``y`` in ``pc``, both by the ``"pc"`` learner ``method``. It joins when a
        test finds it dependent on ``target`` (a p-value at or below ``alpha``) given its set in
        ``sepsets`` together with such a ``y`` [1]_. The ``y`` are visited in the order of ``pc``
        and each one's parents and children in joining order, and a candidate joins at the first
        ``y`` that admits it. Under faithfulness, two non-adjacent variables' separating set holds
        each non-collider between them and no collider, so the test is dependent exactly when
        ``y`` is their common child. Pairing a candidate only with a ``y`` adjacent to it, rather
        than with every member of ``pc``, is the correction of Pena et al. [2]_.

        A ``y`` already in the candidate's separating set leaves that set unchanged, so it admits
        nothing and is not tested. With ``max_depth``, a test that would condition on more
        variables is not run. A candidate without a separating set, one the knowledge makes
        non-adjacent to ``target``, never joins: such a pair is left out of every returned set.

        References
        ----------
        .. [1] Aliferis, C. F., Statnikov, A., Tsamardinos, I., Mani, S., and Koutsoukos, X. D.
           (2010). Local causal and Markov blanket induction for causal discovery and feature
           selection for classification part I: Algorithms and empirical evaluation. Journal of
           Machine Learning Research 11, 171-234.
        .. [2] Pena, J. M., Nilsson, R., Bjorkegren, J., and Tegner, J. (2007). Towards scalable
           and data efficient learning of Markov boundaries. International Journal of Approximate
           Reasoning 45(2), 211-232.
        """
        found: list[int] = []
        for y in pc:
            for x in self.parents_children(method, y):
                if x == target or x in pc or x in found or x not in sepsets or y in sepsets[x]:
                    continue
                cond = tuple(sorted((*sepsets[x], y)))
                if self.max_depth is not None and len(cond) > self.max_depth:
                    continue
                if self.test(target, x, cond) <= self.alpha:
                    found.append(x)
        return found

    def structure(self, method: str, structure: str) -> np.ndarray:
        """Every variable's set by the learner ``method`` joined into the graph ``structure``
        names, as an endpoint-mark matrix (:func:`local_structure`)."""
        learner = LEARNERS[method]
        name = method if learner.pc is None else learner.pc
        sets = self._sets(lambda i: self.run(name, i)[0])
        if learner.kind == "pc":
            skeleton, blankets = sets & sets.T, reach(sets)
        elif learner.kind == "blanket":
            blankets = sets & sets.T
            skeleton = (
                blankets
                if structure == "moral"
                else prune_spouses(
                    self.test, blankets, self.alpha, self.max_depth, self.constraints
                )
            )
        else:
            mb = self._sets(lambda i: self.local(method, i)[1] or [])
            blankets = mb & mb.T
            skeleton = blankets if structure == "moral" else sets & sets.T
        if structure != "cpdag":
            return np.where(skeleton, TAIL, NULL).astype(np.int8)
        return blanket_cpdag(
            self.test, skeleton, blankets, self.alpha, self.max_depth, self.constraints
        )

    def _sets(self, members: Callable[[int], list[int]]) -> np.ndarray:
        """``[i, j]`` when ``j`` is in ``members(i)``, a boolean ``(d, d)`` matrix."""
        raw = np.zeros((self.d, self.d), dtype=bool)
        for i in range(self.d):
            raw[i, members(i)] = True
        return raw


def local_search(
    X: np.ndarray,
    *,
    alpha: float,
    indep_test: str,
    categorical: np.ndarray | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    constraints: Constraints | None = None,
) -> LocalSearch:
    """A :class:`LocalSearch` over the ``indep_test`` test of ``X``.

    ``categorical`` marks the categorical columns for a mixed-data test; ``seed`` fixes the draws
    of a stochastic test (``None`` uses ``ANDREY_SEED``); ``alpha``, ``max_depth``, and
    ``constraints`` are as :class:`LocalSearch` takes them. A column the test reads as categories
    (:func:`~andrey.core.ties.tie_mask`) has as many levels as distinct observed values.
    """
    test, ranks = _ranked_test(X, indep_test, categorical, seed)
    mask = tie_mask(indep_test, categorical, X.shape[1])
    levels = None
    if mask.any():
        levels = np.array(
            [np.unique(c[~np.isnan(c)]).size if m else 0 for c, m in zip(X.T, mask, strict=True)]
        )
    return LocalSearch(
        test,
        ranks,
        alpha,
        max_depth=max_depth,
        constraints=constraints,
        levels=levels,
        n=X.shape[0],
    )


def local_structure(
    X: np.ndarray,
    method: str,
    structure: str,
    *,
    alpha: float,
    indep_test: str,
    categorical: np.ndarray | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    constraints: Constraints | None = None,
) -> np.ndarray:
    """Every variable's set by the learner ``method`` of :data:`LEARNERS` joined into the graph
    ``structure`` names, as an endpoint-mark matrix.

    The sets of a ``"blanket"`` learner (IAMB, Inter-IAMB) join a pair when each variable is in the
    other's Markov blanket: the moral graph, ``"moral"``, undirected; ``"skeleton"`` is the moral
    graph less the pairs :func:`prune_spouses` separates. A ``"pc"`` learner (MMPC) gives the
    skeleton by the AND rule over its sets, and a ``"pc_spouses"`` learner the skeleton from its
    parents and children and the moral graph from its blankets, both by the AND rule.
    ``"cpdag"`` orients the skeleton (:func:`blanket_cpdag`) against the blankets: the moral graph,
    or for a ``"pc"`` learner each variable's set together with its members' sets before the
    symmetry check (:func:`reach`). The other arguments are :func:`local_search`'s.
    """
    search = local_search(
        X,
        alpha=alpha,
        indep_test=indep_test,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
        constraints=constraints,
    )
    with search.pinned():
        return search.structure(method, structure)


def local_set(
    X: np.ndarray,
    method: str,
    target: int,
    *,
    alpha: float,
    indep_test: str,
    categorical: np.ndarray | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    constraints: Constraints | None = None,
) -> list[int]:
    """``target``'s set by the learner ``method`` of :data:`LEARNERS`, in the order its members
    joined: the Markov blanket of a ``"blanket"`` or ``"pc_spouses"`` learner, and the parents and
    children of a ``"pc"`` learner, the members of ``target``'s set whose own set holds ``target``
    (the symmetry check). The other arguments are :func:`local_search`'s.
    """
    search = local_search(
        X,
        alpha=alpha,
        indep_test=indep_test,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
        constraints=constraints,
    )
    with search.pinned():
        pc, mb, _ = search.local(method, target)
    found = pc if mb is None else mb
    assert found is not None  # every learner learns one of the two
    return found


def table_size(
    levels: np.ndarray | Sequence[int], x: int, y: int, cond: Sequence[int], *, df: bool = False
) -> int | None:
    """The cells of the contingency table that a categorical test of ``x`` and ``y`` given
    ``cond`` counts, or with ``df`` its degrees of freedom; ``None`` when a column is continuous.

    ``levels`` holds each column's number of categories, ``0`` for a continuous column
    (:attr:`LocalSearch.levels`). The cells are the product of the levels of ``x``, ``y``, and
    ``cond``; the degrees of freedom are ``(levels[x] - 1) * (levels[y] - 1)`` times the product of
    the levels of ``cond``, as for a table without structural zeros [1]_. A sparse-table rule
    compares the rows of the data with this size; the authors of MMPC ran a test only when the
    data held at least five rows on average per cell [1]_. ``iamb``, ``inter_iamb``, and ``mmpc``
    run every test whatever its size.

    References
    ----------
    .. [1] Tsamardinos, I., Brown, L. E., and Aliferis, C. F. (2006). The max-min hill-climbing
       Bayesian network structure learning algorithm. Machine Learning 65(1), 31-78.
    """
    if any(levels[c] == 0 for c in (x, y, *cond)):
        return None
    rest = math.prod(int(levels[c]) for c in cond)
    if df:
        return (int(levels[x]) - 1) * (int(levels[y]) - 1) * rest
    return int(levels[x]) * int(levels[y]) * rest
