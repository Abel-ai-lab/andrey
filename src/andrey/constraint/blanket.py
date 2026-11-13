"""Markov blankets and parents-and-children sets, the engines behind :func:`andrey.iamb`,
:func:`andrey.inter_iamb`, :func:`andrey.gs`, :func:`andrey.fast_iamb`, :func:`andrey.mmpc`,
:func:`andrey.hiton_pc`, :func:`andrey.hiton_mb`, :func:`andrey.mmmb`, and :func:`andrey.pcmb`.

:func:`markov_blanket` runs IAMB or Inter-IAMB for one target, the ``"gs"`` and ``"fast_iamb"``
learners Grow-Shrink and Fast-IAMB, the ``"iamb_forward"`` learner IAMB's grow phase alone, the
``"total_conditioning"`` learner a test of each variable given all the others,
:func:`parents_children` the forward and backward phases of MMPC, the ``"hiton_pc"`` learner
semi-interleaved HITON-PC, the ``"hiton_mb"`` learner semi-interleaved HITON-MB, HITON-PC
followed by the spouse step, the ``"mmmb"`` learner MMMB, MMPC followed by the same step, the
``"get_pcd"`` learner PCMB's GetPCD, and the ``"pcmb"`` learner PCMB, GetPCD followed by the same
step.
:func:`blanket_graph` and :func:`mmpc_sets` run them for every variable. Keeping a pair when each
lies in the other's set (the AND rule) gives the moral graph and the skeleton that
:func:`andrey.calm`, :func:`andrey.mmhc`, and :func:`andrey.rsmax2` search within. Each takes any
:class:`~andrey.core.ci.CITest`, and background knowledge as
:class:`~andrey.core.knowledge.Constraints`: each end of a required edge is a member of the other's
set from the start and never leaves it, and a pair the knowledge makes non-adjacent is never in a
returned set.

:data:`LEARNERS` names each learner and its kind: a ``"blanket"`` learner returns a Markov blanket
(``"iamb_forward"`` a set that holds it under correct tests), a ``"pc"`` learner the parents and
children before the symmetry check, and a ``"pc_spouses"`` learner runs a ``"pc"`` learner and
adds the spouses (:meth:`LocalSearch.spouses`). A
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

A learner asks its tests through its :class:`LocalSearch`. :meth:`LocalSearch.scan` tests many
candidates against one conditioning set, in one call where the test can; :meth:`LocalSearch.weakest`
gives each candidate's weakest association over many sets (a max-min step); and
:meth:`LocalSearch.separating` gives the sets of one size that separate a pair.
:class:`SearchCounts` adds up the queries they ask and the time they take. Candidates are compared
by the strength of their dependence on the target: the absolute partial correlation for a test that
reports one (``statistic_pvalue``), else the p-value. Strengths that agree go to the candidate with
the lower rank, from :func:`~andrey.core.ties.tie_ranks`, which :func:`strongest` and
:func:`ordered` read only for candidates that tie, so the sets do not depend on the column order.
Every grow step of IAMB, Inter-IAMB, Grow-Shrink, and Fast-IAMB orders its candidates with
:meth:`LocalSearch.by_strength`, which compares a partial correlation to ``KEY_DP`` decimals and a
p-value to ``KEY_DP`` significant digits.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from itertools import combinations, islice
from time import perf_counter
from typing import Any, Literal

import numpy as np

from andrey.constraint._skeleton import CHUNK, adjacent_pairs, scan_conditioning_sets
from andrey.core import ARROW, NULL, TAIL, backend
from andrey.core.ci import CITest, counting_tests, make_indep_test, pins_blas
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
#: A separating set for each candidate a learner dropped, ascending, keyed by the candidate.
Sepsets = dict[int, tuple[int, ...]]
#: A query as the tests canonicalize it: the lower and the higher variable and the sorted set.
Query = tuple[int, int, tuple[int, ...]]


def _tests_every_set(test: CITest) -> bool:
    """Whether the searches test every set they reach without stopping early: a test that deletes
    rows per test, on data with missing values, counts each test's rows, which the fit's
    ``metadata["rows_tested"]`` summarizes."""
    return isinstance(test, TestwiseDeletion)


def _scan(
    test: CITest, target: int, xs: Sequence[int], cond: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """:meth:`LocalSearch.scan` on ``test``: the strengths and p-values of ``target`` against each
    of ``xs`` given ``cond``.

    A test with ``statistic_pvalue`` (a partial-correlation test) answers two or more candidates in
    one ``statistic_pvalues`` call, whose entries equal the calls one by one bit for bit, and its
    strength is the absolute partial correlation. Another test's strength is the negated p-value; a
    test with ``pvalues_given`` (the chi-square tests) answers two or more candidates in one call,
    with the same bits, and any other test answers them one call at a time.
    """
    one = getattr(test, "statistic_pvalue", None)
    if one is not None:
        many = getattr(test, "statistic_pvalues", None)
        if many is not None and len(xs) > 1:
            k = len(xs)
            sets = np.tile(np.asarray(cond, dtype=np.intp).reshape(1, -1), (k, 1))
            r, p = many(np.full(k, target, dtype=np.intp), np.asarray(xs, dtype=np.intp), sets)
            return np.abs(np.asarray(r, dtype=np.float64)), np.asarray(p, dtype=np.float64)
        answers = np.array([one(target, x, cond) for x in xs], dtype=np.float64).reshape(-1, 2)
        return np.abs(answers[:, 0]), answers[:, 1]
    given = getattr(test, "pvalues_given", None)
    if given is not None and len(xs) > 1:
        p = np.asarray(given(target, xs, cond), dtype=np.float64)
    else:
        p = np.array([test(target, x, cond) for x in xs], dtype=np.float64)
    return -p, p


def strongest(keys: Mapping[int, Any], ranks: Sequence[int]) -> int:
    """The candidate with the smallest key, the lowest rank among candidates whose keys are equal.

    ``keys`` maps each candidate to its key, smaller for a stronger candidate, such as a negated
    strength rounded to ``KEY_DP`` decimals or a tuple of rounded values; candidates are visited in
    the order of ``keys``. ``ranks`` (:func:`~andrey.core.ties.tie_ranks`) is read only for two
    candidates whose keys are equal, so the choice does not depend on the column order, and a
    search whose keys never tie never reads it. The choice is the first candidate, in the order of
    ``keys``, whose key followed by its rank is the smallest. Raises ``ValueError`` when ``keys``
    is empty.
    """
    if not keys:
        raise ValueError("no candidate to choose from")
    items = iter(keys.items())
    best, best_key = next(items)
    for x, key in items:
        if key == best_key:
            if ranks[x] < ranks[best]:
                best, best_key = x, key
        elif key < best_key:
            best, best_key = x, key
    return best


def ordered(keys: Mapping[int, Any], ranks: Sequence[int]) -> list[int]:
    """The candidates from the smallest key to the largest, those whose keys are equal by rank.

    The order :func:`strongest` gives when asked again for what is left: ``keys`` and ``ranks`` are
    as there, and ``ranks`` is read only for candidates whose keys are equal.
    """
    items = sorted(keys.items(), key=lambda item: item[1])
    order: list[int] = []
    start = 0
    while start < len(items):
        stop = start + 1
        while stop < len(items) and items[stop][1] == items[start][1]:
            stop += 1
        tied = [x for x, _ in items[start:stop]]
        order.extend(tied if len(tied) == 1 else sorted(tied, key=lambda x: ranks[x]))
        start = stop
    return order


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
    search = LocalSearch(test, ranks, alpha, max_depth=max_depth, constraints=constraints)
    search.d = d
    return _grow_shrink(search, target, sepsets, interleaved=interleaved)


def _grow_shrink(
    search: LocalSearch,
    target: int,
    sepsets: Sepsets | None,
    *,
    interleaved: bool,
    shrink: bool = True,
) -> list[int]:
    """:func:`markov_blanket` on the engine: each grow step is one :meth:`LocalSearch.scan` of the
    candidates given the blanket, and :meth:`LocalSearch.by_strength` puts first the dependent
    candidate with the strongest dependence. Without ``shrink``, IAMB's grow phase alone: no member
    is tested again, so the set may hold variables outside the blanket."""
    alpha, max_depth, constraints = search.alpha, search.max_depth, search.constraints
    required = _required(constraints, target, search.ranks)
    blanket: list[int] = list(required)
    seen: set[frozenset[int]] = set()
    while max_depth is None or len(blanket) <= max_depth:
        candidates = [x for x in range(search.d) if x != target and x not in blanket]
        strengths, pvalues = search.scan(target, candidates, blanket)
        dependent = np.flatnonzero(pvalues <= alpha).tolist()
        if sepsets is not None:
            given = tuple(sorted(blanket))
            sepsets.update((candidates[i], given) for i in np.flatnonzero(pvalues > alpha).tolist())
        if not dependent:
            break
        xs = [candidates[i] for i in dependent]
        blanket.append(search.by_strength(xs, strengths[dependent])[0])
        if interleaved:
            _shrink(search, target, blanket, required, sepsets)
            state = frozenset(blanket)
            if state in seen:
                break
            seen.add(state)
    if shrink and not interleaved:
        _shrink(search, target, blanket, required, sepsets)
    if sepsets is not None:
        for x in blanket:
            sepsets.pop(x, None)
    if constraints is None:
        return blanket
    support = constraints.support()[target]
    return [x for x in blanket if support[x]]


def _shrink(
    search: LocalSearch,
    target: int,
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
        if search.scan(target, [x], rest)[1][0] > search.alpha:
            blanket.remove(x)
            if sepsets is not None:
                sepsets[x] = tuple(sorted(rest))


def _gs(search: LocalSearch, target: int, sepsets: Sepsets) -> list[int]:
    """``target``'s Markov blanket by Grow-Shrink (GS) [1]_ [2]_, in the order its members joined.

    The candidates are visited in one fixed order, the most dependent on ``target`` first: by their
    unconditional strength (:meth:`LocalSearch.scan`, :meth:`LocalSearch.by_strength`), ties by
    rank.
    Grow: each pass visits every candidate outside the blanket in that order, tests it given the
    current blanket, and adds it at once when the p-value is at or below ``alpha``, so the next
    candidate of the pass is tested given the larger blanket; passes repeat until one adds nothing.
    Shrink: each pass visits the members in the same order and drops each one the test finds
    independent of ``target`` given the other current members; passes repeat until one drops
    nothing. A test whose conditioning set has not changed since the candidate's last test is not
    asked again, and while the blanket is still empty the grow test is the ordering test itself.

    ``max_depth``, ``constraints``, and ``sepsets`` act as in :func:`markov_blanket`: the grow
    stops once the blanket holds more than ``max_depth`` members, a member a required edge joins to
    ``target`` is in the blanket from the start and never tested, a pair the knowledge makes
    non-adjacent is left out of the returned set, and a variable outside it that a test found
    independent keeps the conditioning set of the last such test.

    References
    ----------
    .. [1] Margaritis, D., and Thrun, S. (2000). Bayesian network induction via local
       neighborhoods. Advances in Neural Information Processing Systems 12, 505-511.
    .. [2] Margaritis, D. (2003). Learning Bayesian network model structure from data. PhD thesis,
       Carnegie Mellon University, CMU-CS-03-153.
    """
    alpha, max_depth, constraints = search.alpha, search.max_depth, search.constraints
    required = _required(constraints, target, search.ranks)
    blanket = list(required)
    candidates = [x for x in range(search.d) if x != target and x not in required]
    strengths, pvalues = search.scan(target, candidates, ())
    order = search.by_strength(candidates, strengths)
    unconditional = dict(zip(candidates, pvalues.tolist(), strict=True))
    # The blanket size at each candidate's last independent verdict: the blanket only grows here,
    # so an equal size is the same conditioning set.
    last: dict[int, int] = {}
    growing = True
    while growing:
        added = False
        for y in order:
            if y in blanket or last.get(y) == len(blanket):
                continue
            if max_depth is not None and len(blanket) > max_depth:
                growing = False
                break
            if blanket:
                p = search.scan(target, [y], blanket)[1][0]
            else:
                p = unconditional[y]
            if p <= alpha:
                blanket.append(y)
                added = True
            else:
                last[y] = len(blanket)
                sepsets[y] = tuple(sorted(blanket))
        growing = growing and added
    # The removals made when each member was last found dependent: the blanket only shrinks here.
    kept: dict[int, int] = {}
    removed = 0
    while True:
        before = removed
        for x in [y for y in order if y in blanket]:
            if kept.get(x) == removed:
                continue
            rest = [z for z in blanket if z != x]
            if search.scan(target, [x], rest)[1][0] > alpha:
                blanket.remove(x)
                removed += 1
                sepsets[x] = tuple(sorted(rest))
            else:
                kept[x] = removed
        if removed == before:
            break
    for x in blanket:
        sepsets.pop(x, None)
    if constraints is None:
        return blanket
    support = constraints.support()[target]
    return [x for x in blanket if support[x]]


#: Fast-IAMB's least average number of rows per cell of the table of a test it adds a member by
#: (Yaramakala and Margaritis 2005, Eq. 2, k = 5).
CELL_ROWS = 5


def _fast_iamb(search: LocalSearch, target: int, sepsets: Sepsets) -> list[int]:
    """``target``'s Markov blanket by Fast-IAMB [1]_, in the order its members joined.

    Each round tests every variable outside the blanket given the blanket in one
    :meth:`LocalSearch.scan`, the first round given no variable, and orders those whose p-value is
    at or below ``alpha`` from the strongest dependence to the weakest
    (:meth:`LocalSearch.by_strength`). It adds them in that order, without a test, while the data
    suffice: on categorical data, while the rows per cell of the table of ``target``, the variable,
    and the blanket so far exceed ``CELL_ROWS``, the cells the product of the columns' levels
    (:func:`table_size`) and the rows every row of the data; a variable that fails the rule marks
    the data insufficient and ends the batch. On a table with a continuous column the rule does not
    apply and the whole batch joins. Then one shrink pass in joining order (:func:`_shrink`). The
    search stops when no variable tests dependent; when the data were insufficient and the shrink
    dropped nothing, the variables the rule kept out taken as independent; or when the blanket after
    a shrink repeats one it held after an earlier shrink, which ends a cycle of additions and
    removals the paper does not guard against.

    ``max_depth``, ``constraints``, and ``sepsets`` act as in :func:`markov_blanket`: a batch stops
    once the blanket holds more than ``max_depth`` members, and the search stops while it does, so
    no test conditions on more than ``max_depth`` variables.

    References
    ----------
    .. [1] Yaramakala, S., and Margaritis, D. (2005). Speculative Markov blanket discovery for
       optimal feature selection. Fifth IEEE International Conference on Data Mining (ICDM).
    """
    alpha, max_depth, constraints = search.alpha, search.max_depth, search.constraints
    required = _required(constraints, target, search.ranks)
    blanket: list[int] = list(required)
    seen: set[frozenset[int]] = set()
    while max_depth is None or len(blanket) <= max_depth:
        candidates = [x for x in range(search.d) if x != target and x not in blanket]
        strengths, pvalues = search.scan(target, candidates, blanket)
        given = tuple(sorted(blanket))
        sepsets.update((candidates[i], given) for i in np.flatnonzero(pvalues > alpha).tolist())
        dependent = np.flatnonzero(pvalues <= alpha).tolist()
        if not dependent:
            break
        batch = search.by_strength([candidates[i] for i in dependent], strengths[dependent])
        insufficient = False
        for x in batch:
            if max_depth is not None and len(blanket) > max_depth:
                break
            size = None if search.levels is None else table_size(search.levels, target, x, blanket)
            if size is not None and search.n <= CELL_ROWS * size:
                insufficient = True
                break
            blanket.append(x)
        before = len(blanket)
        _shrink(search, target, blanket, required, sepsets)
        if insufficient and len(blanket) == before:
            break
        state = frozenset(blanket)
        if state in seen:
            break
        seen.add(state)
    for x in blanket:
        sepsets.pop(x, None)
    if constraints is None:
        return blanket
    support = constraints.support()[target]
    return [x for x in blanket if support[x]]


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
    search = LocalSearch(test, ranks, alpha, max_depth=max_depth, constraints=constraints)
    search.d = d
    raw = np.zeros((d, d), dtype=bool)
    for i in range(d):
        raw[i, _grow_shrink(search, i, None, interleaved=interleaved)] = True
    return raw & raw.T


def _weakest(
    test: CITest,
    target: int,
    xs: Sequence[int],
    conds: Sequence[tuple[int, ...]],
    pvalues: list[np.ndarray] | None = None,
) -> list[tuple[float, float]]:
    """Each candidate's weakest association with ``target`` over the conditioning sets ``conds``.

    Returns ``(largest p-value, smallest strength)`` per entry of ``xs``. The strength is the
    absolute partial correlation for a test with ``statistic_pvalues``, evaluated for every query
    of one conditioning-set size in one call, else the negated p-value from ``batched_call`` or the
    scalar test. With ``pvalues``, the p-values are appended to it as one ``(len(xs),
    len(conds))`` array, a row per candidate and a column per set in the order of ``conds``.
    """
    if not xs:
        return []
    batch = getattr(test, "statistic_pvalues", None)
    if batch is None:
        batched = getattr(test, "batched_call", None)
        out = []
        found = []
        for x in xs:
            ps = (
                batched(target, x, conds)
                if batched is not None
                else [test(target, x, c) for c in conds]
            )
            top = float(max(ps))
            out.append((top, -top))
            found.append(ps)
        if pvalues is not None:
            pvalues.append(np.array(found, dtype=np.float64).reshape(len(xs), len(conds)))
        return out
    pmax = np.full(len(xs), -np.inf)
    rmin = np.full(len(xs), np.inf)
    by_size: dict[int, list[int]] = {}
    for j, cond in enumerate(conds):
        by_size.setdefault(len(cond), []).append(j)
    rows = None if pvalues is None else np.empty((len(xs), len(conds)))
    for size, at in by_size.items():
        cols = np.array([conds[j] for j in at], dtype=np.intp).reshape(len(at), size)
        x = np.repeat(np.asarray(xs, dtype=np.intp), len(at))
        r, p = batch(np.full(x.size, target, dtype=np.intp), x, np.tile(cols, (len(xs), 1)))
        p = np.asarray(p, dtype=np.float64).reshape(len(xs), len(at))
        if rows is not None:
            rows[:, at] = p
        pmax = np.maximum(pmax, p.max(axis=1))
        rmin = np.minimum(rmin, np.abs(np.asarray(r)).reshape(len(xs), len(at)).min(axis=1))
    if pvalues is not None and rows is not None:
        pvalues.append(rows)
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
    search = LocalSearch(test, ranks, alpha, max_depth=max_depth, constraints=constraints)
    search.d = d
    return _max_min(search, target, sepsets)


def _max_min(search: LocalSearch, target: int, sepsets: Sepsets | None) -> list[int]:
    """:func:`parents_children` on the engine: each candidate's association is
    :meth:`LocalSearch.weakest`, :meth:`LocalSearch.strongest` picks the candidate to add, and the
    backward phase asks :meth:`LocalSearch.separating`."""
    alpha, ranks, max_depth, constraints = (
        search.alpha,
        search.ranks,
        search.max_depth,
        search.constraints,
    )
    required = _required(constraints, target, ranks)
    every = _tests_every_set(search.test)
    allowed = None if constraints is None else constraints.support()[target]
    others = [
        x
        for x in range(search.d)
        if x != target and x not in required and (allowed is None or allowed[x])
    ]
    weakest = dict(zip(others, search.weakest(target, others, [()]), strict=True))
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
            found: list[np.ndarray] = []
            weakened = search.weakest(target, xs, conds, None if sepsets is None else found)
            separated = []
            for i, (x, (p, strength)) in enumerate(zip(xs, weakened, strict=True)):
                pool[x] = (max(pool[x][0], p), min(pool[x][1], strength))
                if sepsets is not None and pool[x][0] > alpha and x not in sepsets:
                    separated.append(i)
            if sepsets is not None and separated:
                sets = search.largest_sets(found[0][separated], conds)
                sepsets.update(zip([xs[i] for i in separated], sets, strict=True))
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
        best = search.strongest(
            {x: (round(p, KEY_DP), -round(strength, KEY_DP)) for x, (p, strength) in open_.items()}
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
            separating = search.separating(target, x, rest, size, before)
            if separating:
                members.remove(x)
                if sepsets is not None:
                    pvalues = np.array([[p for _, p in separating]])
                    [sepsets[x]] = search.largest_sets(pvalues, [s for s, _ in separating])
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
    search = LocalSearch(test, ranks, alpha, max_depth=max_depth, constraints=constraints)
    search.d = d
    raw = np.zeros((d, d), dtype=bool)
    for i in range(d):
        raw[i, _max_min(search, i, None)] = True
    return raw


#: HITON-PC's least number of rows per cell of a categorical test's table for the test to run
#: (Aliferis et al. 2010, p. 201, the parameter h-ps, whose value the paper leaves open; 5 is the
#: authors' value).
HITON_CELL_ROWS = 5


def _hiton_pc(search: LocalSearch, target: int, sepsets: Sepsets) -> list[int]:
    """``target``'s parents and children by semi-interleaved HITON-PC without its symmetry check
    [1]_, in the order the members joined.

    The candidates are the variables whose unconditional test with ``target`` finds them dependent
    (a p-value at or below ``alpha``), ordered once by that test, the most dependent first: by the
    p-value to ``KEY_DP`` significant digits, then, where p-values agree, as when they underflow to
    zero, by the larger statistic (:meth:`LocalSearch.weakest`), then by rank. The candidates join
    the set one at a time in that order, and each newcomer is tested against the subsets of the
    other members, by increasing size, and leaves for good at the first size at which a subset
    separates it; the members before it are not tested then. Once every candidate has joined, each
    member, in joining order, is tested against the subsets of the other current members, and
    leaves when one separates it, so a later member's tests read the set without the members
    dropped before it. A member skips the subsets of the members it was tested against when it
    joined, so the last one is not tested again. With ``max_depth``, a test conditions on at most
    ``max_depth`` variables; the set itself may grow larger.

    On categorical data a test runs only when the data hold at least ``HITON_CELL_ROWS`` rows per
    cell of its table (:func:`table_size`) [1]_: a candidate whose unconditional test fails the
    rule is taken as independent of ``target`` without a test, and a conditioning set that fails it
    is not tested, so it removes nothing.

    ``constraints`` act as in :func:`parents_children`: each variable a required edge joins to
    ``target`` is a member from the start, in rank order, and never tested, and a pair the knowledge
    makes non-adjacent is never a candidate. Each candidate dropped keeps in ``sepsets`` the set
    that separated it, the one with the largest p-value at the first size that separates
    (:meth:`LocalSearch.largest_sets`), and ``()`` for one dropped by its unconditional test or the
    rule.

    References
    ----------
    .. [1] Aliferis, C. F., Statnikov, A., Tsamardinos, I., Mani, S., and Koutsoukos, X. D.
       (2010). Local causal and Markov blanket induction for causal discovery and feature
       selection for classification part I: Algorithms and empirical evaluation. Journal of
       Machine Learning Research 11, 171-234.
    """
    alpha, max_depth, constraints = search.alpha, search.max_depth, search.constraints
    required = _required(constraints, target, search.ranks)
    allowed = None if constraints is None else constraints.support()[target]

    def reliable(x: int, cond: tuple[int, ...]) -> bool:
        return search.reliable(target, x, cond, cell_rows=HITON_CELL_ROWS)

    def separating(x: int, among: list[int], tested: set[int]) -> tuple[int, ...] | None:
        """The set that separates ``x`` from ``target`` at the first size at which one does,
        among the subsets of ``among`` not inside ``tested`` that pass the rule; ``None`` when
        none does."""
        among = sorted(among)
        top = len(among) if max_depth is None else min(len(among), max_depth)
        for size in range(1, top + 1):
            conds = [
                c for c in combinations(among, size) if not tested.issuperset(c) and reliable(x, c)
            ]
            if not conds:
                continue
            found: list[np.ndarray] = []
            search.weakest(target, [x], conds, found)
            row = found[0][0]
            at = np.flatnonzero(row > alpha)
            if at.size:
                [z] = search.largest_sets(row[at].reshape(1, -1), [conds[j] for j in at])
                return z
        return None

    others = []
    for x in range(search.d):
        if x == target or x in required or (allowed is not None and not allowed[x]):
            continue
        if reliable(x, ()):
            others.append(x)
        else:
            sepsets[x] = ()
    keys: dict[int, tuple[float, float]] = {}
    for x, (p, strength) in zip(others, search.weakest(target, others, [()]), strict=True):
        if p <= alpha:
            keys[x] = (float(f"{p:.{KEY_DP}g}"), -round(strength, KEY_DP))
        elif p > alpha:
            sepsets[x] = ()
    order = search.ordered(keys)
    members = list(required)
    # The members each candidate was tested against when it joined.
    seen: dict[int, set[int]] = {}
    for new in order:
        rest = list(members)
        found = separating(new, rest, set())
        if found is None:
            members.append(new)
            seen[new] = set(rest)
        else:
            sepsets[new] = found
    for x in list(members):
        if x in required:
            continue
        found = separating(x, [z for z in members if z != x], seen.get(x, set()))
        if found is not None:
            members.remove(x)
            sepsets[x] = found
    return members


#: PCMB's least number of rows per degree of freedom of a categorical test for the test to run
#: (Pena et al. 2007, pp. 5 and 12).
PCMB_DF_ROWS = 5


def _get_pcd(search: LocalSearch, target: int, sepsets: Sepsets) -> list[int]:
    """``target``'s parents and children by GetPCD of PCMB [1]_, the set before PCMB's symmetry
    check, in the order the members joined.

    A candidate's separating set is, among the subsets of the current members, the one with the
    largest p-value, and its association with ``target`` the weakest over those subsets. Each pass
    of the search runs three steps [1]_. First, every candidate whose separating set separates it
    (a p-value above ``alpha``) leaves for good. Second, the candidate whose weakest association
    is the strongest joins without a further test: by the largest p-value to ``KEY_DP``
    significant digits, the smallest first, then, where p-values agree, as when they underflow to
    zero, by the larger smallest strength (:meth:`LocalSearch.weakest`), then by rank. Third, each
    earlier member is tested against the subsets of the other members, and every member some
    subset separates leaves for good: all of them at once, each judged against the members before
    any leaves. The newcomer is not tested then: its subsets are the ones it passed as a candidate.
    So the set changes on every pass, and the search stops when no candidate is left; then no
    member joins and none leaves, so the paper's rule, repeat "until PCD does not change", stops
    there too. Each test runs once: a pass tests only the subsets that hold the newest member, and
    keeps the others from earlier passes, less those that hold a member that left. With
    ``max_depth``, a test conditions on at most ``max_depth`` variables; the set itself may grow
    larger.

    On categorical data a test runs only when the data hold at least ``PCMB_DF_ROWS`` rows per
    degree of freedom of its table (:func:`table_size` with ``df=True``) [1]_: a subset that fails
    the rule is not tested and takes no part, and a candidate whose unconditional test fails it is
    never tested and never joins.

    ``constraints`` act as in :func:`parents_children`: each variable a required edge joins to
    ``target`` is a member from the start, in rank order, and never tested, and a pair the knowledge
    makes non-adjacent is never a candidate. Each variable that leaves keeps in ``sepsets`` the
    separating set it left by, of sets whose p-values agree to ``KEY_DP`` decimals the smaller,
    then the one whose ascending ranks come first (:meth:`LocalSearch.largest_sets`).

    References
    ----------
    .. [1] Pena, J. M., Nilsson, R., Bjorkegren, J., and Tegner, J. (2007). Towards scalable and
       data efficient learning of Markov boundaries. International Journal of Approximate
       Reasoning 45(2), 211-232.
    """
    alpha, max_depth, constraints = search.alpha, search.max_depth, search.constraints
    required = _required(constraints, target, search.ranks)
    allowed = None if constraints is None else constraints.support()[target]
    # The p-value and the strength of each variable's test given each subset of the members.
    tested: dict[int, dict[tuple[int, ...], tuple[float, float]]] = {}
    candidates: list[int] = []
    for x in range(search.d):
        if x == target or x in required or (allowed is not None and not allowed[x]):
            continue
        if search.reliable(target, x, (), df_rows=PCMB_DF_ROWS):
            candidates.append(x)
            tested[x] = {}
    members = list(required)

    def measure(xs: list[int], newest: int | None) -> None:
        """Test each of ``xs`` given each subset of the other members that holds ``newest`` (every
        subset when ``newest`` is ``None``), of at most ``max_depth`` variables, that passes the
        rule, listing only the sizes whose smallest table passes it; candidates with the same
        subsets are tested together, one subset at a time."""
        groups: dict[tuple[tuple[int, ...], ...], list[int]] = {}
        for x in xs:
            rest = sorted(z for z in members if z not in (x, newest))
            top = len(rest) if max_depth is None else max_depth - (newest is not None)
            forced = () if newest is None else (newest,)
            top = min(top, search.reliable_depth(target, x, rest, forced, df_rows=PCMB_DF_ROWS))
            conds = [
                tuple(sorted((*s, newest) if newest is not None else s))
                for size in range(min(top, len(rest)) + 1)
                for s in combinations(rest, size)
            ]
            usable = tuple(c for c in conds if search.reliable(target, x, c, df_rows=PCMB_DF_ROWS))
            groups.setdefault(usable, []).append(x)
        for conds, group in groups.items():
            for cond in conds:
                for x, weak in zip(group, search.weakest(target, group, [cond]), strict=True):
                    tested[x][cond] = weak

    def separating(x: int) -> tuple[int, ...] | None:
        """``x``'s separating set when it separates ``x`` from ``target``, else ``None``."""
        conds = list(tested[x])
        pvalues = np.array([tested[x][c][0] for c in conds])
        if not pvalues.max() > alpha:
            return None
        [found] = search.largest_sets(pvalues.reshape(1, -1), conds)
        return found

    measure(candidates, None)
    while True:
        dropped = {x: z for x in candidates if (z := separating(x)) is not None}
        sepsets.update(dropped)
        candidates = [x for x in candidates if x not in dropped]
        for x in dropped:
            del tested[x]
        if not candidates:
            return members
        keys = {}
        for x in candidates:
            weak = tested[x].values()
            p, strength = max(p for p, _ in weak), min(s for _, s in weak)
            keys[x] = (float(f"{p:.{KEY_DP}g}"), -round(strength, KEY_DP))
        newest = search.strongest(keys)
        candidates.remove(newest)
        earlier = [x for x in members if x not in required]
        members.append(newest)
        measure(earlier, newest)
        left = {x: z for x in earlier if (z := separating(x)) is not None}
        if left:
            sepsets.update(left)
            members = [x for x in members if x not in left]
            for x in left:
                del tested[x]
            for weak in tested.values():
                for cond in [c for c in weak if not left.keys().isdisjoint(c)]:
                    del weak[cond]
        measure(candidates, newest)


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
    drops after a test, so that :meth:`LocalSearch.local` has one for each. ``cell_rows`` and
    ``df_rows``, for a ``"pc_spouses"`` learner, are the least numbers of rows per cell and per
    degree of freedom of a categorical spouse test's table for the test to run
    (:meth:`LocalSearch.reliable`); ``None`` sets no such bound.
    """

    kind: Literal["blanket", "pc", "pc_spouses"]
    search: Search | None = None
    pc: str | None = None
    cell_rows: int | None = None
    df_rows: int | None = None

    def __post_init__(self) -> None:
        spouses = self.kind == "pc_spouses"
        if spouses == (self.search is not None) or spouses != (self.pc is not None):
            raise ValueError("a 'pc_spouses' learner names its 'pc' learner; another has a search")
        if (self.cell_rows is not None or self.df_rows is not None) and not spouses:
            raise ValueError("only a 'pc_spouses' learner takes cell_rows or df_rows")


def _iamb(search: LocalSearch, target: int, sepsets: Sepsets, *, interleaved: bool) -> list[int]:
    return _grow_shrink(search, target, sepsets, interleaved=interleaved)


def _iamb_forward(search: LocalSearch, target: int, sepsets: Sepsets) -> list[int]:
    """IAMB's grow phase without its shrink, the blanket learner of MB-by-MB's runs (Wang, Zhou,
    Zhao, and Geng 2014, Sections 3.4 and 5): a variable added in error stays, so the set holds
    the blanket under correct tests and may hold more."""
    return _grow_shrink(search, target, sepsets, interleaved=False, shrink=False)


def _total_conditioning(search: LocalSearch, target: int, sepsets: Sepsets) -> list[int]:
    """``target``'s Markov blanket by total conditioning (TC) [1]_: every other variable that the
    test finds dependent on ``target`` given all the rest, from the strongest dependence to the
    weakest (:meth:`LocalSearch.by_strength`), ties by rank.

    Under faithfulness these are exactly the blanket's members: the variables outside the blanket
    are independent of ``target`` given the blanket, and each member is dependent given the rest
    [1]_. The same holds with hidden variables, for the blanket in the maximal ancestral graph over
    the observed variables, whose separation keeps those properties [2]_. Each variable left out
    keeps all the rest as its separating set. A member a required edge joins to ``target`` is in
    the blanket from the start and never tested, and a pair the knowledge makes non-adjacent is
    tested, since it may be a child's other parent, and left out of the returned set, as in
    :func:`markov_blanket`. Every test conditions on ``d - 2`` variables, so with ``max_depth``
    below that no test runs and every variable stays in the set, as a skeleton search keeps an
    edge that only a larger set would remove.

    References
    ----------
    .. [1] Pellet, J.-P., and Elisseeff, A. (2008). Using Markov blankets for causal structure
       learning. Journal of Machine Learning Research 9, 1295-1342.
    .. [2] Pellet, J.-P., and Elisseeff, A. (2008). Finding latent causes in causal networks: an
       efficient approach based on Markov blankets. Advances in Neural Information Processing
       Systems 21, 1249-1256.
    """
    required = _required(search.constraints, target, search.ranks)
    others = [x for x in range(search.d) if x != target and x not in required]
    if search.max_depth is not None and search.d - 2 > search.max_depth:
        blanket = required + sorted(others, key=lambda x: search.ranks[x])
    else:
        dependent, strengths = [], []
        for x in others:
            rest = [z for z in range(search.d) if z not in (target, x)]
            strength, pvalue = search.scan(target, [x], rest)
            if pvalue[0] <= search.alpha:
                dependent.append(x)
                strengths.append(strength[0])
            else:
                sepsets[x] = tuple(rest)
        blanket = required + search.by_strength(dependent, np.asarray(strengths))
    if search.constraints is None:
        return blanket
    support = search.constraints.support()[target]
    return [x for x in blanket if support[x]]


def _mmpc(search: LocalSearch, target: int, sepsets: Sepsets) -> list[int]:
    return _max_min(search, target, sepsets)


#: Each learner by the name of the method that runs it.
LEARNERS: dict[str, Learner] = {
    "iamb": Learner("blanket", partial(_iamb, interleaved=False)),
    "inter_iamb": Learner("blanket", partial(_iamb, interleaved=True)),
    "iamb_forward": Learner("blanket", _iamb_forward),
    "mmpc": Learner("pc", _mmpc),
    "gs": Learner("blanket", _gs),
    "fast_iamb": Learner("blanket", _fast_iamb),
    "hiton_pc": Learner("pc", _hiton_pc),
    "hiton_mb": Learner("pc_spouses", pc="hiton_pc", cell_rows=HITON_CELL_ROWS),
    "total_conditioning": Learner("blanket", _total_conditioning),
    "mmmb": Learner("pc_spouses", pc="mmpc"),
    "get_pcd": Learner("pc", _get_pcd),
    "pcmb": Learner("pc_spouses", pc="get_pcd", df_rows=PCMB_DF_ROWS),
}


@dataclass(slots=True)
class SearchCounts:
    """The queries a :class:`LocalSearch` asked and the time it took, over its learners' runs and
    spouse steps.

    ``issued`` counts the queries the learners asked through :meth:`LocalSearch.scan`,
    :meth:`LocalSearch.weakest`, and :meth:`LocalSearch.separating`, one per candidate and
    conditioning set, each time it is asked, whether the test computes its answer or reads it from
    an answer it keeps. ``computed`` counts the tests computed during the runs and spouse steps as
    :func:`~andrey.core.ci.counting_tests` counts them, the unit of ``metadata["n_tests"]``: a query
    answered from a kept answer, as the chi-square tests keep their marginal answers, counts
    nothing, and one a batch computes past the point where the search stops reading counts.
    ``queries`` holds the distinct queries among the issued ones, each the lower and the higher
    variable and the sorted conditioning set, when the search keeps them (``trace=True``), and is
    ``None`` otherwise; ``distinct`` is their number. ``test_seconds`` is the time spent in those
    query calls and ``search_seconds`` the time of the runs and spouse steps, the query calls
    included, so the rest of the search took their difference.
    """

    issued: int = 0
    computed: int = 0
    queries: set[Query] | None = None
    test_seconds: float = 0.0
    search_seconds: float = 0.0

    @property
    def distinct(self) -> int | None:
        """The number of distinct queries among the issued ones, ``None`` when not kept."""
        return None if self.queries is None else len(self.queries)


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

    A learner asks its tests through :meth:`scan`, :meth:`weakest`, and :meth:`separating`, and
    picks between candidates with :meth:`strongest`, which reads ``ranks`` only when two
    candidates tie; it reads neither the test's internals nor ``ranks`` itself. ``counts``
    (:class:`SearchCounts`) adds up the queries and the time; with ``trace``, it also keeps the
    distinct queries, which costs memory in proportion to their number.
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
        trace: bool = False,
    ) -> None:
        self.test = test
        self.ranks = ranks
        self.d = len(ranks)
        self.alpha = alpha
        self.max_depth = max_depth
        self.constraints = constraints
        self.levels = levels
        self.n = n
        self.counts = SearchCounts(queries=set() if trace else None)
        # Whether a scan's strengths are negated p-values, as for a test without statistic_pvalue.
        self._pvalue_strengths = getattr(test, "statistic_pvalue", None) is None
        self._measuring = False
        self._runs: dict[tuple[str, int], tuple[list[int], Sepsets]] = {}
        self._spouses: dict[tuple[str, int], list[int]] = {}

    def scan(
        self, target: int, xs: Sequence[int], cond: Sequence[int] = ()
    ) -> tuple[np.ndarray, np.ndarray]:
        """The strength and the p-value of ``target`` against each candidate in ``xs`` given the
        one conditioning set ``cond``: two float arrays in the order of ``xs``.

        A larger strength is a stronger dependence: the absolute partial correlation for a test
        that reports one (``statistic_pvalue``), else the negated p-value, so that ``-round(s,
        KEY_DP)`` is a key for :meth:`strongest`. The candidates are tested together where the test
        can (a partial-correlation test's ``statistic_pvalues``, a chi-square test's
        ``pvalues_given``), else one call at a time; either way each entry equals the test's answer
        to that query alone, bit for bit. ``xs`` excludes ``target`` and the members of ``cond``.
        """
        start = perf_counter()
        strengths, pvalues = _scan(self.test, target, xs, cond)
        self._asked(start, target, xs, 1, lambda: [cond])
        return strengths, pvalues

    def weakest(
        self,
        target: int,
        xs: Sequence[int],
        conds: Sequence[tuple[int, ...]],
        pvalues: list[np.ndarray] | None = None,
    ) -> list[tuple[float, float]]:
        """Each candidate's weakest association with ``target`` over the conditioning sets
        ``conds``, as ``(largest p-value, smallest strength)`` in the order of ``xs``: the step of
        a max-min search (:func:`_weakest`).

        Here the strength is the absolute statistic for a test with ``statistic_pvalues``, every
        query of one conditioning-set size answered in one call, else the negated p-value. With
        ``pvalues``, the p-values are appended to it as one array, a row per candidate and a column
        per set, for :meth:`largest_sets`.
        """
        start = perf_counter()
        out = _weakest(self.test, target, xs, conds, pvalues)
        self._asked(start, target, xs, len(conds), lambda: conds)
        return out

    def largest_sets(
        self, pvalues: np.ndarray, conds: Sequence[tuple[int, ...]]
    ) -> list[tuple[int, ...]]:
        """For each row of ``pvalues``, one p-value per set of ``conds``, the set with the largest
        p-value at ``KEY_DP`` decimals; of sets that agree, the smaller, then the one whose members'
        ranks, ascending, come first, so ``ranks`` is read only when sets tie
        (:func:`_largest_sets`)."""
        return _largest_sets(np.asarray(pvalues, dtype=np.float64), list(conds), self.ranks)

    def separating(
        self,
        target: int,
        x: int,
        among: Sequence[int],
        size: int,
        skip_within: set[int] | None = None,
    ) -> list[tuple[tuple[int, ...], float]]:
        """The sets of ``size`` members of ``among`` that separate ``x`` from ``target`` (a p-value
        above ``alpha``), each with its p-value, in the order of ``combinations``.

        Every such set is tested, those lying inside ``skip_within`` excepted, in chunks of one
        call where the test has ``batched_call``
        (:func:`~andrey.constraint._skeleton.scan_conditioning_sets`). ``among`` is ascending and
        excludes ``target`` and ``x``. A test that stops its shuffles once the side of ``alpha`` is
        certain may report a bound in place of a p-value above ``alpha``.
        """
        start = perf_counter()
        trace: list[dict] = []
        scan_conditioning_sets(
            self.test, target, x, list(among), size, self.alpha, set(), skip_within, trace
        )
        self._asked(start, target, [x], len(trace), lambda: (record["S"] for record in trace))
        return [(record["S"], record["p"]) for record in trace if record["removed"]]

    def reliable(
        self,
        x: int,
        y: int,
        cond: Sequence[int],
        *,
        cell_rows: int | None = None,
        df_rows: int | None = None,
    ) -> bool:
        """Whether a sparse-table rule lets the test of ``x`` and ``y`` given ``cond`` run: on
        categorical columns, the ``n`` rows are at least ``cell_rows`` times the cells of its table
        and at least ``df_rows`` times its degrees of freedom (:func:`table_size`); a test that
        reads a continuous column, or a bound that is ``None``, always passes."""
        if self.levels is None:
            return True
        for rows, df in ((cell_rows, False), (df_rows, True)):
            if rows is not None:
                size = table_size(self.levels, x, y, cond, df=df)
                if size is not None and self.n < rows * size:
                    return False
        return True

    def reliable_depth(
        self, x: int, y: int, among: Sequence[int], forced: Sequence[int] = (), *, df_rows: int
    ) -> int:
        """The most variables of ``among`` that can join ``forced`` in a conditioning set of ``x``
        and ``y`` and still pass the rule of :meth:`reliable` with ``df_rows``: the set of the
        fewest categories grows until its table has too many degrees of freedom for the rows, and a
        larger set only adds categories. ``len(among)`` when no bound follows, on a test that
        reads a continuous column or a column of one category."""
        if self.levels is None or any(self.levels[c] == 0 for c in (x, y, *forced, *among)):
            return len(among)
        size = (int(self.levels[x]) - 1) * (int(self.levels[y]) - 1)
        size *= math.prod(int(self.levels[c]) for c in forced)
        if size == 0:
            return len(among)
        depth = 0
        for categories in sorted(int(self.levels[c]) for c in among):
            size *= categories
            if self.n < df_rows * size:
                break
            depth += 1
        return depth

    def strongest(self, keys: Mapping[int, Any]) -> int:
        """The candidate with the smallest key, ties broken by ``ranks`` (:func:`strongest`)."""
        return strongest(keys, self.ranks)

    def ordered(self, keys: Mapping[int, Any]) -> list[int]:
        """The candidates from the smallest key to the largest, ties by ``ranks``
        (:func:`ordered`)."""
        return ordered(keys, self.ranks)

    def by_strength(self, xs: Sequence[int], strengths: np.ndarray) -> list[int]:
        """``xs`` from the strongest dependence to the weakest, by the strengths :meth:`scan` gave
        them, ties by ``ranks`` (:meth:`ordered`).

        A partial correlation is compared to ``KEY_DP`` decimals, as :meth:`strongest` keys it. A
        p-value, the strength of a test without one, is compared to ``KEY_DP`` significant digits,
        so p-values far below ``10**-KEY_DP``, common for strong dependence on many rows, keep the
        order of their sizes rather than tying.
        """
        values = np.asarray(strengths, dtype=np.float64).tolist()
        if self._pvalue_strengths:
            keys = {x: float(f"{-s:.{KEY_DP}g}") for x, s in zip(xs, values, strict=True)}
        else:
            keys = {x: -round(s, KEY_DP) for x, s in zip(xs, values, strict=True)}
        return self.ordered(keys)

    def _asked(
        self,
        start: float,
        target: int,
        xs: Sequence[int],
        n_sets: int,
        conds: Callable[[], Iterable[Sequence[int]]],
    ) -> None:
        """Add a query call that began at ``start`` and asked each of ``xs`` given each of the
        ``n_sets`` sets ``conds()`` gives to :attr:`counts`; ``conds`` is called only to keep the
        distinct queries."""
        counts = self.counts
        counts.test_seconds += perf_counter() - start
        counts.issued += len(xs) * n_sets
        if counts.queries is not None:
            sets = [tuple(sorted(int(c) for c in cond)) for cond in conds()]
            counts.queries.update(
                (min(int(target), int(x)), max(int(target), int(x)), s) for x in xs for s in sets
            )

    @contextmanager
    def _measured(self) -> Generator[None]:
        """Add the tests computed and the time spent in the block to :attr:`counts`, unless an
        enclosing block already measures them."""
        if self._measuring:
            yield
            return
        self._measuring = True
        start = perf_counter()
        try:
            with counting_tests() as tests:
                yield
            self.counts.computed += tests.n
        finally:
            self._measuring = False
            self.counts.search_seconds += perf_counter() - start

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
            with self._measured():
                members = search(self, target, sepsets)
            self._runs[key] = (members, sepsets)
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
        variable no test separated has no entry, as one a ``"pc"`` learner never tests because
        the knowledge makes it non-adjacent to ``target``.
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
            self._spouses[method, target] = self.spouses(
                name, target, pc, sepsets, cell_rows=learner.cell_rows, df_rows=learner.df_rows
            )
        return pc, pc + self._spouses[method, target], sepsets

    def spouses(
        self,
        method: str,
        target: int,
        pc: list[int],
        sepsets: Sepsets,
        *,
        cell_rows: int | None = None,
        df_rows: int | None = None,
    ) -> list[int]:
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
        variables is not run, and with ``cell_rows`` or ``df_rows``, a test of categorical columns
        whose table (:func:`table_size`) has more cells than ``n / cell_rows`` or more degrees of
        freedom than ``n / df_rows`` (:meth:`reliable`); a test not run admits nothing, so the
        candidate joins only through another ``y``. A candidate without a separating set,
        one the knowledge makes non-adjacent to ``target``, never joins: such a pair is left out
        of every returned set.

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
        with self._measured():
            for y in pc:
                for x in self.parents_children(method, y):
                    if x == target or x in pc or x in found or x not in sepsets or y in sepsets[x]:
                        continue
                    cond = tuple(sorted((*sepsets[x], y)))
                    if self.max_depth is not None and len(cond) > self.max_depth:
                        continue
                    if not self.reliable(target, x, cond, cell_rows=cell_rows, df_rows=df_rows):
                        continue
                    if self.scan(target, [x], cond)[1][0] <= self.alpha:
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
    trace: bool = False,
    ranked: bool = True,
) -> LocalSearch:
    """A :class:`LocalSearch` over the ``indep_test`` test of ``X``.

    ``categorical`` marks the categorical columns for a mixed-data test; ``seed`` fixes the draws
    of a stochastic test (``None`` uses ``ANDREY_SEED``); ``alpha``, ``max_depth``, and
    ``constraints`` are as :class:`LocalSearch` takes them, and ``trace`` keeps the distinct queries
    in its ``counts``. A column the test reads as categories (:func:`~andrey.core.ties.tie_mask`)
    has as many levels as distinct observed values. With ``ranked=False``, the ranks are the column
    positions and ``andrey.config.tie_break`` is not read: for a caller that asks the tests only
    through :meth:`~LocalSearch.scan` and never picks between tied candidates.
    """
    if ranked:
        test, ranks = _ranked_test(X, indep_test, categorical, seed)
    else:
        test = make_indep_test(indep_test, X, categorical=categorical, seed=seed)
        ranks = list(range(X.shape[1]))
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
        trace=trace,
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
