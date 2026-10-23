"""RFCI (Really Fast Causal Inference) causal discovery, the engine behind :func:`andrey.rfci`.

The algorithm, its settings, and its references are described on :func:`andrey.rfci`. RFCI shares
FCI's adjacency search, collider step, and orientation rules (:mod:`andrey.constraint.fci`), skips
the possible-d-separation step, and adds the tests of Colombo et al. (2012), Algorithms 4.4 and 4.5
of their supplement: on the unshielded triples before they are oriented, and on each discriminating
path before rule R4 orients it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from andrey.constraint._skeleton import unseparated_triples
from andrey.constraint.fci import (
    _arrow_allowed,
    _circle_pag,
    _discriminating_paths,
    _fast_adjacency_search,
    _Implied,
    _into,
    _is_parent_of,
    _knowledge_marks,
    _neighbor_lists,
    _orient_colliders,
    _rule_r1r2_cycle,
    _rule_r3,
    _rule_r5,
    _rule_r6,
    _rule_r7,
    _rule_r8,
    _rule_r9,
    _rule_r10,
    _sepset,
)
from andrey.constraint.pc import check_knowledge_cycle
from andrey.core import ARROW, CIRCLE, NULL, TAIL, GraphStructure, backend
from andrey.core.ci import CITest, make_indep_test
from andrey.search._parallel import _KEY_DP
from andrey.search.hc import tie_ranks

if TYPE_CHECKING:
    from andrey.core.knowledge import Constraints

Sepsets = dict[tuple[int, int], tuple[int, ...]]
# One way to separate a pair: ``(sort key, separating set)``; the smallest key wins.
_Option = tuple[tuple, tuple[int, ...]]


def rfci(
    data: np.ndarray,
    *,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    categorical: np.ndarray | None = None,
    max_depth: int | None = None,
    constraints: Constraints | None = None,
    workers: int | None = None,
) -> GraphStructure:
    """Discover an RFCI-PAG from ``data`` with the RFCI algorithm and the ``indep_test`` CI test.

    ``data`` is an ``(n_samples, n_features)`` array; ``alpha`` is the independence-test
    significance level in ``(0, 1)``, and a test separates a pair when its p-value is above it.
    ``categorical``, a boolean mask of the categorical columns, reaches a mixed-data test.
    ``max_depth`` caps the conditioning sets of the adjacency search; every later test conditions on
    a subset of a recorded separating set, which the cap already bounds. ``constraints`` act as in
    :func:`~andrey.constraint.fci.fci`: no step tests or removes a required pair, and their
    endpoints are set before the orientation rules. ``workers`` worker processes share the
    adjacency search's larger conditioning-size passes; ``None`` reads
    ``andrey.config.num_workers``.

    The steps: the adjacency search, a minimal separating set for each pair it removes
    (:meth:`_Tests.minimal`), the unshielded-triple step (:func:`_check_triples`), the colliders,
    and the orientation rules with RFCI's rule R4 (:func:`_rfci_orient`).
    Returns the PAG as a ``GraphStructure`` of kind ``"pag"``.
    """
    X = np.asarray(data, dtype=np.float64)
    test = make_indep_test(indep_test, X, categorical=categorical)
    allowed = keep = apart = None
    if constraints is not None:
        allowed = constraints.support()
        keep = constraints.required | constraints.required.T
        apart = ~allowed
    support, unions = _fast_adjacency_search(
        X,
        test,
        alpha,
        workers=backend.worker_count(workers),
        support=allowed,
        keep=keep,
        apart=apart,
        max_depth=max_depth,
    )
    tests = _Tests(test, alpha, tie_ranks(X, categorical), keep)
    # Only a pair with a common neighbor forms an unshielded triple or ends a discriminating path,
    # and removals never add one, so the steps after the search read only those pairs' sets.
    shared = support.astype(np.float32) @ support.astype(np.float32) > 0
    sepsets: Sepsets = {}
    for (x, y), union in unions.items():
        if x < y:
            found = tests.minimal(x, y, union) if shared[x, y] else union
            sepsets[(x, y)] = sepsets[(y, x)] = found

    marks = _circle_pag(support)
    ambiguous = unseparated_triples(support, sepsets, constraints)
    _check_triples(marks, tests, sepsets, ambiguous)
    _knowledge_marks(marks, constraints)
    if constraints is not None:
        check_knowledge_cycle((marks == TAIL) & (marks.T == ARROW), closing=(marks == ARROW).T)
    _orient_colliders(marks, sepsets, ambiguous)
    _rfci_orient(marks, tests, sepsets, ambiguous)
    return GraphStructure.from_numpy(marks, kind="pag")


class _Tests:
    """The CI tests RFCI runs after the adjacency search, each answered once.

    ``ranks`` are the columns' :func:`~andrey.search.hc.tie_ranks`, which order separating sets
    whose p-values agree to ``_KEY_DP`` decimals. ``keep``, when given, marks the required pairs,
    which are never tested.
    """

    def __init__(
        self, test: CITest, alpha: float, ranks: np.ndarray, keep: np.ndarray | None
    ) -> None:
        self.test = test
        self.alpha = alpha
        self.ranks = ranks.tolist()
        self.keep = keep
        self.answers: dict[tuple[int, int, tuple[int, ...]], float] = {}

    def pvalues(self, x: int, y: int, subsets: list[tuple[int, ...]]) -> list[float]:
        """The p-values of ``x`` and ``y`` given each of ``subsets`` (sorted members)."""
        x, y = min(x, y), max(x, y)
        missing = [s for s in subsets if (x, y, s) not in self.answers]
        if missing:
            batched = getattr(self.test, "batched_call", None)
            found = (
                batched(x, y, missing)
                if callable(batched)
                else [self.test(x, y, s) for s in missing]
            )
            for s, p in zip(missing, found):
                self.answers[(x, y, s)] = float(p)
        return [self.answers[(x, y, s)] for s in subsets]

    def option(self, subset: tuple[int, ...], p: float) -> _Option:
        """The sort key of separating set ``subset`` with p-value ``p``: smaller sets first, then
        larger p-values, then lower ranks."""
        ranked = tuple(sorted(self.ranks[v] for v in subset))
        return (len(subset), -round(p, _KEY_DP), ranked), subset

    def separate(self, x: int, y: int, candidates: Iterable[int]) -> _Option | None:
        """The minimal set that separates ``x`` and ``y`` among the subsets of ``candidates``.

        Tests the subsets by increasing size from one member, as the adjacency search already
        tested the empty set on every pair it kept; the first size with a separating subset decides,
        so no proper subset of the set returned separates the pair. Of that size's separating
        subsets, the one with the largest p-value wins, then the one with the lowest ranks. Returns
        ``None`` when no subset separates the pair or the pair is required.
        """
        if self.keep is not None and self.keep[x, y]:
            return None
        pool = sorted(candidates)
        for size in range(1, len(pool) + 1):
            subsets = list(combinations(pool, size))
            options = [
                self.option(s, p)
                for s, p in zip(subsets, self.pvalues(x, y, subsets))
                if p > self.alpha
            ]
            if options:
                return min(options)
        return None

    def minimal(self, x: int, y: int, union: tuple[int, ...]) -> tuple[int, ...]:
        """A minimal separating set for a pair the adjacency search removed with ``union``.

        ``union`` joins every conditioning set of one size that separated the pair, so the
        search for a minimal set runs over its subsets (:meth:`separate`). A union of at most one
        member is minimal as it is.
        """
        if len(union) <= 1:
            return union
        found = self.separate(x, y, union)
        return union if found is None else found[1]


def _remove(
    marks: np.ndarray, sepsets: Sepsets, removals: Mapping[tuple[int, int], list[_Option]]
) -> None:
    """Remove each pair of ``removals`` and record its separating set: of the ways each was found
    separated, the one with the smallest sort key (:meth:`_Tests.option`)."""
    for (x, y), options in removals.items():
        marks[x, y] = marks[y, x] = NULL
        sepsets[(x, y)] = sepsets[(y, x)] = min(options)[1]


def _found(removals: dict[tuple[int, int], list[_Option]], x: int, y: int, option: _Option) -> None:
    removals.setdefault((min(x, y), max(x, y)), []).append(option)


def _check_triples(
    marks: np.ndarray,
    tests: _Tests,
    sepsets: Sepsets,
    ambiguous: frozenset[tuple[int, int, int]],
) -> None:
    """RFCI's unshielded-triple step (Algorithm 4.4 of Colombo et al. 2012), in place.

    For each unshielded triple ``a *-* b *-* c`` with a separating set ``S`` for ``a`` and ``c``,
    ``a`` and ``b``, and ``b`` and ``c``, are tested given ``S`` without ``b``. A pair found
    independent loses its edge and takes a minimal separating set from the subsets of that set
    (:meth:`_Tests.separate`). A removal can create new unshielded triples, so the step repeats
    until no pair is removed; then every unshielded triple left has passed both tests, and the
    caller orients the colliders among them.

    The step runs in rounds, as the adjacency search runs its sizes: each round tests every
    unshielded triple of the graph as the round starts and removes the pairs found independent
    together. A pair several triples separate takes the smallest set, then the one with the largest
    p-value, then the one with the lowest ranks. No round reads a node's position, so the result
    does not depend on the column order.
    """
    while True:
        removals: dict[tuple[int, int], list[_Option]] = {}
        for b, adjacent in enumerate(_neighbor_lists(marks)):
            for a, c in combinations(adjacent, 2):
                if marks[a, c] != NULL or (a, b, c) in ambiguous:
                    continue
                sep = _sepset(sepsets, a, c)
                if sep is None or not sep - {b}:
                    continue  # the adjacency search tested every kept pair given no set
                condition = tuple(sorted(sep - {b}))
                for end in (a, c):
                    if tests.keep is not None and tests.keep[end, b]:
                        continue
                    if tests.pvalues(end, b, [condition])[0] > tests.alpha:
                        option = tests.separate(end, b, condition)
                        if option is not None:
                            _found(removals, end, b, option)
        if not removals:
            return
        _remove(marks, sepsets, removals)


def _rfci_orient(
    marks: np.ndarray,
    tests: _Tests,
    sepsets: Sepsets,
    ambiguous: frozenset[tuple[int, int, int]],
) -> None:
    """Apply rules R1-R10, with RFCI's rule R4 (Algorithm 4.5 of Colombo et al. 2012), in place.

    The rules run in order-free rounds, as :func:`~andrey.constraint.fci._fci_orient` runs them: in
    a round, every rule reads the marks as the round starts and records the marks it implies, and a
    circle takes a mark only when the rules imply that one alone. Rule R4 first tests the pairs
    along each shortest discriminating path (:func:`_rule_r4`). When those tests separate a pair,
    the round sets no mark: the pairs found are removed, the unshielded-triple step runs on the new
    graph (:func:`_check_triples`), the new colliders are oriented, and the next round starts.
    """
    while True:
        implied = _Implied()
        _rule_r1r2_cycle(marks, False, ambiguous, out=implied)
        _rule_r3(marks, sepsets, False, ambiguous, out=implied)
        removals = _rule_r4(marks, tests, sepsets, implied)
        if removals:
            _remove(marks, sepsets, removals)
            _check_triples(marks, tests, sepsets, ambiguous)
            _orient_colliders(marks, sepsets, ambiguous)
            continue
        _rule_r5(marks, False, ambiguous, out=implied)
        _rule_r6(marks, False, out=implied)
        _rule_r7(marks, False, ambiguous, out=implied)
        _rule_r8(marks, False, out=implied)
        _rule_r9(marks, False, ambiguous, out=implied)
        _rule_r10(marks, False, ambiguous, out=implied)
        if not implied.apply(marks):
            return


def _rule_r4(
    marks: np.ndarray, tests: _Tests, sepsets: Sepsets, out: _Implied
) -> dict[tuple[int, int], list[_Option]]:
    """RFCI's R4: test each shortest discriminating path, then orient the endpoint it decides.

    For each triangle of ``b *-> a``, ``a -> c``, and ``b o-> c``, the shortest
    discriminating paths ``d, ..., a, b, c`` for ``b`` come from
    :func:`~andrey.constraint.fci._discriminating_paths`. On each path whose ends ``d`` and ``c``
    have a separating set ``S``, every two consecutive nodes are tested given each subset of ``S``
    without them (:meth:`_Tests.separate`). Returns the pairs those tests separate. A path whose
    pairs all stay dependent records its marks in ``out``: ``b -> c`` when ``b`` is in ``S``,
    ``a <-> b <-> c`` otherwise. Paths that decide differently leave the endpoint at ``b`` a circle.
    """
    removals: dict[tuple[int, int], list[_Option]] = {}
    n = marks.shape[0]
    for b in range(n):
        poss_a = [a for a in range(n) if marks[a, b] == ARROW]  # b *-> a
        for c in _into(marks, b, CIRCLE):  # c *-o b
            if marks[c, b] != ARROW:
                continue
            for a in poss_a:
                if not _is_parent_of(marks, a, c):
                    continue
                for paths in _discriminating_paths(marks, a, b, c):
                    decided = False
                    for path in paths:
                        sep = _sepset(sepsets, path[0], c)
                        if sep is None:
                            continue
                        decided = True
                        if _path_removals(tests, [*path, c], sep, removals):
                            continue
                        if b in sep:
                            out[b, c] = TAIL
                        elif _arrow_allowed(marks, a, b):
                            out[b, a] = ARROW
                            out[b, c] = ARROW
                    if decided:
                        break
    return removals


def _path_removals(
    tests: _Tests,
    path: list[int],
    sep: frozenset[int],
    removals: dict[tuple[int, int], list[_Option]],
) -> bool:
    """Test every two consecutive nodes of ``path`` given the subsets of ``sep`` without them.

    Adds each pair a subset separates to ``removals`` with its minimal set, and returns whether any
    was found.
    """
    found = False
    for r, q in zip(path, path[1:]):
        option = tests.separate(r, q, sep - {r, q})
        if option is not None:
            _found(removals, r, q, option)
            found = True
    return found
