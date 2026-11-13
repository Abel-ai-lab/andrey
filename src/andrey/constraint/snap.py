"""SNAP, the engine behind :func:`andrey.snap`.

Sequential non-ancestor pruning (Schubert, Claassen, and Magliacane 2025) learns the CPDAG over the
possible ancestors of a set of targets only. It runs PC's skeleton search one conditioning size
(order) at a time; after each order it orients the v-structures the separating sets found so far
imply and drops every variable with no possibly directed path to a target, so the next order tests
only the variables left (Algorithm 1). Run to the end, it applies Meek's rules and prunes once more
(Algorithm 2).

Two v-structures that disagree on an edge give it arrowheads at both ends. Before the end that
marks neither end as an ancestor of the other, as the paper reads it. At the end, where exact tests
leave no such edge (Theorem 3.2), the v-structures are oriented again as :func:`~andrey.pc` orients
them: the one whose pair the tests separated more clearly is kept.

The search visits the variables, their neighbors, and the conditioning sets in the order of
``search.ranks`` (:func:`~andrey.core.ties.tie_ranks`), so reordering the columns relabels the
result and changes nothing else. Each query ``(a, b | S)`` is asked of the test once, the
lower-ranked variable first, and its p-value kept. A p-value above ``alpha`` is an independence.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from andrey.constraint.blanket import LocalSearch
from andrey.constraint.pc import orient_colliders
from andrey.core.adjustment import possible_ancestors
from andrey.core.orient import meek
from andrey.core.structure import ARROW, NULL, TAIL
from andrey.core.ties import resolve_collider_order

__all__ = ["SNAPResult", "snap"]

Pair = tuple[int, int]


@dataclass(frozen=True)
class SNAPResult:
    """SNAP's answer, by 0-based column position.

    ``marks`` holds the endpoint marks over every column (``marks[a, b]`` the mark at ``a``); a
    column outside ``kept`` has no edge. An edge with arrowheads at both ends marks two v-structures
    that disagree, so neither end is an ancestor of the other; a complete search leaves one only
    where their separating p-values tie. ``kept`` lists the possible ancestors of the targets the
    search kept, the targets included, ascending. ``complete`` is whether the search ran out of
    pairs to test before the cap on the order, so that ``marks`` is SNAP(infinity)'s CPDAG over
    ``kept``; otherwise ``marks`` is SNAP(k)'s graph, the skeleton and v-structures found up to the
    cap, and ``kept`` its pruning. ``order`` is the last order the skeleton search tested.
    ``sepsets`` maps each pair ``(a, b)``, ``a < b``, of kept columns that the search made
    non-adjacent to the separating set recorded for it.
    """

    marks: np.ndarray
    kept: list[int]
    complete: bool
    order: int
    sepsets: dict[Pair, tuple[int, ...]]


class _Tests:
    """The p-values of the queries asked of ``search``, each asked once, the lower-ranked variable
    first."""

    def __init__(self, search: LocalSearch) -> None:
        self.search = search
        self.alpha = search.alpha
        self.ranks = search.ranks
        self.kept: dict[tuple[int, int, tuple[int, ...]], float] = {}

    @staticmethod
    def key(a: int, b: int, cond: Iterable[int]) -> tuple[int, int, tuple[int, ...]]:
        return (min(a, b), max(a, b), tuple(sorted(cond)))

    def ask(self, a: int, bs: Sequence[int], cond: Sequence[int] = ()) -> None:
        """Keep the p-value of ``a`` against each of ``bs`` given ``cond``, asking the test
        together for those not kept yet; ``a`` ranks below every one of ``bs``."""
        new = [b for b in bs if self.key(a, b, cond) not in self.kept]
        if new:
            _, pvalues = self.search.scan(a, new, tuple(cond))
            for b, p in zip(new, pvalues, strict=True):
                self.kept[self.key(a, b, cond)] = float(p)

    def independent(self, a: int, b: int, cond: Sequence[int] = ()) -> bool:
        """Whether ``a`` and ``b`` test independent given ``cond``."""
        key = self.key(a, b, cond)
        if key not in self.kept:
            low, high = (a, b) if self.ranks[a] < self.ranks[b] else (b, a)
            self.ask(low, [high], cond)
        return self.kept[key] > self.alpha


def _by_rank(nodes: Iterable[int], ranks: Sequence[int]) -> list[int]:
    return sorted(nodes, key=lambda v: ranks[v])


def _separating(sepsets: dict[Pair, tuple[int, ...]], a: int, b: int) -> tuple[int, ...]:
    return sepsets[(min(a, b), max(a, b))]


def skeleton_step(
    tests: _Tests,
    adjacent: np.ndarray,
    nodes: Sequence[int],
    order: int,
    sepsets: dict[Pair, tuple[int, ...]],
) -> None:
    """PC's skeleton search at one ``order`` (Algorithm 1, lines 4-9), in place.

    For each node ``x`` of ``nodes`` (in rank order) and each neighbor ``y``, the subsets of
    ``order`` other neighbors of ``x``, as they stand when ``y``'s turn comes, are tested in rank
    order; the first that separates the pair removes the edge and becomes its separating set.
    """
    ranks = tests.ranks
    if order == 0:  # the tests read no neighbor, so asking them together changes nothing
        for x in nodes:
            later = [y for y in np.flatnonzero(adjacent[x]).tolist() if ranks[y] > ranks[x]]
            tests.ask(x, _by_rank(later, ranks))
    for x in nodes:
        if np.count_nonzero(adjacent[x]) - 1 < order:
            continue
        for y in _by_rank(np.flatnonzero(adjacent[x]).tolist(), ranks):
            if not adjacent[x, y]:
                continue
            others = _by_rank([v for v in np.flatnonzero(adjacent[x]).tolist() if v != y], ranks)
            for cond in combinations(others, order):
                if tests.independent(x, y, cond):
                    adjacent[x, y] = adjacent[y, x] = False
                    sepsets[(min(x, y), max(x, y))] = tuple(sorted(cond))
                    break


def unshielded_triples(adjacent: np.ndarray, ranks: Sequence[int]) -> list[tuple[int, int, int]]:
    """Every unshielded triple ``(a, z, b)``, ``a - z - b`` with ``a`` and ``b`` not adjacent,
    ``a`` ranked below ``b``, ordered by the ranks of ``a``, ``z``, then ``b``."""
    found = []
    for z in range(len(adjacent)):
        nbrs = np.flatnonzero(adjacent[z]).tolist()
        for a, b in combinations(nbrs, 2):
            if not adjacent[a, b]:
                a, b = (a, b) if ranks[a] < ranks[b] else (b, a)
                found.append((a, z, b))
    return sorted(found, key=lambda t: (ranks[t[0]], ranks[t[1]], ranks[t[2]]))


def _marks(adjacent: np.ndarray, colliders: Iterable[tuple[int, int, int]]) -> np.ndarray:
    """The skeleton's marks with an arrowhead at the middle of each collider ``(a, z, b)``, every
    other mark a tail: two colliders that disagree on an edge make it bidirected."""
    marks = np.where(adjacent, np.int8(TAIL), np.int8(NULL))
    for a, z, b in colliders:
        marks[z, a] = marks[z, b] = ARROW
    return marks


def orient_pc(
    adjacent: np.ndarray, sepsets: dict[Pair, tuple[int, ...]], ranks: Sequence[int]
) -> np.ndarray:
    """Algorithm 3: an arrowhead at ``z`` on both edges of each unshielded triple ``a - z - b``
    whose separating set lacks ``z``."""
    triples = unshielded_triples(adjacent, ranks)
    return _marks(adjacent, [t for t in triples if t[1] not in _separating(sepsets, t[0], t[2])])


def _minimal(tests: _Tests, v: int, z: int, start: tuple[int, ...]) -> tuple[int, ...]:
    """Algorithm 4, lines 11-19: drop members of ``start``, one at a time in rank order, while the
    rest still separates ``v`` and ``z``; a separating set no member of which can be dropped."""
    found = list(_by_rank(start, tests.ranks))
    while True:
        for s in found:
            rest = [u for u in found if u != s]
            if tests.independent(v, z, rest):
                found = rest
                break
        else:
            return tuple(sorted(found))


def orient_rfci(
    tests: _Tests, adjacent: np.ndarray, sepsets: dict[Pair, tuple[int, ...]]
) -> np.ndarray:
    """Algorithm 4 (RFCI's v-structure rule, Colombo et al. 2012, Algorithm 4.4), in place on
    ``adjacent`` and ``sepsets``; returns the marks.

    An unshielded triple ``a - z - b`` whose separating set ``S`` lacks ``z`` is a v-structure
    only when ``a`` and ``z``, and ``z`` and ``b``, both test dependent given ``S``. When one of
    those pairs tests independent given ``S`` instead, its edge is removed, with a minimal
    separating set within ``S``; the triangles over that pair become unshielded triples to check,
    and the triples and v-structures that used the edge are dropped. Triples are checked first in,
    first out.
    """
    ranks = tests.ranks
    queue = deque(unshielded_triples(adjacent, ranks))
    waiting = set(queue)
    legit: list[tuple[int, int, int]] = []

    def uses(triple: tuple[int, int, int], v: int, z: int) -> bool:
        a, m, b = triple
        return {a, m} == {v, z} or {m, b} == {v, z}

    while queue:
        triple = queue.popleft()
        waiting.discard(triple)
        a, z, b = triple
        cond = _separating(sepsets, a, b)
        if z in cond:
            continue
        free = [v for v in (a, b) if tests.independent(v, z, cond)]
        if not free:
            legit.append(triple)
            continue
        for v in free:
            if not adjacent[v, z]:
                continue
            sepsets[(min(v, z), max(v, z))] = _minimal(tests, v, z, cond)
            low, high = (v, z) if ranks[v] < ranks[z] else (z, v)
            for w in _by_rank(np.flatnonzero(adjacent[v] & adjacent[z]).tolist(), ranks):
                if (low, w, high) not in waiting:
                    queue.append((low, w, high))
                    waiting.add((low, w, high))
            for t in [t for t in queue if uses(t, v, z)]:
                queue.remove(t)
                waiting.discard(t)
            legit = [t for t in legit if not uses(t, v, z)]
            adjacent[v, z] = adjacent[z, v] = False
    return _marks(adjacent, legit)


def _testable(adjacent: np.ndarray, order: int) -> bool:
    """Whether some edge has ``order`` other neighbors at one end to condition on."""
    return bool(adjacent.any()) and int(adjacent.sum(axis=1).max()) - 1 >= order


def snap(
    search: LocalSearch, targets: Sequence[int], *, max_depth: int | None = None
) -> SNAPResult:
    """SNAP for the ``targets`` (positions) over every column of ``search``'s test.

    Each order ``i`` runs the skeleton step at that conditioning size, orients the v-structures by
    PC's rule below order 2 and by RFCI's from order 2 on, and keeps the variables with a possibly
    directed path to a target (Algorithm 1). An order with no pair left to test changes nothing
    once its rule is the previous order's, so the search ends there; that leaves an order 2 with
    no test to run RFCI's rule once, as Algorithm 2 runs every order to ``|V| - 2``. With
    ``max_depth``, an order above it ends the search when pairs are left to test (SNAP(k)).
    Otherwise the v-structures of the final skeleton and separating sets are oriented in the order
    of :func:`~andrey.constraint.pc.orient_colliders` (by default the most clearly separated pair
    first, a collider that would make an edge bidirected skipped), Meek's rules R1-R3 run, and the
    variables are pruned once more (Algorithm 2). Under exact tests the v-structures are those of
    the last order, and the graph is Algorithm 2's. Every test conditions on at most
    ``max_depth`` variables: the RFCI rule's tests condition on separating sets and their
    subsets.
    """
    d = search.d
    ranks = search.ranks
    tests = _Tests(search)
    adjacent = ~np.eye(d, dtype=bool)
    sepsets: dict[Pair, tuple[int, ...]] = {}
    kept = set(range(d))
    marks = np.where(adjacent, np.int8(TAIL), np.int8(NULL))
    order, last_tested, complete = 0, 0, True
    while True:
        can_test = _testable(adjacent, order)
        if can_test and max_depth is not None and order > max_depth:
            complete = False
            break
        if not can_test and order > 0 and order != 2:
            break
        if can_test:
            nodes = _by_rank(kept, ranks)
            skeleton_step(tests, adjacent, nodes, order, sepsets)
            last_tested = order
        if order < 2:
            marks = orient_pc(adjacent, sepsets, ranks)
        else:
            marks = orient_rfci(tests, adjacent, sepsets)
        kept = possible_ancestors(marks, targets)
        dropped = [v for v in range(d) if v not in kept]
        adjacent[dropped, :] = adjacent[:, dropped] = False
        marks[dropped, :] = marks[:, dropped] = NULL
        order += 1
    if complete:
        inside = {p: s for p, s in sepsets.items() if p[0] in kept and p[1] in kept}
        pvalues = {(a, b): tests.kept[tests.key(a, b, s)] for (a, b), s in inside.items()}
        skeleton = np.where(adjacent, np.int8(TAIL), np.int8(NULL))
        order_rule = resolve_collider_order(None)
        marks = meek(orient_colliders(skeleton, inside, pvalues, order=order_rule))
        kept = possible_ancestors(marks, targets)
        dropped = [v for v in range(d) if v not in kept]
        marks[dropped, :] = marks[:, dropped] = NULL
    inside = {p: s for p, s in sepsets.items() if p[0] in kept and p[1] in kept}
    return SNAPResult(marks, sorted(kept), complete, last_tested, inside)
