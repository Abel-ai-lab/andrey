"""LocPC and LocPC-CDE: PC-style local discovery around an outcome [1]_.

:func:`locpc` learns the local essential graph of a target at hop ``h``: the edges at every
variable within ``h`` edges of the target, oriented where the tests run there decide. Its tests
join each variable it visits to every variable not yet separated from it and remove an edge at the
first conditioning set, among the variable's current neighbors, that separates the pair, as PC
does. :func:`locpc_cde` grows the hop until it can tell whether the controlled direct effect of a
treatment on an outcome is identifiable: when every edge at the outcome is oriented, its parents
are known.

References
----------
.. [1] Loranchet, T., and Assaad, C. K. (2026). Local Markov equivalence for PC-style local causal
   discovery and identification of controlled direct effects. Proceedings of the Fifth Conference
   on Causal Learning and Reasoning, PMLR 323, 84-118.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from andrey.constraint.pc import knowledge_masks, orient_colliders, undirect_conflicts
from andrey.core import ARROW, NULL, TAIL
from andrey.core.knowledge import orient as orient_knowledge
from andrey.core.orient import meek

if TYPE_CHECKING:
    from andrey.constraint.blanket import LocalSearch

Pair = tuple[int, int]

#: Why :func:`locpc_cde` stopped, in the order its final check reads them.
REASONS: tuple[str, ...] = ("not adjacent", "treatment is a child", "oriented", "not orientable")


def _pair(a: int, b: int) -> Pair:
    return (a, b) if a < b else (b, a)


@dataclass
class Memory:
    """What the LocPC runs of one fit share.

    ``sepsets`` maps a pair, lower position first, to the set that separated it, ascending;
    ``separated`` to the p-value of that test; ``pvalues`` maps ``(lower, higher, set)`` to every
    p-value computed, so a query asked again is not tested again.
    """

    sepsets: dict[Pair, tuple[int, ...]] = field(default_factory=dict)
    separated: dict[Pair, float] = field(default_factory=dict)
    pvalues: dict[tuple[int, int, tuple[int, ...]], float] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalEssentialGraph:
    """A local essential graph over all ``d`` variables.

    ``marks`` is a ``d x d`` endpoint-mark matrix (``marks[i, j]`` the mark at ``i`` on its edge
    with ``j``); every edge has at least one end in ``visited``. ``neighborhood`` holds the
    variables within ``hop`` edges of the target, ascending; an edge between two of them is a true
    adjacency, and an edge from one of them to a variable outside may join a descendant that no
    set of the inside variable's neighbors separates. ``double_bar`` holds the pairs ``(inside,
    outside)`` of the undirected edges the no-non-collider rule marks: no outside variable needs
    the outside end to separate it from the inside end. ``visited`` lists the variables whose tests
    ran, in order.
    """

    marks: np.ndarray
    double_bar: frozenset[Pair]
    neighborhood: tuple[int, ...]
    visited: tuple[int, ...]
    hop: int

    def neighbors(self, v: int) -> list[int]:
        """The variables adjacent to ``v``, ascending."""
        return np.flatnonzero(self.marks[v] != NULL).tolist()

    def non_arrow(self, v: int) -> list[int]:
        """The neighbors of ``v`` joined by an edge without an arrowhead: undirected or double-bar
        (the paper's ``Ne_nar``)."""
        return [u for u in self.neighbors(v) if self.marks[v, u] == TAIL == self.marks[u, v]]

    def parents(self, v: int) -> list[int]:
        """The variables with an edge into ``v``, ascending."""
        return [u for u in self.neighbors(v) if self.marks[v, u] == ARROW != self.marks[u, v]]


def _skeleton(
    search: LocalSearch, target: int, hop: int, memory: Memory
) -> tuple[list[set[int]], list[int]]:
    """The adjacencies of LocPC's skeleton phase (Algorithm 1, lines 1-19) and the visited
    variables, in order. ``memory`` holds the separating sets of earlier runs, which keep their
    pairs apart, and receives the new ones."""
    d, alpha, max_depth = search.d, search.alpha, search.max_depth
    ranks = search.ranks
    support, keep, _ = knowledge_masks(search.constraints)
    before = None
    if search.constraints is not None:
        before = search.constraints.precedence()

    def ordered(nodes) -> list[int]:
        return sorted(nodes, key=lambda v: (ranks[v], v))

    def pvalue(a: int, b: int, cond: tuple[int, ...]) -> float:
        lo, hi = _pair(a, b)
        key = (lo, hi, tuple(sorted(cond)))
        p = memory.pvalues.get(key)
        if p is None:
            p = float(search.scan(a, [b], list(cond))[1][0])
            memory.pvalues[key] = p
        return p

    adjacent: list[set[int]] = [set() for _ in range(d)]
    frontier, visited, seen = [target], [target], {target}
    for _ in range(hop + 1):
        for a in frontier:
            for b in range(d):
                if b != a and _pair(a, b) not in memory.sepsets:
                    if support is None or support[a, b]:
                        adjacent[a].add(b)
                        adjacent[b].add(a)
        size = 0
        while max_depth is None or size <= max_depth:
            stored = {a: ordered(adjacent[a]) for a in frontier}
            if not any(len(stored[a]) - 1 >= size for a in frontier):
                break
            for a in frontier:
                if a not in seen:
                    seen.add(a)
                    visited.append(a)
                for b in ordered(adjacent[a]):
                    if keep is not None and keep[a, b]:
                        continue
                    if before is not None and b in seen and before[a, b]:
                        continue
                    others = [v for v in stored[a] if v != b]
                    for cond in combinations(others, size):
                        p = pvalue(a, b, cond)
                        if p > alpha:
                            pair = _pair(a, b)
                            memory.sepsets[pair] = tuple(sorted(cond))
                            memory.separated[pair] = p
                            adjacent[a].discard(b)
                            adjacent[b].discard(a)
                            break
            size += 1
        frontier = ordered({b for a in frontier for b in adjacent[a]})
    return adjacent, visited


def _within(adjacent: list[set[int]], target: int, hop: int) -> list[int]:
    """The variables a path of at most ``hop`` edges joins to ``target``, ascending."""
    reach, layer = {target}, {target}
    for _ in range(hop):
        layer = {b for a in layer for b in adjacent[a]} - reach
        reach |= layer
    return sorted(reach)


def locpc(
    search: LocalSearch, target: int, hop: int, memory: Memory | None = None
) -> LocalEssentialGraph:
    """Learn the local essential graph of ``target`` at ``hop`` with LocPC (Algorithm 1 of [1]_).

    Starting from ``target``, each round joins the current variables to every variable not yet
    separated from them and runs PC's levels over them: at size ``s``, each current variable is
    tested against each neighbor given the size-``s`` subsets of its neighbors at the start of
    the level, and the edge goes at the first set that separates the pair. The next round's
    variables are the neighbors of this round's, visited ones included, whose edges are tested
    again with their new neighbors. After ``hop + 1`` rounds, the unshielded colliders among the
    variables within ``hop`` edges of ``target`` are oriented, Meek's rules run among them, and
    the no-non-collider rule (Definition 9) marks an undirected edge from an inside variable ``D``
    to an outside variable ``A`` double-bar when no outside variable not adjacent to ``D`` has
    ``A`` in its separating set with ``D``.

    ``search`` answers the tests (its ``scan``) and fixes their order: variables, neighbors, and
    conditioning sets are visited by ``search.ranks``, so the graph does not depend on the column
    order. ``search.max_depth`` caps the size of every conditioning set. ``search.constraints``
    keeps a forbidden pair apart and a required one joined without a test, directs the edges they
    decide before the colliders (Meek's rule R4 joins R1-R3), and skips the retest of a visited
    variable's edge to a variable the knowledge places before it, which cannot be a spurious
    neighbor (Appendix C.3). ``memory`` carries the separating sets and p-values of earlier runs;
    a pair separated before is never joined again.

    Returns
    -------
    LocalEssentialGraph
        Double-bar edges are undirected in ``marks`` and listed in ``double_bar``.

    References
    ----------
    .. [1] Loranchet, T., and Assaad, C. K. (2026). Local Markov equivalence for PC-style local
       causal discovery and identification of controlled direct effects. Proceedings of the Fifth
       Conference on Causal Learning and Reasoning, PMLR 323, 84-118.
    """
    memory = Memory() if memory is None else memory
    adjacent, visited = _skeleton(search, target, hop, memory)
    d = search.d
    marks = np.zeros((d, d), dtype=np.int8)
    for a in range(d):
        for b in adjacent[a]:
            marks[a, b] = TAIL
    if search.constraints is not None:
        marks = orient_knowledge(marks, search.constraints)
    inside = _within(adjacent, target, hop)
    index = {v: k for k, v in enumerate(inside)}
    sub = marks[np.ix_(inside, inside)]
    sepsets, pvalues, unknown = {}, {}, set()
    for i, j in combinations(range(len(inside)), 2):
        pair = (inside[i], inside[j])
        if pair in memory.sepsets:
            sepsets[i, j] = tuple(index[v] for v in memory.sepsets[pair] if v in index)
            pvalues[i, j] = memory.separated[pair]
    for b in range(len(inside)):
        for a, c in combinations(np.flatnonzero(sub[b]).tolist(), 2):
            if sub[a, c] == NULL and (a, c) not in sepsets:
                unknown.add((a, b, c))
    sub = orient_colliders(sub, sepsets, pvalues, ambiguous=frozenset(unknown))
    sub = meek(sub, rule4=search.constraints is not None, ambiguous=frozenset(unknown))
    marks[np.ix_(inside, inside)] = undirect_conflicts(sub)
    double_bar = set()
    outside = set(range(d)) - set(inside)
    for a in inside:
        apart = [w for w in outside if marks[a, w] == NULL and _pair(a, w) in memory.sepsets]
        for b in sorted(outside & adjacent[a]):
            if marks[a, b] == TAIL == marks[b, a] and not any(
                b in memory.sepsets[_pair(a, w)] for w in apart
            ):
                double_bar.add((a, b))
    return LocalEssentialGraph(marks, frozenset(double_bar), tuple(inside), tuple(visited), hop)


@dataclass(frozen=True)
class CDEAnswer:
    """What :func:`locpc_cde` found.

    ``graph`` is the last local essential graph learned, at hop ``hop``; ``reason`` is one of
    :data:`REASONS`: the treatment is not adjacent to the outcome or is its child (the controlled
    direct effect is zero), every edge at the outcome is oriented (its parents are known), or
    some edge at the outcome stays unoriented once a larger hop adds no variable or every
    variable was visited.
    """

    graph: LocalEssentialGraph
    hop: int
    reason: str

    @property
    def identifiable(self) -> bool:
        """The paper's verdict: every reason but ``"not orientable"``."""
        return self.reason != "not orientable"


def locpc_cde(search: LocalSearch, treatment: int, outcome: int) -> CDEAnswer:
    """Decide whether the controlled direct effect of ``treatment`` on ``outcome`` is identifiable
    with LocPC-CDE (Algorithm 2 of [1]_, without its non-orientability stop).

    It learns the local essential graph of ``outcome`` at hop 0, 1, 2, ... (:func:`locpc`, the
    separating sets and p-values of each run kept for the next) while the treatment is adjacent
    to the outcome and not its child, some edge at the outcome is undirected or double-bar, and
    some variable is unvisited, and stops when the hop adds no variable to the neighborhood:
    every variable the outcome connects to is visited, and a larger hop reruns the same tests (a
    variable the tests separate from all of them is never visited, so the paper's other stop,
    every variable visited, need not come). Algorithm 2 also stops when a set of variables around
    the outcome meets the non-orientability criterion (Definition 12 of [1]_); this function
    leaves that stop out, as the strategy of Section 6 of [1]_ that precedes the criterion does,
    so a hop that would still direct an edge at the outcome is always learned. The effect is
    identifiable unless the treatment is adjacent to the outcome, not its child, and some edge
    at the outcome is unoriented.

    References
    ----------
    .. [1] Loranchet, T., and Assaad, C. K. (2026). Local Markov equivalence for PC-style local
       causal discovery and identification of controlled direct effects. Proceedings of the Fifth
       Conference on Causal Learning and Reasoning, PMLR 323, 84-118.
    """
    memory = Memory()
    x, y = treatment, outcome
    graph = locpc(search, y, 0, memory)

    def undecided(g: LocalEssentialGraph) -> bool:
        return g.marks[x, y] != NULL and y not in g.parents(x) and bool(g.non_arrow(y))

    hop = 1
    while undecided(graph) and len(graph.visited) < search.d:
        previous = graph.neighborhood
        graph = locpc(search, y, hop, memory)
        if graph.neighborhood == previous:
            break
        hop += 1
    if graph.marks[x, y] == NULL:
        reason = "not adjacent"
    elif y in graph.parents(x):
        reason = "treatment is a child"
    elif not graph.non_arrow(y):
        reason = "oriented"
    else:
        reason = "not orientable"
    return CDEAnswer(graph, graph.hop, reason)
