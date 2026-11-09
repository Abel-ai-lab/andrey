"""Claims that one variable is or is not an ancestor of another, applied to an equivalence class.

A claim that ``x`` is, or is not, an ancestor of ``y`` restricts the DAGs a CPDAG stands for.
Fang, Zhao, Liu, and He (2025) represent every such claim by direct causal clauses: a clause
``(t, heads)`` says that ``t`` is a direct cause of at least one vertex in ``heads``. By their
Theorem 17, ``x`` is an ancestor of ``y`` in a DAG of the class exactly when ``x`` is a direct
cause of a vertex of the critical set of ``x`` with respect to ``y``, and it is not one exactly
when every vertex of that set is a direct cause of ``x``. :func:`critical_set` finds the set
(Fang, Liu, Geng, and He 2022, Algorithm 2 and Proposition 4), :func:`consistent` checks a set of
clauses against the class (Fang et al. 2025, Algorithm 1), :func:`clause_mpdag` gives the MPDAG of
the DAGs that meet them (Algorithm 3 and Corollary 38), and :func:`residual_clauses` the clauses the
MPDAG cannot show (Algorithm 4). :func:`restrict` applies the claims to a learned graph.

References
----------
Fang, Z., Zhao, R., Liu, Y., and He, Y. (2025). On the representation of pairwise causal background
knowledge and its applications in causal inference. Journal of Machine Learning Research, 26(229).

Fang, Z., Liu, Y., Geng, Z., and He, Y. (2022). A local method for identifying causal relations
under Markov equivalence. Artificial Intelligence, 305, 103669.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .knowledge import KnowledgeError
from .structure import ARROW, TAIL

__all__ = [
    "Clause",
    "ChainGraph",
    "critical_set",
    "claim_clauses",
    "consistent",
    "clause_mpdag",
    "residual_clauses",
    "restrict",
]

#: A direct causal clause ``(tail, heads)``: ``tail`` is a direct cause of a vertex of ``heads``.
Clause = tuple[int, frozenset[int]]


@dataclass(frozen=True)
class ChainGraph:
    """A CPDAG's mark matrix with the sets the clause algorithms read.

    ``sib[v]`` holds the vertices joined to ``v`` by an undirected edge, ``children[v]`` and
    ``parents[v]`` the ends of its directed edges, and ``component[v]`` labels the chain component
    of ``v``: the vertices its undirected edges connect it to.
    """

    marks: np.ndarray
    sib: tuple[frozenset[int], ...]
    children: tuple[frozenset[int], ...]
    parents: tuple[frozenset[int], ...]
    component: tuple[int, ...]

    @classmethod
    def of(cls, cpdag: np.ndarray) -> ChainGraph:
        """The chain graph of the CPDAG mark matrix ``cpdag`` (``i -> j`` is ``[i, j] == TAIL``
        with ``[j, i] == ARROW``; an undirected edge has ``TAIL`` at both ends)."""
        tail = cpdag == TAIL
        lines = tail & tail.T
        directed = tail & (cpdag.T == ARROW)
        d = len(cpdag)
        sib = tuple(frozenset(np.flatnonzero(lines[v]).tolist()) for v in range(d))
        children = tuple(frozenset(np.flatnonzero(directed[v]).tolist()) for v in range(d))
        parents = tuple(frozenset(np.flatnonzero(directed[:, v]).tolist()) for v in range(d))
        component = [-1] * d
        for start in range(d):
            if component[start] >= 0:
                continue
            component[start] = start
            stack = [start]
            while stack:
                for w in sib[stack.pop()]:
                    if component[w] < 0:
                        component[w] = start
                        stack.append(w)
        return cls(cpdag, sib, children, parents, tuple(component))

    def members(self, v: int) -> set[int]:
        """The vertices of the chain component of ``v``."""
        return {w for w, c in enumerate(self.component) if c == self.component[v]}

    def explicit_ancestors(self, y: int) -> set[int]:
        """``y`` and the vertices with a path of directed edges to ``y``."""
        found, stack = {y}, [y]
        while stack:
            for p in self.parents[stack.pop()]:
                if p not in found:
                    found.add(p)
                    stack.append(p)
        return found


def critical_set(graph: ChainGraph, x: int, y: int) -> set[int]:
    """The critical set of ``x`` with respect to ``y``: the neighbors of ``x`` on a chordless path
    from ``x`` to ``y`` that no edge points back along.

    For ``x`` with no directed path to ``y``, the set is the siblings of ``x`` that start a
    chordless undirected path, within the chain component of ``x``, to a vertex of that component
    with a directed path to ``y`` or to ``y`` itself (Fang et al. 2022, Proposition 4); the search
    is their Algorithm 2, which extends each path by a vertex not adjacent to its last but one.
    Raises ``ValueError`` when ``x`` has a directed path to ``y``.
    """
    targets = graph.explicit_ancestors(y)
    if x in targets:
        raise ValueError(f"vertex {x} has a directed path to vertex {y}")
    targets &= graph.members(x)
    found: set[int] = set()
    if not targets:
        return found
    queue = deque((a, x, a) for a in sorted(graph.sib[x]))
    queued = set(queue)
    while queue:
        start, before, last = queue.popleft()
        if start in found:
            continue
        if last in targets:
            found.add(start)
            continue
        for nxt in sorted(graph.sib[last] - graph.sib[before] - {before}):
            triple = (start, last, nxt)
            if triple not in queued:
                queued.add(triple)
                queue.append(triple)
    return found


def claim_clauses(
    graph: ChainGraph,
    required_paths: Iterable[tuple[int, int]],
    forbidden_paths: Iterable[tuple[int, int]],
) -> list[Clause]:
    """The clauses of the claims ``x`` is an ancestor of ``y`` and ``x`` is not one (Fang et al.
    2025, Theorem 17), over the class of ``graph``.

    A pair of ``required_paths`` is an ancestor claim: one clause, ``x`` over its critical set, and
    none when ``x`` has a directed path to ``y`` in every DAG of the class. A pair of
    ``forbidden_paths`` is a non-ancestor claim: a clause ``c`` over ``{x}`` for each vertex ``c``
    of the critical set, and the clause ``x`` over no vertex, which no DAG meets, when ``x`` has a
    directed path to ``y`` in every DAG.
    """
    clauses: list[Clause] = []
    for x, y in required_paths:
        if x not in graph.explicit_ancestors(y):
            clauses.append((x, frozenset(critical_set(graph, x, y))))
    for x, y in forbidden_paths:
        if x in graph.explicit_ancestors(y):
            clauses.append((x, frozenset()))
        else:
            clauses.extend((c, frozenset({x})) for c in sorted(critical_set(graph, x, y)))
    return clauses


def _reduced(graph: ChainGraph, clauses: Iterable[Clause]) -> list[Clause]:
    """The reduced form of ``clauses`` (Fang et al. 2025, Equation 3): a clause that a directed
    edge of the class meets is dropped, and the others keep only the heads that are siblings of the
    tail."""
    return [(t, heads & graph.sib[t]) for t, heads in clauses if not heads & graph.children[t]]


def _simplicial(graph: ChainGraph, v: int, nodes: set[int]) -> bool:
    """Whether the neighbors of ``v`` among ``nodes`` are pairwise adjacent."""
    near = graph.sib[v] & nodes
    return all(near - {a} <= graph.sib[a] for a in near)


def _within(clauses: Iterable[Clause], nodes: set[int]) -> list[Clause]:
    """The clauses whose tail and heads lie in ``nodes`` (Fang et al. 2025, Equation 4)."""
    return [(t, heads) for t, heads in clauses if t in nodes and heads <= nodes]


def _eliminate(
    graph: ChainGraph, nodes: set[int], clauses: list[Clause], keep: int | None = None
) -> set[int]:
    """Remove potential leaf nodes other than ``keep`` from ``nodes`` until none is left; return
    the vertices that remain.

    A potential leaf node (Fang et al. 2025, Definition 24) is simplicial among ``nodes`` and the
    tail of no clause over them; removing it removes the clauses whose heads hold it, which it
    meets. Removing a vertex keeps every other potential leaf node one, so the vertices left do not
    depend on the order, and only the neighbors of a removed vertex and the tails of the clauses it
    meets need a new look.
    """
    nodes = set(nodes)
    clauses = _within(clauses, nodes)
    tails = Counter(t for t, _ in clauses)
    meets: dict[int, list[int]] = {}
    for k, (_, heads) in enumerate(clauses):
        for h in heads:
            meets.setdefault(h, []).append(k)
    live = [True] * len(clauses)

    def leaf(v: int) -> bool:
        return v != keep and not tails[v] and _simplicial(graph, v, nodes)

    queue = deque(v for v in sorted(nodes) if leaf(v))
    queued = set(queue)
    while queue:
        y = queue.popleft()
        nodes.discard(y)
        look = graph.sib[y] & nodes
        for k in meets.get(y, ()):
            if live[k]:
                live[k] = False
                tails[clauses[k][0]] -= 1
                look |= {clauses[k][0]} & nodes
        for v in sorted(look - queued):
            if leaf(v):
                queue.append(v)
                queued.add(v)
    return nodes


def _consistent_on(graph: ChainGraph, nodes: set[int], clauses: list[Clause]) -> bool:
    """Algorithm 1 of Fang et al. (2025) on the vertices ``nodes`` with the reduced clauses over
    them: the clauses are consistent when removing potential leaf nodes leaves none."""
    return not _eliminate(graph, nodes, clauses)


def _components(graph: ChainGraph, clauses: list[Clause]) -> list[set[int]]:
    """The chain components that hold the tail of a reduced clause; a component without one meets
    every clause in any of its orientations that adds no collider."""
    seen = sorted({graph.component[t] for t, _ in clauses})
    return [graph.members(c) for c in seen]


def consistent(graph: ChainGraph, clauses: Iterable[Clause]) -> bool:
    """Whether some DAG of the class of ``graph`` meets every clause (Fang et al. 2025, Theorem 25
    and Algorithm 1)."""
    reduced = _reduced(graph, clauses)
    return all(_consistent_on(graph, nodes, reduced) for nodes in _components(graph, reduced))


def _orientation_component(graph: ChainGraph, x: int, clauses: list[Clause]) -> set[int]:
    """The maximal orientation component for ``x`` (Fang et al. 2025, Algorithm 3 and Theorem 39):
    the chain component of ``x`` less the potential leaf nodes other than ``x`` that can be removed
    in turn. ``clauses`` are reduced and consistent with the class."""
    return _eliminate(graph, graph.members(x), clauses, keep=x)


def clause_mpdag(graph: ChainGraph, clauses: Iterable[Clause]) -> np.ndarray:
    """The MPDAG of the DAGs of the class of ``graph`` that meet ``clauses``, which must be
    consistent with it.

    ``y -> x`` is in every such DAG exactly when the class directs it or the maximal orientation
    component for ``x`` holds ``y`` (Fang et al. 2025, Corollary 38). Returns a new mark matrix.
    """
    reduced = _reduced(graph, clauses)
    out = graph.marks.copy()
    for nodes in _components(graph, reduced):
        for x in sorted(nodes):
            for y in graph.sib[x] & _orientation_component(graph, x, reduced):
                out[x, y] = ARROW  # y -> x: the mark at x is an arrowhead
    return out


def _directed(mpdag: np.ndarray, graph: ChainGraph) -> list[Clause]:
    """The edges ``mpdag`` directs and the class of ``graph`` leaves undirected, as clauses."""
    lines = (graph.marks == TAIL) & (graph.marks.T == TAIL)
    found = np.argwhere((mpdag == TAIL) & (mpdag.T == ARROW) & lines)
    return [(i, frozenset({j})) for i, j in found.tolist()]


def residual_clauses(
    graph: ChainGraph, clauses: Iterable[Clause], mpdag: np.ndarray
) -> list[Clause]:
    """The clauses the MPDAG ``mpdag`` of ``clauses`` over the class of ``graph`` cannot show,
    each with the fewest heads (Fang et al. 2025, Algorithm 4, lines 9-22).

    A clause is dropped while the MPDAG's edges and the other clauses imply it: when giving each of
    its heads as a direct cause of its tail is inconsistent with them (their Theorem 27). A head is
    then dropped from a clause while the clause without it is implied by the MPDAG's edges and the
    clauses, the restricted class of their Definition 28.
    """
    edges = _directed(mpdag, graph)
    into = (mpdag == ARROW) & (mpdag.T == TAIL)  # [t, h]: h -> t in every DAG kept
    # A clause the MPDAG meets is implied by it, and a head the MPDAG directs into the tail can
    # never be the one the clause names.
    left = [
        (t, frozenset(h for h in heads if not into[t, h]))
        for t, heads in dict.fromkeys(_reduced(graph, clauses))
        if not any(mpdag[t, h] == TAIL and mpdag[h, t] == ARROW for h in heads)
    ]

    def implied(t: int, heads: frozenset[int], others: list[Clause]) -> bool:
        # Only the chain component of t changes; every clause lies within one component.
        near = [c for c in (*edges, *others) if graph.component[c[0]] == graph.component[t]]
        return not consistent(graph, [*((h, frozenset({t})) for h in heads), *near])

    # A clause that is not implied by some clauses is not implied by fewer, so one pass removes
    # every clause the others imply.
    k = 0
    while k < len(left):
        t, heads = left[k]
        if implied(t, heads, left[:k] + left[k + 1 :]):
            del left[k]
        else:
            k += 1
    for k, (t, heads) in enumerate(left):
        for h in sorted(heads):
            if len(heads) > 1 and implied(t, heads - {h}, left):
                heads = heads - {h}
                left[k] = (t, heads)
    return left


def restrict(
    pdag: np.ndarray,
    required_paths: np.ndarray | None,
    forbidden_paths: np.ndarray | None,
    label: Callable[[int], str] = str,
) -> tuple[np.ndarray, list[Clause]]:
    """The DAGs ``pdag`` stands for in which each claim holds: their MPDAG and residual clauses.

    ``pdag`` is a learned graph, a CPDAG or an MPDAG, as a mark matrix; it stands for the DAGs of
    the class of its consistent extensions that direct its directed edges alike.
    ``required_paths[i, j]`` claims that ``i`` is an ancestor of ``j``, and
    ``forbidden_paths[i, j]`` that it is not. The claims and the directed edges of ``pdag`` become
    clauses over the CPDAG of the class (:func:`claim_clauses`), and the result is their
    :func:`clause_mpdag` and :func:`residual_clauses`. ``label`` names a vertex in messages.

    Raises :class:`~andrey.core.knowledge.KnowledgeError` when ``pdag`` has no consistent extension,
    or when no DAG it stands for meets the claims; the message names a claim no such DAG meets, or
    says that only the claims together fail.
    """
    from .orient import cpdag_of_pdag

    try:
        graph = ChainGraph.of(cpdag_of_pdag(pdag))
    except ValueError as error:
        raise KnowledgeError(
            f"the graph found stands for no DAG ({error}), so required_paths and forbidden_paths "
            "cannot restrict it"
        ) from error
    edges = _directed(pdag, graph)
    pairs = {
        part: [] if mask is None else [tuple(p) for p in np.argwhere(mask).tolist()]
        for part, mask in (("required_paths", required_paths), ("forbidden_paths", forbidden_paths))
    }
    clauses = [*edges, *claim_clauses(graph, pairs["required_paths"], pairs["forbidden_paths"])]
    if not consistent(graph, clauses):
        raise KnowledgeError(_conflict(graph, edges, pairs, label))
    mpdag = clause_mpdag(graph, clauses)
    return mpdag, sorted((t, frozenset(h)) for t, h in residual_clauses(graph, clauses, mpdag))


def _conflict(
    graph: ChainGraph,
    edges: list[Clause],
    pairs: dict[str, list[tuple[int, int]]],
    label: Callable[[int], str],
) -> str:
    """The message for claims no DAG of the learned graph meets: the first claim that fails alone,
    or else all of them together."""
    for x, y in pairs["required_paths"]:
        if not consistent(graph, [*edges, *claim_clauses(graph, [(x, y)], [])]):
            return (
                f"knowledge required_paths lists ({label(x)}, {label(y)}), but no DAG the graph "
                f"found stands for has a directed path from {label(x)} to {label(y)}"
            )
    for x, y in pairs["forbidden_paths"]:
        if not consistent(graph, [*edges, *claim_clauses(graph, [], [(x, y)])]):
            return (
                f"knowledge forbidden_paths lists ({label(x)}, {label(y)}), but every DAG the "
                f"graph found stands for has a directed path from {label(x)} to {label(y)}"
            )
    return (
        "knowledge required_paths and forbidden_paths hold together in no DAG the graph found "
        "stands for, though each holds in some"
    )
