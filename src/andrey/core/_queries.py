"""Graph queries behind ``GraphStructure.parents``, ``children``, ``ancestors``, ``descendants``,
``markov_blanket``, and ``d_separated``.

The methods' docstrings define each query for each graph kind and cite the definitions.
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np

from ._convert import _positions, mag_of_pag
from .structure import ARROW, TAIL, GraphStructure

# One incident edge seen from a node: (other end, mark at this node, mark at the other end).
Incident = tuple[int, int, int]


def incident(graph: GraphStructure) -> list[list[Incident]]:
    """Each node's edges as ``(other end, mark here, mark there)``; a self-loop is listed once."""
    rows, cols, near, far = graph._mark_pairs()
    out: list[list[Incident]] = [[] for _ in range(graph.n_nodes)]
    for i, j, mi, mj in zip(rows.tolist(), cols.tolist(), near.tolist(), far.tolist(), strict=True):
        out[i].append((j, mi, mj))
        if i != j:
            out[j].append((i, mj, mi))
    return out


def _into(graph: GraphStructure, edges: list[list[Incident]], v: int) -> list[int]:
    """The nodes with an edge directed into ``v`` (a self-loop of a ``digraph`` included)."""
    found = []
    for w, here, there in edges[v]:
        if w == v or (here == ARROW and there == TAIL):
            found.append(w)
        elif graph.kind == "digraph" and here == there == ARROW:  # a 2-cycle: w -> v and v -> w
            found.append(w)
    return found


def _out_of(graph: GraphStructure, edges: list[list[Incident]], v: int) -> list[int]:
    """The nodes ``v`` has an edge directed into (a self-loop of a ``digraph`` included)."""
    found = []
    for w, here, there in edges[v]:
        if w == v or (here == TAIL and there == ARROW):
            found.append(w)
        elif graph.kind == "digraph" and here == there == ARROW:
            found.append(w)
    return found


def parents(graph: GraphStructure, x: int) -> list[int]:
    return sorted(_into(graph, incident(graph), x))


def children(graph: GraphStructure, x: int) -> list[int]:
    return sorted(_out_of(graph, incident(graph), x))


def _reach(
    graph: GraphStructure, edges: list[list[Incident]], start: Collection[int], up: bool
) -> set[int]:
    """The nodes a directed path reaches from ``start`` against (``up``) or along the edges.

    ``start`` itself is in the result only when a directed path of one or more edges returns to it.
    """
    step = _into if up else _out_of
    seen: set[int] = set()
    stack = list(start)
    while stack:
        v = stack.pop()
        for w in step(graph, edges, v):
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return seen


def ancestors(graph: GraphStructure, x: int) -> list[int]:
    return sorted(_reach(graph, incident(graph), [x], up=True) - {x})


def descendants(graph: GraphStructure, x: int) -> list[int]:
    return sorted(_reach(graph, incident(graph), [x], up=False) - {x})


def member(graph: GraphStructure, what: str) -> GraphStructure:
    """The graph a separation or blanket query reads: one member of a ``cpdag``'s or ``pag``'s
    class, whose answers every member shares, or the graph itself.

    Raises ``ValueError`` for a ``digraph``, for a ``dag`` with a directed cycle, and for a ``mag``
    that is not ancestral or a ``pag`` whose MAG (Zhang 2008) is not.
    """
    if graph.kind == "digraph":
        raise ValueError(
            f"{what} is not defined here for a digraph: its directed cycles need a separation "
            "criterion for cyclic models, which depends on the model's assumptions"
        )
    if graph.kind == "cpdag":
        return graph.to_dag(tie_break="label")
    graph.validate()
    if graph.kind == "pag":
        mag = GraphStructure.from_numpy(
            mag_of_pag(graph.to_numpy()), kind="mag", labels=graph.labels
        )
        problem = _not_ancestral(mag)
        if problem:
            raise ValueError(f"the pag is not a PAG: the MAG built from it {problem}")
        return mag
    if graph.kind == "dag":
        cycle = _on_a_cycle(graph)
        if cycle is not None:
            raise ValueError(f"the dag has a directed cycle through node {_name(graph, cycle)!r}")
    if graph.kind == "mag":
        problem = _not_ancestral(graph)
        if problem:
            raise ValueError(f"the mag is not ancestral: it {problem}")
    return graph


def _not_ancestral(graph: GraphStructure) -> str | None:
    """Why ``graph`` is not an ancestral graph (Richardson and Spirtes 2002, Sec. 3.1), or ``None``.

    An ancestral graph has no directed cycle, no edge ``a <-> b`` with ``a`` an ancestor of ``b``,
    and no arrowhead at an end of an undirected edge ``a -- b``.
    """
    cycle = _on_a_cycle(graph)
    if cycle is not None:
        return f"has a directed cycle through node {_name(graph, cycle)!r}"
    edges = incident(graph)
    for a in range(graph.n_nodes):
        for b, here, there in edges[a]:
            if here == there == ARROW and a in _reach(graph, edges, [b], up=True):
                return (
                    f"has {_name(graph, a)!r} <-> {_name(graph, b)!r} with an ancestor at one end"
                )
            if here == there == TAIL and any(m == ARROW for _, m, _ in edges[a]):
                return f"has an arrowhead at {_name(graph, a)!r}, an end of an undirected edge"
    return None


def _on_a_cycle(graph: GraphStructure) -> int | None:
    """A node on a directed cycle, or ``None`` when Kahn's algorithm removes every node."""
    edges = incident(graph)
    waiting = [len(_into(graph, edges, v)) for v in range(graph.n_nodes)]
    stack = [v for v, k in enumerate(waiting) if k == 0]
    while stack:
        for w in _out_of(graph, edges, stack.pop()):
            waiting[w] -= 1
            if waiting[w] == 0:
                stack.append(w)
    left = [v for v, k in enumerate(waiting) if k > 0]
    return left[0] if left else None


def separated(graph: GraphStructure, xs: list[int], ys: list[int], zs: list[int]) -> bool:
    """Whether no path joins ``xs`` and ``ys`` that is open given ``zs``.

    A walk over ``(node, entered through an arrowhead)`` states: a node entered and left through
    arrowheads is a collider, passable when it is in ``zs`` or an ancestor of it; any other inner
    node is passable when it is not in ``zs``.
    """
    edges = incident(graph)
    given = set(zs)
    opened = given | _reach(graph, edges, zs, up=True)
    targets = set(ys)
    seen: set[tuple[int, bool]] = set()
    stack: list[tuple[int, bool]] = []
    for x in xs:
        for w, _, there in edges[x]:
            stack.append((w, there == ARROW))
    while stack:
        state = stack.pop()
        if state in seen:
            continue
        seen.add(state)
        v, entered_arrow = state
        if v in targets:
            return False
        for w, here, there in edges[v]:
            collider = entered_arrow and here == ARROW
            if (v in opened) if collider else (v not in given):
                stack.append((w, there == ARROW))
    return True


def collider_connected(graph: GraphStructure, x: int) -> list[int]:
    """The nodes joined to ``x`` by a path whose every inner node is a collider."""
    edges = incident(graph)
    found: set[int] = set()
    seen: set[int] = set()
    stack: list[int] = []
    for w, _, there in edges[x]:
        found.add(w)
        if there == ARROW:
            stack.append(w)
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        for w, here, there in edges[v]:
            if here == ARROW:  # v is a collider between the arrowhead it was entered by and this
                found.add(w)
                if there == ARROW:
                    stack.append(w)
    found.discard(x)
    return sorted(found)


def node(graph: GraphStructure, x: str | int, what: str) -> int:
    """The position of one node, a label or a 0-based position."""
    if isinstance(x, (list, tuple, set, frozenset, np.ndarray)):
        raise TypeError(f"{what} takes one node, a label or a 0-based position, got {x!r}")
    return _positions(graph, [x], what)[0]


def nodes(graph: GraphStructure, xs: str | int | Collection[str | int], what: str) -> list[int]:
    """The positions of a node or a collection of nodes."""
    if isinstance(xs, (str, int)):
        return _positions(graph, [xs], what)
    if isinstance(xs, np.integer):
        return _positions(graph, [int(xs)], what)
    return _positions(graph, xs, what)


def named(graph: GraphStructure, positions: list[int], index: bool) -> list[str | int]:
    """Positions as ``oriented_edges`` names nodes: labels, or indices with ``index``."""
    if index or graph.labels is None:
        return list(positions)
    return [graph.labels[k] for k in positions]


def _name(graph: GraphStructure, k: int) -> str | int:
    return k if graph.labels is None else graph.labels[k]
