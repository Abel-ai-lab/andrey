"""Orientation primitives shared by the score- and constraint-based engines.

``meek`` completes a PDAG to a CPDAG with Meek's rules R1-R3, ``pdag2dag`` returns a consistent
DAG extension of a PDAG, ``dag2cpdag`` returns a DAG's CPDAG, and ``orient_colliders`` orients a
skeleton's unshielded colliders as PC does. :func:`to_structure` and :func:`from_structure` convert
to and from a :class:`~andrey.core.structure.GraphStructure`.

Every function reads a dense ``(d, d)`` ``int8`` mark matrix in the convention of
:mod:`andrey.core.structure`: ``adj[i, j]`` is the mark at ``i`` on edge ``i-j``, one of ``NULL``
(0), ``TAIL`` (1), or ``ARROW`` (2). ``i -> j`` is ``adj[i, j] == TAIL`` with
``adj[j, i] == ARROW``, and ``i -- j`` is ``TAIL`` at both ends. The diagonal is zero and the
support symmetric.

References
----------
Chickering (2002). Learning Equivalence Classes of Bayesian-Network Structures.
https://jmlr.org/papers/v2/chickering02a.html

Dor and Tarsi (1992). A simple algorithm to construct a consistent extension of a partially
oriented graph. https://ftp.cs.ucla.edu/pub/stat_ser/r185-dor-tarsi.pdf

Meek (1995). Causal Inference and Causal Explanation with Background Knowledge.
https://arxiv.org/abs/1302.4972
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from .structure import ARROW, NULL, TAIL, GraphStructure, Kind

__all__ = [
    "meek",
    "pdag2dag",
    "dag2cpdag",
    "orient_colliders",
    "to_structure",
    "from_structure",
]


# ---- encoding helpers ---------------------------------------------------------------------------


def _validate_adjacency(adj: np.ndarray) -> np.ndarray:
    """Return an owned ``int8`` copy after checking the square/mark/support invariants.

    Rejects non-square input, marks outside ``{NULL, TAIL, ARROW}``, a non-zero diagonal, and
    asymmetric support (an ``i-j`` half-edge with no ``j-i`` mark).
    """
    a = np.asarray(adj)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"expected a square (d, d) matrix, got shape {a.shape}")
    out = a.astype(np.int8)
    if ((out != NULL) & (out != TAIL) & (out != ARROW)).any():
        raise ValueError(f"marks must be in {{{NULL}, {TAIL}, {ARROW}}} (NULL/TAIL/ARROW)")
    if out.size and np.diagonal(out).any():
        raise ValueError("adjacency must have a zero diagonal (no self-loops)")
    if not np.array_equal(out != NULL, out.T != NULL):
        raise ValueError("adjacency support must be symmetric (every i-j has a j-i mark)")
    return out


def _is_adjacent(adj: np.ndarray, a: int, b: int) -> bool:
    return bool(adj[a, b] != NULL)


def _is_directed(adj: np.ndarray, a: int, b: int) -> bool:
    """True when ``a -> b`` (tail at ``a``, arrowhead at ``b``)."""
    return bool(adj[a, b] == TAIL and adj[b, a] == ARROW)


def _is_undirected(adj: np.ndarray, a: int, b: int) -> bool:
    """True when ``a -- b`` (tails at both ends)."""
    return bool(adj[a, b] == TAIL and adj[b, a] == TAIL)


def _orient(adj: np.ndarray, a: int, b: int) -> None:
    """Direct the edge as ``a -> b`` in place."""
    adj[a, b] = TAIL
    adj[b, a] = ARROW


def _reaches_directed(adj: np.ndarray, src: int, dst: int) -> bool:
    """True when a directed path ``src -> ... -> dst`` exists over current arrowheads."""
    if src == dst:
        return True
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    seen[src] = True
    stack = [src]
    while stack:
        node = stack.pop()
        # children c of node: node -> c  (tail at node, arrowhead at c)
        children = np.nonzero((adj[node] == TAIL) & (adj[:, node] == ARROW))[0]
        for c in children.tolist():
            if c == dst:
                return True
            if not seen[c]:
                seen[c] = True
                stack.append(c)
    return False


def _topological_positions(adj: np.ndarray) -> np.ndarray:
    """Rank of each node in a topological order of the DAG (Kahn's algorithm).

    Node ``v``'s parents are ``{u : adj[v, u] == ARROW}``. Raises if the directed graph has a cycle.
    """
    n = adj.shape[0]
    parents = adj == ARROW  # parents[v, u] -> u is a parent of v
    in_degree = parents.sum(axis=1).astype(np.int64)
    ready = [v for v in range(n) if in_degree[v] == 0]
    ready.sort()
    pos = np.empty(n, dtype=np.int64)
    rank = 0
    while ready:
        v = ready.pop(0)
        pos[v] = rank
        rank += 1
        # children c of v: adj[c, v] == ARROW
        children = np.nonzero(parents[:, v])[0]
        newly = []
        for c in children.tolist():
            in_degree[c] -= 1
            if in_degree[c] == 0:
                newly.append(c)
        if newly:
            ready = sorted(ready + newly)
    if rank != n:
        raise ValueError("input is not acyclic; dag2cpdag requires a DAG")
    return pos


# ---- structure interop --------------------------------------------------------------------------


def to_structure(
    adj: np.ndarray, *, kind: Kind, labels: tuple[str, ...] | None = None
) -> GraphStructure:
    """Wrap a validated endpoint-mark matrix as a :class:`GraphStructure` of the given ``kind``.

    Thin emitter: hands the checked adjacency to ``GraphStructure.from_numpy`` (the canonical CSR
    store is built there); adds no CSR logic of its own.
    """
    return GraphStructure.from_numpy(_validate_adjacency(adj), kind=kind, labels=labels)


def from_structure(structure: GraphStructure) -> np.ndarray:
    """Dense ``int8`` endpoint-mark matrix of a ``GraphStructure`` (inverse of ``to_structure``)."""
    return structure.to_numpy(dtype=np.int8)


# ---- Meek completion ----------------------------------------------------------------------------


def meek(pdag: np.ndarray, *, _trace: list[dict] | None = None) -> np.ndarray:
    """Complete a PDAG to a CPDAG by applying Meek's rules R1-R3 to a fixed point.

    Returns a new endpoint-mark matrix. Each rule directs an undirected edge only when doing so
    avoids a new unshielded collider (R1), a directed cycle (R2), or an inconsistency around a
    shared child (R3); an orientation that would close a directed cycle is skipped. Adjacency (which
    pairs share an edge) is never changed -- only tails become arrowheads. R4 is not required to
    complete a v-structure pattern in the absence of background knowledge.

    The completion is defined for a collider-consistent PDAG (as produced by
    :func:`orient_colliders`); on an inconsistent PDAG the fixed point can depend on rule-firing
    order.

    The private ``_trace`` sink records each firing's rule, participating nodes, and full
    before/after endpoint matrices. Matrices are copied only when a trace is requested.
    """
    adj = _validate_adjacency(pdag)
    changed = True
    while changed:
        changed = False
        # R1: i -> j -- k with i, k non-adjacent  =>  j -> k
        directed = np.argwhere((adj == TAIL) & (adj.T == ARROW))  # rows (i, j): i -> j
        for i, j in directed.tolist():
            for k in np.nonzero((adj[j] == TAIL) & (adj[:, j] == TAIL))[0].tolist():
                if k == i or _is_adjacent(adj, i, k):
                    continue
                if not _reaches_directed(adj, k, j):
                    if _trace is not None:
                        before = adj.copy()
                    _orient(adj, j, k)
                    if _trace is not None:
                        _trace.append(
                            dict(rule="R1", triple=(i, j, k), before=before, after=adj.copy())
                        )
                    changed = True
        # R2: i -> l -> k with i -- k  =>  i -> k
        undirected = np.argwhere((adj == TAIL) & (adj.T == TAIL))  # rows (i, k): i -- k
        for i, k in undirected.tolist():
            if not _is_undirected(adj, i, k):
                continue
            mids = np.nonzero((adj[i, :] == TAIL) & (adj[:, i] == ARROW))[0]  # mid: i -> mid
            hit = any(_is_directed(adj, int(mid), k) for mid in mids.tolist())
            if hit and not _reaches_directed(adj, k, i):
                if _trace is not None:
                    before = adj.copy()
                    mid = next(int(m) for m in mids if _is_directed(adj, int(m), k))
                _orient(adj, i, k)
                if _trace is not None:
                    _trace.append(
                        dict(rule="R2", triple=(i, mid, k), before=before, after=adj.copy())
                    )
                changed = True
        # R3: i -- j, i -- k, j -> t, k -> t, i -- t, j and k non-adjacent  =>  i -> t
        undirected = np.argwhere((adj == TAIL) & (adj.T == TAIL))  # rows (i, t): i -- t
        for i, t in undirected.tolist():
            if not _is_undirected(adj, i, t):
                continue
            # common undirected neighbors of i that are directed parents of t
            cand = [
                m
                for m in np.nonzero((adj[i] == TAIL) & (adj[:, i] == TAIL))[0].tolist()
                if m != t and _is_directed(adj, m, int(t))
            ]
            found = False
            for a in range(len(cand)):
                for b in range(a + 1, len(cand)):
                    if not _is_adjacent(adj, cand[a], cand[b]):
                        found = True
                        break
                if found:
                    break
            if found and not _reaches_directed(adj, t, i):
                if _trace is not None:
                    before = adj.copy()
                _orient(adj, i, t)
                if _trace is not None:
                    _trace.append(
                        dict(
                            rule="R3",
                            triple=(i, cand[a], cand[b], t),
                            before=before,
                            after=adj.copy(),
                        )
                    )
                changed = True
    return adj


# ---- PDAG to DAG --------------------------------------------------------------------------------


def pdag2dag(pdag: np.ndarray) -> np.ndarray:
    """Return a consistent DAG extension of a PDAG (Dor & Tarsi), raising if none exists.

    Keeps every directed edge of the PDAG, then repeatedly removes a node ``x`` that has no outgoing
    directed edge and whose undirected neighbors are each adjacent to all neighbors of ``x``
    (so orienting the remaining undirected edges into ``x`` adds no collider). Raises ``ValueError``
    when no such node exists: the PDAG admits no consistent extension.
    """
    src = _validate_adjacency(pdag)
    n = src.shape[0]
    # Seed the extension with the PDAG's directed edges only.
    out = np.zeros((n, n), dtype=np.int8)
    directed = (src == TAIL) & (src.T == ARROW)  # (i, j): i -> j
    out[directed] = TAIL
    out.T[directed] = ARROW

    removed = np.zeros(n, dtype=bool)
    remaining = n
    while remaining > 0:
        progressed = False
        for i in range(n):
            if removed[i]:
                continue
            # i is a candidate sink: no outgoing directed edge to an active node (i -> c).
            children = np.nonzero((src[:, i] == ARROW) & (src[i, :] == TAIL))[0]
            if children.size and not removed[children].all():
                continue
            # Undirected neighbors of i (active) and all adjacent nodes (active).
            und = np.nonzero((src[i] == TAIL) & (src[:, i] == TAIL) & ~removed)[0]
            if und.size:
                adjacent = np.nonzero((src[i] != NULL) & ~removed)[0]
                if not _clique_condition(src, und, adjacent):
                    continue
            # Orient every undirected edge nb - i into i as nb -> i.
            for nb in np.nonzero((src[i] == TAIL) & (src[:, i] == TAIL))[0].tolist():
                if out[i, nb] == NULL and out[nb, i] == NULL:
                    _orient(out, nb, i)
            removed[i] = True
            remaining -= 1
            progressed = True
            break
        if not progressed:
            raise ValueError("PDAG admits no consistent DAG extension")
    return out


def _clique_condition(adj: np.ndarray, neighbours: np.ndarray, adjacent: np.ndarray) -> bool:
    """True when every node in ``neighbours`` is adjacent to all of ``adjacent`` except itself."""
    support = adj != NULL
    for nb in neighbours.tolist():
        row = support[nb, adjacent]
        if not np.all(row | (adjacent == nb)):
            return False
    return True


# ---- DAG to CPDAG -------------------------------------------------------------------------------


def dag2cpdag(dag: np.ndarray) -> np.ndarray:
    """Return the CPDAG (essential graph) of a DAG by Chickering's compelled-edge labeling.

    Orders the directed edges by a topological order of the DAG, then labels each edge compelled
    (its direction is shared by every DAG in the equivalence class) or reversible. Compelled edges
    stay directed; reversible edges become undirected. The output CPDAG is independent of which
    valid topological order is used.
    """
    adj = _validate_adjacency(dag)
    n = adj.shape[0]
    parent = adj == ARROW  # parent[a, b] -> b is a parent of a  (b -> a)
    if (parent & parent.T).any():
        raise ValueError("dag2cpdag requires a DAG (no undirected or bidirected edges)")
    m = int(np.count_nonzero(parent))
    out = np.zeros((n, n), dtype=np.int8)
    if m == 0:
        return out

    pos = _topological_positions(adj)
    heads, tails = np.nonzero(parent)  # parent[head, tail]: tail -> head
    order = np.lexsort((pos[tails], -pos[heads]))
    tails = tails[order].astype(np.int64)
    heads = heads[order].astype(np.int64)

    # sign: 0 unknown, 1 compelled, -1 reversible.
    sign = np.zeros(m, dtype=np.int8)
    in_edges = [np.nonzero(heads == node)[0] for node in range(n)]
    pair_idx = -np.ones((n, n), dtype=np.int64)
    pair_idx[tails, heads] = np.arange(m)

    unknown = m
    ptr = m - 1
    while unknown > 0:
        while ptr >= 0 and sign[ptr] != 0:
            ptr -= 1
        if ptr < 0:
            break
        i = int(tails[ptr])  # x
        j = int(heads[ptr])  # y, edge x -> y
        skip = False

        into_i = in_edges[i]
        if into_i.size:
            compelled_k = into_i[sign[into_i] == 1]  # w -> x compelled
            if compelled_k.size:
                w = tails[compelled_k]
                if np.any(~parent[j, w]):  # some w is not a parent of y
                    idy = in_edges[j][sign[in_edges[j]] == 0]
                    if idy.size:
                        sign[idy] = 1
                        unknown -= int(idy.size)
                    skip = True
                else:  # every such w is a parent of y: label w -> y compelled
                    idy = pair_idx[w, j]
                    idy = idy[idy >= 0]
                    idy = idy[sign[idy] == 0]
                    if idy.size:
                        sign[idy] = 1
                        unknown -= int(idy.size)
        if skip:
            continue

        # Is there z in Pa(y) \ {x} with z not in Pa(x)?
        diff = parent[j] & (~parent[i])
        diff_count = int(np.count_nonzero(diff))
        diff_count -= 1 if diff[i] else 0
        idy = in_edges[j]
        label = 1 if diff_count > 0 else -1
        if sign[ptr] == 0:
            sign[ptr] = label
            unknown -= 1
        idy = idy[sign[idy] == 0]
        if idy.size:
            sign[idy] = label
            unknown -= int(idy.size)

    compelled = sign == 1
    ct, ch = tails[compelled], heads[compelled]
    out[ch, ct] = ARROW  # arrowhead at head: tail -> head stays directed
    out[ct, ch] = TAIL
    rt, rh = tails[~compelled], heads[~compelled]
    out[rt, rh] = TAIL  # reversible edge becomes undirected
    out[rh, rt] = TAIL
    return out


# ---- collider orientation -----------------------------------------------------------------------


def orient_colliders(
    skeleton: np.ndarray, sepsets: Mapping[tuple[int, int], Iterable[int] | None]
) -> np.ndarray:
    """Orient unshielded colliders on an undirected skeleton (the PC collider rule).

    For every unshielded triple ``x - z - y`` (``x`` and ``y`` both adjacent to ``z`` but not to
    each other), ``z`` is a collider ``x -> z <- y`` iff ``z`` is not in ``sepsets[(x, y)]``. Every
    other edge is left undirected. ``sepsets`` maps a node pair to its separating set; the key is
    looked up in either order and a missing pair (or ``None`` value) is treated as an empty set.

    Conflicting colliders resolve last-writer-wins on the shared edge. An engine that prioritizes
    stronger colliders must resolve conflicts using strengths unavailable in ``sepsets``.
    """
    adj = _validate_adjacency(skeleton)
    n = adj.shape[0]
    pdag = np.where(adj != NULL, np.int8(TAIL), np.int8(NULL))
    np.fill_diagonal(pdag, np.int8(NULL))
    for z in range(n):
        nbrs = np.nonzero(adj[z] != NULL)[0].tolist()
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                x, y = nbrs[a], nbrs[b]
                if _is_adjacent(adj, x, y):
                    continue
                if z not in _lookup_sepset(sepsets, x, y):
                    pdag[z, x] = ARROW
                    pdag[x, z] = TAIL
                    pdag[z, y] = ARROW
                    pdag[y, z] = TAIL
    return pdag


def _lookup_sepset(
    sepsets: Mapping[tuple[int, int], Iterable[int] | None], x: int, y: int
) -> frozenset[int]:
    """The separating set for the pair, in either key order; empty when absent or ``None``."""
    value = sepsets.get((x, y))
    if value is None:
        value = sepsets.get((y, x))
    if value is None:
        return frozenset()
    return frozenset(int(v) for v in value)
