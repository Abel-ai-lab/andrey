"""GES Insert and Delete operators and their validity tests over a CPDAG mark matrix.

Every function reads a dense ``(d, d)`` ``int8`` mark matrix in the convention of
:mod:`andrey.core.orient`. An Insert is valid when ``NA union T`` is a clique and every
semi-directed path from ``y`` to ``x`` meets it; a Delete is valid when ``NA`` without ``H`` is a
clique.

References
----------
Chickering (2002). Optimal Structure Identification With Greedy Search.
https://jmlr.org/papers/v3/chickering02b.html
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
from itertools import chain, combinations

import numpy as np

from andrey.core import ARROW, NULL, TAIL
from andrey.core.orient import _cpdag_of_pdag, _cpdag_of_pdag_near


def adjacent(adj: np.ndarray, node: int) -> np.ndarray:
    """Nodes sharing an edge with ``node`` (support is symmetric, so the row suffices)."""
    return np.nonzero(adj[node] != NULL)[0]


def undirected_neighbors(adj: np.ndarray, node: int) -> np.ndarray:
    """Nodes joined to ``node`` by an undirected edge (``TAIL`` at both ends)."""
    return np.nonzero((adj[node] == TAIL) & (adj[:, node] == TAIL))[0]


def parents(adj: np.ndarray, node: int) -> np.ndarray:
    """Directed parents of ``node``: each ``k`` with ``k -> node`` (``adj[node, k] == ARROW``)."""
    return np.nonzero(adj[node] == ARROW)[0]


def subsets(elements: Sequence[int]) -> Iterable[tuple[int, ...]]:
    """Every subset of ``elements`` as a tuple, empty set first."""
    return chain.from_iterable(combinations(elements, r) for r in range(len(elements) + 1))


def is_clique(adj: np.ndarray, nodes: Iterable[int]) -> bool:
    """True when every pair in ``nodes`` is adjacent (ignoring edge direction)."""
    seq = [int(v) for v in nodes]
    for a in range(len(seq) - 1):
        row = adj[seq[a]]
        for b in range(a + 1, len(seq)):
            if row[seq[b]] == NULL:
                return False
    return True


# Neighborhood size below which the subset-validity scans (clique_extensions/clique_residuals) take
# the plain per-subset ``is_clique`` scan instead of the adjacency bitmasks that collapse the 2^m
# lattice. The bitmasks only repay their fixed setup cost (submatrix extraction, conflict masks)
# once the subset count 2^m is large; for a handful of subsets -- the overwhelmingly common sparse
# case, where nearly every (x, y) has an empty or tiny t0 -- the plain scan is cheaper, so gating
# on m keeps the fast path from ever regressing below the plain-scan cost.
_LATTICE_MIN_M = 4

# |t0| ceiling below which the position-subset plan is materialized and cached (2^15 = 32768 tuples
# per entry). Above it the plan is generated lazily so a single hub node cannot balloon the cache.
_PLAN_CACHE_MAX_M = 15


@lru_cache(maxsize=256)
def _position_plan(m: int) -> tuple[tuple[int, ...], ...]:
    """The subset lattice over positions ``0..m-1``, materialized once and cached.

    ``subsets(range(m))`` depends only on ``m`` -- not on the node identities -- so one entry keyed
    by the neighborhood size serves every candidate whose undirected slack has that size, sparing
    the repeated ``combinations`` rebuild. The sequence is exactly ``subsets``'s, so a subset's
    position (``t_index``/``h_index``) is its index here and the raw-key parallel reduce is intact.
    """
    return tuple(subsets(tuple(range(m))))


def _lattice_conflicts(sub_nonadj: np.ndarray, m: int) -> list[int]:
    """Per-position non-adjacency bitmasks: ``conflict[i]`` bit ``j`` set iff ``i``, ``j`` clash.

    ``sub_nonadj`` is the ``m x m`` boolean ``adj[submatrix] == NULL`` over the neighborhood nodes;
    two positions clash when the corresponding nodes are non-adjacent, which is what forbids them
    from sharing a clique. Symmetric, so a subset is a clique iff no member's mask meets the subset.
    """
    conflict = [0] * m
    for i in range(m):
        row = sub_nonadj[i]
        for j in range(i + 1, m):
            if row[j]:
                conflict[i] |= 1 << j
                conflict[j] |= 1 << i
    return conflict


def clique_extensions(
    adj: np.ndarray, na: Iterable[int], t0: Sequence[int]
) -> Iterable[tuple[int, tuple[int, ...]]]:
    """Yield ``(t_index, T)`` for every ``T`` in ``subsets(t0)`` with ``NA union T`` a clique.

    ``t_index`` is ``T``'s position in the *full* ``subsets(t0)`` enumeration -- never renumbered --
    so a caller keying its reduce on it (``_parallel_ges``) stays bit-identical to serial. The
    yielded stream equals ``[(i, T) for i, T in enumerate(subsets(t0)) if
    is_clique(adj, set(na) | set(T))]``: non-clique subsets never yield a GES candidate, so omitting
    them changes nothing. ``na`` and ``t0`` must be disjoint (the neighborhood split guarantees
    it). Validity is decided from precomputed adjacency bitmasks in O(|T|) instead
    of a per-subset O(k^2) matrix scan, and the whole lattice is skipped when NA is already not a
    clique (superset-of-invalid pruning at the root).
    """
    na_list = [int(v) for v in na]
    t0 = list(t0)
    m = len(t0)

    if m < _LATTICE_MIN_M:
        # Small (the common sparse case): the plain per-subset is_clique scan, cheaper here than
        # the bitmask precompute.
        na_set = set(na_list)
        for t_index, T in enumerate(subsets(t0)):
            if is_clique(adj, na_set | set(T)):
                yield t_index, T
        return

    if not is_clique(adj, na_list):
        return  # NA is not a clique, so NA union T never is -- no subset can be valid.

    t0_arr = np.asarray(t0, dtype=np.intp)
    if na_list:
        na_arr = np.asarray(na_list, dtype=np.intp)
        na_ok = (adj[np.ix_(t0_arr, na_arr)] != NULL).all(axis=1)
    else:
        na_ok = np.ones(m, dtype=bool)
    conflict = _lattice_conflicts(adj[np.ix_(t0_arr, t0_arr)] == NULL, m)
    bad = 0  # positions whose node is not adjacent to all of NA -> in no clique extension
    for i in range(m):
        if not na_ok[i]:
            bad |= 1 << i

    plan = _position_plan(m) if m <= _PLAN_CACHE_MAX_M else subsets(tuple(range(m)))
    for t_index, pos in enumerate(plan):
        acc = 0
        ok = True
        for i in pos:
            if (bad >> i) & 1 or (conflict[i] & acc):
                ok = False
                break
            acc |= 1 << i
        if ok:
            yield t_index, tuple(t0[i] for i in pos)


def clique_residuals(adj: np.ndarray, h0: Sequence[int]) -> Iterable[tuple[int, tuple[int, ...]]]:
    """Yield ``(h_index, H)`` per ``H`` in ``subsets(h0)`` whose residual ``h0 \\ H`` is a clique.

    ``h_index`` is ``H``'s position in the full ``subsets(h0)`` enumeration. Same as the filter
    ``[(i, H) for i, H in enumerate(subsets(h0)) if is_clique(adj, set(h0) - set(H))]`` (the Delete
    validity test), but decided from precomputed non-adjacency bitmasks, not a per-subset rescan.
    """
    h0 = list(h0)
    m = len(h0)

    if m < _LATTICE_MIN_M:
        # Small (the common sparse case): the plain per-subset residual-clique scan, cheaper here
        # than the bitmask precompute.
        h0_set = set(h0)
        for h_index, H in enumerate(subsets(h0)):
            if is_clique(adj, h0_set - set(H)):
                yield h_index, H
        return

    h0_arr = np.asarray(h0, dtype=np.intp)  # m >= _LATTICE_MIN_M >= 1 here, so h0 is non-empty
    conflict = _lattice_conflicts(adj[np.ix_(h0_arr, h0_arr)] == NULL, m)
    full = (1 << m) - 1
    plan = _position_plan(m) if m <= _PLAN_CACHE_MAX_M else subsets(tuple(range(m)))
    for h_index, pos in enumerate(plan):
        h_mask = 0
        for i in pos:
            h_mask |= 1 << i
        residual = full & ~h_mask
        ok = True
        bits = residual
        while bits:
            lsb = bits & (-bits)
            i = lsb.bit_length() - 1
            if conflict[i] & residual:
                ok = False
                break
            bits ^= lsb
        if ok:
            yield h_index, tuple(h0[i] for i in pos)


def tail_successor_lists(adj: np.ndarray) -> list[list[int]]:
    """Per-node lists of semi-directed successors (each ``c`` with ``adj[node, c] == TAIL``).

    Each list contains the nodes reachable in one step along an outgoing directed edge or an
    undirected edge. The lists are built once from ``adj`` so :func:`blocks_semi_directed_paths`
    can reuse them across path checks without rescanning the matrix with ``np.nonzero``.
    """
    tail = adj == TAIL
    return [np.nonzero(tail[node])[0].tolist() for node in range(adj.shape[0])]


def blocks_semi_directed_paths(
    succ: Sequence[Sequence[int]], start: int, target: int, blocked: set[int]
) -> bool:
    """True when every semi-directed path ``start ~> target`` passes through ``blocked``.

    ``succ`` is the per-node semi-directed-successor lists from :func:`tail_successor_lists`.
    Equivalent to ``target`` being unreachable from ``start`` once ``blocked`` is removed; ``start``
    and ``target`` themselves are never treated as blocked. ``start == target`` is reachable (the
    empty path), so it returns ``False`` -- the negation the uint64 bitmask BFS returns. No bit
    packing, so it holds for any node count.
    """
    if start == target:
        return False
    barrier = blocked - {start, target}
    visited = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for c in succ[node]:
            if c in barrier or c in visited:
                continue
            if c == target:
                return False
            visited.add(c)
            stack.append(c)
    return True


def apply_insert(adj: np.ndarray, x: int, y: int, T: Iterable[int]) -> None:
    """Insert ``x -> y`` and orient every ``t -- y`` (``t`` in ``T``) as ``t -> y``, in place."""
    adj[x, y] = TAIL
    adj[y, x] = ARROW
    for t in T:
        adj[t, y] = TAIL
        adj[y, t] = ARROW


def apply_delete(adj: np.ndarray, x: int, y: int, H: Iterable[int]) -> None:
    """Delete edge ``x - y`` and orient ``y -> h`` and ``x -> h`` for every ``h`` in ``H``."""
    adj[x, y] = NULL
    adj[y, x] = NULL
    for h in H:
        adj[y, h] = TAIL
        adj[h, y] = ARROW
        adj[x, h] = TAIL
        adj[h, x] = ARROW


# A move that can change more than this share of the nodes rebuilds the whole CPDAG.
_REGION_SHARE = 0.25


def _shared_children(adj: np.ndarray, x: int, y: int) -> list[int]:
    """The nodes that are a child of both ``x`` and ``y``."""
    both = (adj[x] == TAIL) & (adj[:, x] == ARROW) & (adj[y] == TAIL) & (adj[:, y] == ARROW)
    return np.flatnonzero(both).tolist()


def _reform(adj: np.ndarray, touched: list[int]) -> np.ndarray:
    """The CPDAG of ``adj``, a CPDAG changed by one move that touched the nodes ``touched``."""
    out = _cpdag_of_pdag_near(adj, touched, int(_REGION_SHARE * adj.shape[0]))
    return _cpdag_of_pdag(adj) if out is None else out


def insert_cpdag(adj: np.ndarray, x: int, y: int, T: Iterable[int]) -> np.ndarray:
    """Apply ``Insert(x, y, T)`` to the CPDAG ``adj`` and return the CPDAG of the result.

    ``adj`` is rewritten in place and returned unless the whole CPDAG is rebuilt. The result is
    the CPDAG of the extension of :func:`apply_insert`'s graph; only the edges the move can reach
    are derived again (:func:`~andrey.core.orient._cpdag_of_pdag_near`).
    """
    T = tuple(T)
    shared = _shared_children(adj, x, y)
    apply_insert(adj, x, y, T)
    return _reform(adj, [x, y, *T, *shared])


def delete_cpdag(adj: np.ndarray, x: int, y: int, H: Iterable[int]) -> np.ndarray:
    """Apply ``Delete(x, y, H)`` to the CPDAG ``adj`` and return the CPDAG of the result.

    ``adj`` is rewritten in place and returned unless the whole CPDAG is rebuilt; see
    :func:`insert_cpdag`.
    """
    H = tuple(H)
    shared = _shared_children(adj, x, y)
    apply_delete(adj, x, y, H)
    return _reform(adj, [x, y, *H, *shared])
