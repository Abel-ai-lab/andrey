"""PC (Peter-Clark) constraint-based causal discovery, the engine behind :func:`andrey.pc`.

The algorithm, its settings, and its references are described on :func:`andrey.pc`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from itertools import combinations
from math import comb

import numpy as np

from andrey.constraint._parallel_skeleton import SkeletonWorkerPool
from andrey.constraint._skeleton import (
    adjacent_pairs,
    first_order_separating,
    marginal_removals,
    prescreen_capable,
    scan_pairs,
)
from andrey.core import ARROW, NULL, TAIL, GraphStructure, backend
from andrey.core.ci import make_indep_test, pins_blas
from andrey.core.orient import meek, to_structure
from andrey.core.warning_policy import PerformanceWarning, warn_once

_EXPENSIVE_PASS_TESTS = 10_000_000


def pc(
    data: np.ndarray,
    *,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    categorical: np.ndarray | None = None,
    _trace: list[dict] | None = None,
) -> GraphStructure:
    """Discover a CPDAG from ``data`` with the PC algorithm and the ``indep_test`` CI test.

    ``data`` is an ``(n_samples, n_features)`` array; ``alpha`` is the independence-test
    significance level in ``(0, 1)`` (a pair is separated when its p-value exceeds ``alpha``);
    ``indep_test`` selects the conditional-independence test from the registry, and
    ``categorical``, a boolean mask of the categorical columns, reaches a mixed-data test. Returns
    the CPDAG as a ``GraphStructure`` of kind ``"cpdag"``.

    The private ``_trace`` sink receives one record per step, in the order the steps run: each
    independence test (``step="test"``, from :func:`_discover_skeleton`), each unshielded triple
    (``step="collider"``, from :func:`_orient_colliders`), each Meek rule firing (``step="meek"``,
    with the fields :func:`~andrey.core.orient.meek` records), and, when colliders conflict, the
    edges left undirected (``step="conflict"``, from :func:`_undirect_conflicts`).
    """
    X = np.asarray(data, dtype=np.float64)
    skeleton, sepsets = _discover_skeleton(
        X, alpha, indep_test, categorical=categorical, warn_expensive=True, _trace=_trace
    )
    pdag = _orient_colliders(skeleton, sepsets, _trace=_trace)
    if _trace is None:
        return to_structure(_undirect_conflicts(meek(pdag)), kind="cpdag")
    firings: list[dict] = []
    cpdag = meek(pdag, _trace=firings)
    _trace.extend(dict(step="meek", **firing) for firing in firings)
    return to_structure(_undirect_conflicts(cpdag, _trace=_trace), kind="cpdag")


def _orient_colliders(
    skeleton: np.ndarray,
    sepsets: Mapping[tuple[int, int], Iterable[int] | None],
    *,
    _trace: list[dict] | None = None,
) -> np.ndarray:
    """Orient every unshielded collider at once; an edge two colliders dispute becomes bidirected.

    Starts from the fully undirected skeleton and visits every unshielded triple ``x - y - z``
    (``x`` and ``z`` both adjacent to ``y`` but not to each other) with ``y`` the middle node,
    ordered by middle node then by endpoint pair ``(x, z)`` with ``x < z`` ascending. ``y`` is a
    collider ``x *-> y <-* z`` iff ``y`` is not in the separating set of ``x`` and ``z``; it then
    sets the arrowhead at ``y`` on both edges and changes no other mark. An edge that receives an
    arrowhead at each end, from colliders at both of its nodes, ends bidirected (``x <-> y``). The
    result is the set of all the colliders' arrowheads, so the visiting order does not change it:
    the list rule of Colombo and Maathuis (2014), Section 4.3.

    The private ``_trace`` sink records every unshielded triple visited: ``step="collider"``, the
    ``triple`` ``(x, y, z)``, the pair's ``sepset``, whether ``y`` was ``oriented`` as a collider,
    and the ``before``/``after`` endpoint matrices, which are copied only when a trace is requested.
    """
    n = skeleton.shape[0]
    pdag = np.where(skeleton != NULL, np.int8(TAIL), np.int8(NULL))
    np.fill_diagonal(pdag, np.int8(NULL))
    for y in range(n):
        nbrs = np.nonzero(skeleton[y] != NULL)[0].tolist()
        for a, b in combinations(range(len(nbrs)), 2):
            x, z = nbrs[a], nbrs[b]
            if skeleton[x, z] != NULL:
                continue
            sepset = _lookup_sepset(sepsets, x, z)
            collider = y not in sepset
            if _trace is not None:
                before = pdag.copy()
            if collider:
                pdag[y, x] = pdag[y, z] = ARROW
            if _trace is not None:
                _trace.append(
                    dict(
                        step="collider",
                        triple=(x, y, z),
                        sepset=tuple(sorted(sepset)),
                        oriented=collider,
                        before=before,
                        after=pdag.copy(),
                    )
                )
    return pdag


def _undirect_conflicts(pdag: np.ndarray, *, _trace: list[dict] | None = None) -> np.ndarray:
    """Make every bidirected edge, where two colliders conflict, undirected.

    It runs after Meek's rules, which neither use nor orient a bidirected edge, so a conflicting
    edge stays undirected in the CPDAG. Returns ``pdag`` itself when no edge is bidirected,
    otherwise a copy. The private ``_trace`` sink receives one record when an edge changes:
    ``step="conflict"``, the ``edges`` ``(x, y)`` with ``x < y``, and the ``before``/``after``
    endpoint matrices.
    """
    bidirected = (pdag == ARROW) & (pdag.T == ARROW)
    if not bidirected.any():
        return pdag
    out = pdag.copy()
    out[bidirected] = TAIL
    if _trace is not None:
        edges = [tuple(e) for e in np.argwhere(np.triu(bidirected)).tolist()]
        _trace.append(dict(step="conflict", edges=edges, before=pdag.copy(), after=out.copy()))
    return out


def _lookup_sepset(
    sepsets: Mapping[tuple[int, int], Iterable[int] | None], x: int, z: int
) -> frozenset[int]:
    """The separating set for the pair, in either key order; empty when absent or ``None``."""
    value = sepsets.get((x, z))
    if value is None:
        value = sepsets.get((z, x))
    if value is None:
        return frozenset()
    return frozenset(int(v) for v in value)


def _discover_skeleton(
    X: np.ndarray,
    alpha: float,
    indep_test: str = "fisherz",
    *,
    categorical: np.ndarray | None = None,
    prescreen: bool = True,
    workers: int = 1,
    warn_expensive: bool = False,
    _trace: list[dict] | None = None,
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, ...]]]:
    """Learn the undirected skeleton and the separating set of every removed pair.

    Grows the conditioning size from zero; at each size, every currently adjacent pair is tested
    against subsets of a start-of-size neighborhood snapshot (in both directions, as PC does), and
    edges found independent are removed together after the size completes. A removed pair's
    separating set is the union of every conditioning set that made it independent. ``indep_test``
    selects the CI test (shared by PC and CDNOD, which pass the augmented matrix); ``categorical``
    marks the categorical columns for a mixed-data test.

    When ``prescreen`` is set and the test exposes the vectorized closed forms, the size-0 and
    size-1 passes read the correlation matrix directly (all-pairs marginal threshold, then one
    first-order read per surviving pair) in place of the dense scalar scan -- same removals, same
    separating sets. With ``workers > 1`` the ``|S| >= 2`` scan fans out across worker processes;
    the reduce is a plain union of per-chunk removals, so the skeleton is identical to the serial
    scan whatever the partition.

    ``warn_expensive`` emits a :class:`~andrey.PerformanceWarning` before each conditioning-size
    pass requiring at least ten million CI tests. The search still evaluates every conditioning set.

    The private ``_trace`` sink records every test in the order it runs: ``step="test"``,
    ``depth``, the pair ``x < y``, the conditioning set ``S``, the p-value ``p``, and ``removed``
    (``p > alpha``: the test separates the pair, so the edge is removed when its depth ends). A
    trace needs every p-value in this process, so tracing scans serially.
    """
    d = X.shape[1]
    test = make_indep_test(indep_test, X, categorical=categorical)
    adj = np.full((d, d), TAIL, dtype=np.int8)
    np.fill_diagonal(adj, NULL)
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {}

    can_prescreen = prescreen and prescreen_capable(test)
    seedable = callable(getattr(test, "to_seed", None))
    parallel = workers > 1 and seedable and _trace is None
    pool = SkeletonWorkerPool(test, workers) if parallel else None
    try:
        size = 0
        while True:
            neighbors = [np.flatnonzero(adj[i] != NULL) for i in range(d)]
            max_degree = max((nb.size for nb in neighbors), default=0)
            if size > max_degree - 1:
                break
            remove: list[tuple[int, int]] = []
            if can_prescreen and size == 0:
                for x, y in marginal_removals(test, np.triu(adj != NULL, 1), alpha, _trace):
                    remove.append((x, y))
                    sepsets[(x, y)] = ()
            elif can_prescreen and size == 1:
                for x, y in adjacent_pairs(adj):
                    candidates = sorted(
                        (set(neighbors[x].tolist()) | set(neighbors[y].tolist())) - {x, y}
                    )
                    separating = first_order_separating(test, x, y, candidates, alpha, _trace)
                    if separating:
                        remove.append((x, y))
                        sepsets[(x, y)] = tuple(separating)
            else:
                pairs = adjacent_pairs(adj)
                if warn_expensive and size >= 2 and _expensive_pass(pairs, neighbors, size):
                    warn_once(
                        f"PC conditioning size {size} requires at least "
                        f"{_EXPENSIVE_PASS_TESTS:,} CI tests "
                        f"(d={d}, largest remaining neighborhood={max_degree}). "
                        "Runtime grows combinatorially with neighborhood size; "
                        "the full search remains enabled.",
                        PerformanceWarning,
                    )
                if pool is not None:
                    removals = pool.scan(pairs, adj != NULL, size, alpha)
                else:
                    # Pin BLAS to one thread for a test built on matrix algebra: its p-values then
                    # match the (1-thread) parallel skeleton workers, where it has them, whatever
                    # the host thread count.
                    with backend.one_blas_thread() if pins_blas(test) else nullcontext():
                        removals = scan_pairs(test, pairs, neighbors, size, alpha, _trace)
                for x, y, separating in removals:
                    remove.append((x, y))
                    sepsets[(x, y)] = separating
            for x, y in remove:
                adj[x, y] = NULL
                adj[y, x] = NULL
            size += 1
    finally:
        if pool is not None:
            pool.close()

    return adj, sepsets


def _expensive_pass(pairs: list[tuple[int, int]], neighbors: list[np.ndarray], size: int) -> bool:
    """Whether the pass reaches the warning threshold, counting shared subsets once."""
    tests = 0
    for x, y in pairs:
        shared = len(set(neighbors[x]) & set(neighbors[y]))
        tests += (
            comb(len(neighbors[x]) - 1, size)
            + comb(len(neighbors[y]) - 1, size)
            - comb(shared, size)
        )
        if tests >= _EXPENSIVE_PASS_TESTS:
            return True
    return False
