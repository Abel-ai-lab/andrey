"""PC (Peter-Clark) constraint-based causal discovery, the engine behind :func:`andrey.pc`.

The algorithm, its settings, and its references are described on :func:`andrey.pc`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from itertools import combinations, groupby
from math import comb
from operator import itemgetter
from typing import TYPE_CHECKING

import numpy as np

from andrey.constraint._parallel_skeleton import SkeletonWorkerPool
from andrey.constraint._skeleton import (
    Removal,
    adjacent_pairs,
    first_order_separating,
    marginal_removals,
    prescreen_capable,
    scan_pairs,
    separate_apart,
    unseparated_triples,
)
from andrey.core import ARROW, NULL, TAIL, GraphStructure, backend
from andrey.core.ci import make_indep_test, pins_blas
from andrey.core.knowledge import KnowledgeError, _has_cycle, orient
from andrey.core.orient import meek, to_structure
from andrey.core.warning_policy import PerformanceWarning, warn_once

if TYPE_CHECKING:
    from andrey.core.knowledge import Constraints

_EXPENSIVE_PASS_TESTS = 10_000_000
COLLIDER_ORDERS = ("p_value", "column")


def pc(
    data: np.ndarray,
    *,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    collider_order: str = "p_value",
    categorical: np.ndarray | None = None,
    max_depth: int | None = None,
    constraints: Constraints | None = None,
    _trace: list[dict] | None = None,
) -> GraphStructure:
    """Discover a CPDAG from ``data`` with the PC algorithm and the ``indep_test`` CI test.

    ``data`` is an ``(n_samples, n_features)`` array; ``alpha`` is the independence-test
    significance level in ``(0, 1)`` (a pair is separated when its p-value exceeds ``alpha``);
    ``indep_test`` selects the conditional-independence test from the registry, ``collider_order``
    the order of :func:`_orient_colliders`, and ``categorical``, a boolean mask of the categorical
    columns, reaches a mixed-data test. ``max_depth`` caps the conditioning-set size of the
    skeleton search. With ``constraints``, the skeleton search starts from the pairs they allow,
    never tests a required pair, and gives a pair they keep apart a separating set when one exists
    (:func:`_discover_skeleton`); every edge they decide is directed before the colliders, and
    Meek's rules run with R4, so the result is the class narrowed by the knowledge. A triple over a
    kept-apart pair that no set separates is neither a collider nor a non-collider to any rule
    (:func:`~andrey.constraint._skeleton.unseparated_triples`). Edges they decide that close a
    directed cycle raise ``ValueError`` (:func:`check_knowledge_cycle`).
    Returns the CPDAG as a ``GraphStructure`` of kind ``"cpdag"``.

    The private ``_trace`` sink receives one record per step, in the order the steps run: each
    independence test (``step="test"``, from :func:`_discover_skeleton`), each unshielded triple
    (``step="collider"``, from :func:`_orient_colliders`), each Meek rule firing (``step="meek"``,
    with the fields :func:`~andrey.core.orient.meek` records), and, when colliders of equal rank
    conflict, the edges left undirected (``step="conflict"``, from :func:`_undirect_conflicts`).
    """
    check_collider_order(collider_order)
    X = np.asarray(data, dtype=np.float64)
    skeleton, sepsets, pvalues = _discover_skeleton(
        X,
        alpha,
        indep_test,
        categorical=categorical,
        record_pvalues=collider_order == "p_value",
        warn_expensive=True,
        max_depth=max_depth,
        constraints=constraints,
        _trace=_trace,
    )
    rule4 = constraints is not None
    ambiguous = unseparated_triples(skeleton != NULL, sepsets, constraints)
    if constraints is not None:
        skeleton = orient(skeleton, constraints)
        check_knowledge_cycle((skeleton == TAIL) & (skeleton.T == ARROW))
    pdag = _orient_colliders(skeleton, sepsets, pvalues, order=collider_order, _trace=_trace)
    if _trace is None:
        cpdag = meek(pdag, rule4=rule4, ambiguous=ambiguous)
        return to_structure(_undirect_conflicts(cpdag), kind="cpdag")
    firings: list[dict] = []
    cpdag = meek(pdag, rule4=rule4, ambiguous=ambiguous, _trace=firings)
    _trace.extend(dict(step="meek", **firing) for firing in firings)
    return to_structure(_undirect_conflicts(cpdag, _trace=_trace), kind="cpdag")


def check_collider_order(collider_order: str) -> None:
    """Raise ``ValueError`` unless ``collider_order`` is one of :data:`COLLIDER_ORDERS`."""
    if collider_order not in COLLIDER_ORDERS:
        raise ValueError(f"collider_order must be 'p_value' or 'column', got {collider_order!r}")


def check_knowledge_cycle(paths: np.ndarray, closing: np.ndarray | None = None) -> None:
    """Raise :class:`~andrey.core.knowledge.KnowledgeError` when the edges background knowledge
    directs close a cycle.

    The cycle follows edges ``paths[i, j]`` (``i -> j``) from a node ``y`` to a node ``x`` and
    closes with ``closing[x, y]``; ``closing`` defaults to ``paths``, so any directed cycle of
    ``paths`` counts. PC and CD-NOD pass the edges the knowledge directs. FCI passes its required
    edges as ``paths`` and its arrowheads as ``closing``: a required path from ``y`` to ``x`` makes
    ``y`` an ancestor of ``x``, which an arrowhead at ``y`` on the edge ``x - y`` denies, while a
    cycle of arrowheads alone fits bidirected edges. The message names the cycle by column position.
    """
    closing = paths if closing is None else closing
    if not _has_cycle(paths | closing):
        return
    successors = [np.flatnonzero(row).tolist() for row in paths]
    for y in np.flatnonzero(closing.any(axis=0)).tolist():
        previous = {y: y}
        queue = deque([y])
        while queue:
            x = queue.popleft()
            if closing[x, y]:
                cycle = [x]
                while cycle[-1] != y:
                    cycle.append(previous[cycle[-1]])
                path = " -> ".join(str(v) for v in [*cycle[::-1], y])
                raise KnowledgeError(
                    f"the knowledge directs the edges {path} that the data keep into a cycle; "
                    "forbid or allow one of them"
                )
            for z in successors[x]:
                if z not in previous:
                    previous[z] = x
                    queue.append(z)


def _orient_colliders(
    pdag: np.ndarray,
    sepsets: Mapping[tuple[int, int], Iterable[int] | None],
    pvalues: Mapping[tuple[int, int], float],
    *,
    order: str = "p_value",
    _trace: list[dict] | None = None,
) -> np.ndarray:
    """Orient the unshielded colliders in ``order``, by default most clearly separated pair first.

    ``pdag`` holds the skeleton's endpoint marks, every edge undirected or some already directed
    (CDNOD directs the context's edges first, and background knowledge the edges it decides); the
    result is a new matrix. An unshielded triple ``x - y - z`` (``x`` and ``z`` both adjacent to
    ``y`` but not to each other) is a collider ``x -> y <- z`` iff ``y`` is not in the separating
    set of ``x`` and ``z``; a triple whose endpoints have no recorded separating set, a pair the
    knowledge keeps apart that no conditioning set separates, is skipped. A collider is skipped
    when an edge it would orient already has an arrowhead at ``x`` or ``z``, which it would make
    bidirected; so of two conflicting colliders, the one oriented first is kept, and an edge
    directed before the colliders is never reversed or made bidirected.

    With ``order="p_value"``, the collider's rank is ``pvalues[(x, z)]``, keyed with the lower index
    first: the largest p-value among the conditioning sets that separated ``x`` and ``z``. Colliders
    are oriented from the highest rank to the lowest. Colliders of exactly equal rank are each
    checked against the marks set before their rank, then all set: an edge two of them dispute ends
    bidirected, and :func:`_undirect_conflicts` leaves it undirected after Meek's rules. The result
    depends on the ranks, not on the column order. With ``order="column"``, ``pvalues`` is not read
    (it may be empty): the triples are visited by middle node ``y``, then by endpoint pair
    ``(x, z)``, in ascending column order, each decided against the marks set before it, so no edge
    ends bidirected and the result depends on the column order.

    The private ``_trace`` sink records every unshielded triple in the order decided:
    ``step="collider"``, the ``triple`` ``(x, y, z)``, the pair's ``sepset`` and rank ``p``
    (``None`` with ``order="column"``), whether ``y`` was ``oriented`` as a collider, and the
    ``before``/``after`` endpoint matrices, which are copied only when a trace is requested.
    """
    ranked = order == "p_value"
    pdag = pdag.copy()
    triples = []
    for y in range(pdag.shape[0]):
        nbrs = np.flatnonzero(pdag[y] != NULL).tolist()
        for x, z in combinations(nbrs, 2):
            if pdag[x, z] == NULL and ((x, z) in sepsets or (z, x) in sepsets):
                triples.append((pvalues[(x, z)] if ranked else None, x, y, z))
    if ranked:
        triples.sort(key=itemgetter(0), reverse=True)
        groups = [list(tied) for _, tied in groupby(triples, key=itemgetter(0))]
    else:
        groups = [[triple] for triple in triples]
    for group in groups:
        decided = []
        for p, x, y, z in group:
            sepset = _lookup_sepset(sepsets, x, z)
            collider = y not in sepset and ARROW not in (pdag[x, y], pdag[z, y])
            decided.append((p, x, y, z, sepset, collider))
        for p, x, y, z, sepset, collider in decided:
            before = None if _trace is None else pdag.copy()
            if collider:
                pdag[y, x] = pdag[y, z] = ARROW
            if _trace is not None:
                _trace.append(
                    dict(
                        step="collider",
                        triple=(x, y, z),
                        sepset=tuple(sorted(sepset)),
                        p=p,
                        oriented=collider,
                        before=before,
                        after=pdag.copy(),
                    )
                )
    return pdag


def _undirect_conflicts(pdag: np.ndarray, *, _trace: list[dict] | None = None) -> np.ndarray:
    """Make every bidirected edge, where two colliders of equal rank conflict, undirected.

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
    record_pvalues: bool = True,
    prescreen: bool = True,
    workers: int = 1,
    warn_expensive: bool = False,
    max_depth: int | None = None,
    constraints: Constraints | None = None,
    _trace: list[dict] | None = None,
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, ...]], dict[tuple[int, int], float]]:
    """Learn the undirected skeleton and, for every removed pair, its separating set and p-value.

    Grows the conditioning size from zero, up to ``max_depth`` when one is given, over the complete
    graph, or over the pairs ``constraints`` allow when given; a pair joined by a required edge is
    never tested, so it stays.
    At each size, every currently adjacent pair is tested against subsets of a start-of-size
    neighborhood snapshot (in both directions, as PC does), and edges found independent are
    removed together after the size completes. A removed pair's
    separating set is the union of every conditioning set that made it independent, and its p-value
    the largest among those sets' tests; both are keyed ``(x, y)`` with ``x < y``. ``indep_test``
    selects the CI test (shared by PC and CDNOD, which pass the augmented matrix); ``categorical``
    marks the categorical columns for a mixed-data test. With ``record_pvalues`` false, as for the
    column collider order, which ranks nothing, no p-value is compared or kept and the returned
    p-value dict is empty; the skeleton and the separating sets are the same.

    After the last size, a pair ``constraints`` keep apart that shares a neighbor gets its
    separating set and p-value the same way, from the final neighborhoods
    (:func:`~andrey.constraint._skeleton.separate_apart`), so colliders over it are oriented as
    usual; a pair no conditioning set separates gets neither.

    When ``prescreen`` is set and the test exposes the vectorized closed forms, the size-0 and
    size-1 passes read the correlation matrix directly (all-pairs marginal threshold, then one
    first-order read per surviving pair) in place of the dense scalar scan -- same removals, same
    separating sets, same p-values. With ``workers > 1`` the ``|S| >= 2`` scan fans out across
    worker processes; the reduce is a plain union of per-chunk removals, so the result is identical
    to the serial scan whatever the partition.

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
    keep = None
    if constraints is not None:
        adj[~constraints.support()] = NULL
        keep = constraints.required | constraints.required.T
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {}
    pvalues: dict[tuple[int, int], float] = {}

    can_prescreen = prescreen and prescreen_capable(test)
    seedable = callable(getattr(test, "to_seed", None))
    parallel = workers > 1 and seedable and _trace is None
    pool = SkeletonWorkerPool(test, workers) if parallel else None
    try:
        size = 0
        while True:
            neighbors = [np.flatnonzero(adj[i] != NULL) for i in range(d)]
            max_degree = max((nb.size for nb in neighbors), default=0)
            if size > max_degree - 1 or (max_depth is not None and size > max_depth):
                break
            testable = adj if keep is None else (adj != NULL) & ~keep
            removals: list[Removal] = []
            if can_prescreen and size == 0:
                upper = np.triu(testable != NULL, 1)
                for x, y, p in marginal_removals(test, upper, alpha, _trace, record_pvalues):
                    removals.append((x, y, (), p))
            elif can_prescreen and size == 1:
                for x, y in adjacent_pairs(testable):
                    candidates = sorted(
                        (set(neighbors[x].tolist()) | set(neighbors[y].tolist())) - {x, y}
                    )
                    separating, p = first_order_separating(
                        test, x, y, candidates, alpha, _trace, record_pvalues
                    )
                    if separating:
                        removals.append((x, y, tuple(separating), p))
            else:
                pairs = adjacent_pairs(testable)
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
                    removals = pool.scan(pairs, adj != NULL, size, alpha, record_pvalues)
                else:
                    # Pin BLAS to one thread for a test built on matrix algebra: its p-values then
                    # match the (1-thread) parallel skeleton workers, where it has them, whatever
                    # the host thread count.
                    with backend.one_blas_thread() if pins_blas(test) else nullcontext():
                        removals = scan_pairs(
                            test, pairs, neighbors, size, alpha, _trace, record_pvalues
                        )
            for x, y, separating, p in removals:
                adj[x, y] = NULL
                adj[y, x] = NULL
                sepsets[(x, y)] = separating
                if p is not None:
                    pvalues[(x, y)] = p
            size += 1
    finally:
        if pool is not None:
            pool.close()

    if constraints is not None:
        removals = separate_apart(
            test,
            adj != NULL,
            ~constraints.support(),
            alpha,
            max_depth=max_depth,
            trace=_trace,
            record_pvalues=record_pvalues,
        )
        for x, y, separating, p in removals:
            sepsets[(x, y)] = separating
            if p is not None:
                pvalues[(x, y)] = p
    return adj, sepsets, pvalues


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
