"""FCI (Fast Causal Inference) causal discovery, the engine behind :func:`andrey.fci`.

The algorithm, its settings, and its references are described on :func:`andrey.fci`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import nullcontext
from functools import cache
from itertools import combinations, islice, permutations
from typing import TYPE_CHECKING

import numpy as np

from andrey.constraint._parallel_skeleton import skeleton_pool
from andrey.constraint._skeleton import (
    adjacent_pairs,
    first_order_removals,
    marginal_removals,
    prescreen_capable,
    scan_pairs,
    separate_apart,
    unseparated_triples,
)
from andrey.constraint.pc import check_knowledge_cycle
from andrey.core import ARROW, CIRCLE, NULL, TAIL, GraphStructure, backend
from andrey.core._bitset import row_masks, semidirected_reach_set
from andrey.core.ci import CITest, make_indep_test, pins_blas
from andrey.search.hc import tie_ranks

if TYPE_CHECKING:
    from andrey.core.knowledge import Constraints

# Bound discriminating and possible-d-sep paths by this finite cap.
_UNBOUNDED_PATH = 1000
_R4_PARTIAL_PATHS = 10_000
_R5_PARTIAL_PATHS = 10_000


def fci(
    data: np.ndarray,
    *,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    collider_rule: str = "sepsets",
    categorical: np.ndarray | None = None,
    possible_dsep: str = "order_free",
    max_depth: int | None = None,
    constraints: Constraints | None = None,
    in_column_order: bool = False,
    workers: int | None = None,
) -> GraphStructure:
    """Discover a PAG from ``data`` with the FCI algorithm and the ``indep_test`` CI test.

    ``data`` is an ``(n_samples, n_features)`` array; ``alpha`` is the independence-test
    significance level in ``(0, 1)``. Skeleton pruning requires a p-value above ``alpha``;
    the optional majority collider rule also counts equality as a separating vote.
    ``categorical``, a boolean mask of the categorical columns, reaches a mixed-data test.
    ``possible_dsep`` (``"order_free"`` or ``"fast"``) selects the order the possible-d-sep step
    visits the nodes in: the :func:`~andrey.search.hc.tie_ranks` order or the column order.
    ``in_column_order`` applies the orientation rules one firing at a time in column order instead
    of in order-free rounds (:func:`_fci_orient`).
    ``max_depth`` caps the size of every conditioning set a test uses: the adjacency search, the
    possible-d-sep subsets, the majority votes, and the discriminating-path tests. With
    ``constraints``, the adjacency search starts from the pairs they allow, no step tests or removes
    a required pair, a pair they make non-adjacent gets a separating set when the search finds one
    (:func:`_fast_adjacency_search`), and :func:`_knowledge_marks` sets their endpoints before the
    orientation rules, which change circles only. A triple over a non-adjacent pair with no
    separating set is neither a collider nor a non-collider to any rule
    (:func:`~andrey.constraint._skeleton.unseparated_triples`).
    An arrowhead the knowledge puts at ``x`` on an edge ``x - y`` while its required edges lead from
    ``x`` to ``y`` raises ``ValueError`` (:func:`~andrey.constraint.pc.check_knowledge_cycle`).
    ``workers`` worker processes share the adjacency search's larger conditioning-size passes
    (:func:`_fast_adjacency_search`); ``None`` reads ``andrey.config.num_workers``.
    Returns the PAG as a ``GraphStructure`` of kind ``"pag"`` (circle marks and all).
    """
    X = np.asarray(data, dtype=np.float64)

    _validate_collider_rule(collider_rule)
    _validate_possible_dsep(possible_dsep)
    test = make_indep_test(indep_test, X, categorical=categorical)
    allowed = keep = apart = None
    if constraints is not None:
        allowed = constraints.support()
        keep = constraints.required | constraints.required.T
        apart = ~allowed
    support, sepsets = _fast_adjacency_search(
        X,
        test,
        alpha,
        workers=backend.worker_count(workers),
        support=allowed,
        keep=keep,
        apart=apart,
        max_depth=max_depth,
    )

    marks = _circle_pag(support)
    _knowledge_marks(marks, constraints)
    _orient_colliders(marks, sepsets)
    if possible_dsep == "fast":
        _remove_by_possible_dsep(marks, test, alpha, sepsets, keep=keep, max_depth=max_depth)
    else:
        by_rank, ranked_test = _test_in_rank_order(X, indep_test, categorical)
        _remove_by_possible_dsep(
            marks, ranked_test, alpha, sepsets, by_rank, keep=keep, max_depth=max_depth
        )
    marks = _circle_pag(marks != NULL)
    _knowledge_marks(marks, constraints)
    if constraints is not None:
        check_knowledge_cycle((marks == TAIL) & (marks.T == ARROW), closing=(marks == ARROW).T)
    ambiguous = unseparated_triples(marks != NULL, sepsets, constraints)
    if collider_rule == "majority":
        ambiguous |= _majority_sepsets(
            marks, test, alpha, sepsets, max_depth=max_depth, skip=ambiguous
        )
    _orient_colliders(marks, sepsets, ambiguous)
    _fci_orient(
        marks, test, alpha, sepsets, ambiguous, max_depth=max_depth, in_column_order=in_column_order
    )

    return GraphStructure.from_numpy(marks, kind="pag")


# ---- skeleton (fast adjacency search) -----------------------------------------------------------


def _fast_adjacency_search(
    X: np.ndarray,
    test: CITest,
    alpha: float,
    *,
    prescreen: bool = True,
    workers: int = 1,
    support: np.ndarray | None = None,
    keep: np.ndarray | None = None,
    apart: np.ndarray | None = None,
    max_depth: int | None = None,
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, ...]]]:
    """Learn the undirected skeleton and every removed pair's separating set (stable, FAS-style).

    Grows the conditioning size from zero, up to ``max_depth`` when one is given, over a complete
    graph, or over the boolean adjacency ``support`` when one is given; at each size, every
    still-adjacent pair ``(x, y)`` is tested against subsets of a start-of-size snapshot of *either*
    endpoint's neighborhood, and pairs found independent are removed together after the size
    completes. A pair marked in the boolean ``keep`` (the required edges, both ways) is never
    tested, so it stays; it still conditions the tests of other pairs.
    A removed pair's separating set is the union of every conditioning set that made it independent.
    After the last size, a pair marked in the boolean ``apart`` (the pairs background knowledge
    makes non-adjacent, outside ``support``) that shares a neighbor gets its separating set the same
    way, from the final neighborhoods (:func:`~andrey.constraint._skeleton.separate_apart`); a pair
    no tested conditioning set separates gets none.

    Either endpoint supplies the conditioning candidates, so a separating set is reachable wherever
    it lives and the skeleton does not depend on how the caller ordered the columns.

    When ``prescreen`` is set and the test exposes the vectorized marginal and first-order paths,
    the size-0 and size-1 passes take them instead of the dense scalar scan -- same removals and
    separating sets. With ``workers > 1`` and a FisherZ or StudentT test, an ``|S| >= 2`` pass of
    at least :data:`~andrey.constraint._parallel_skeleton.MIN_POOL_QUERIES` tests, or the
    ``ANDREY_SKELETON_PARALLEL_MIN_WORK`` that overrides it, fans out across worker processes; the
    per-chunk removals are concatenated in pair order, so the skeleton, a raised error included,
    stays identical to the serial scan. With ``workers > 1`` and an RCoT test, a pass whose tests
    times rows reach :data:`~andrey.constraint._parallel_skeleton.MIN_POOL_RCOT_WORK`, or the
    ``ANDREY_RCOT_PARALLEL_MIN_WORK`` that overrides it, runs each conditioning set's queries in one
    worker process, with the serial scan's p-values bit for bit.
    """
    d = X.shape[1]
    support = np.ones((d, d), dtype=bool) if support is None else np.array(support, dtype=bool)
    np.fill_diagonal(support, False)
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {}

    can_prescreen = prescreen and prescreen_capable(test)
    pool = skeleton_pool(test, workers)
    try:
        size = -1
        while True:
            degrees = support.sum(axis=1)
            if int(degrees.max(initial=0)) - 1 <= size:
                break
            size += 1
            if max_depth is not None and size > max_depth:
                break
            neighbors = [np.flatnonzero(support[i]) for i in range(d)]
            testable = support if keep is None else support & ~keep
            remove: list[tuple[int, int]] = []
            if can_prescreen and size == 0:
                upper = np.triu(testable, 1)
                for x, y, _ in marginal_removals(test, upper, alpha, record_pvalues=False):
                    remove.append((x, y))
                    sepsets[(x, y)] = ()
                    sepsets[(y, x)] = ()  # skeleton separations are recorded both ways
            elif can_prescreen and size == 1:
                pairs = adjacent_pairs(testable)
                for x, y, separating, _ in first_order_removals(
                    test, pairs, neighbors, alpha, record_pvalues=False
                ):
                    remove.append((x, y))
                    sepsets[(x, y)] = separating
                    sepsets[(y, x)] = separating
            else:
                pairs = adjacent_pairs(testable)
                pooled = (
                    None
                    if pool is None
                    else pool.scan(
                        pairs, neighbors, support.copy(), size, alpha, record_pvalues=False
                    )
                )
                if pooled is not None:
                    removals = pooled
                else:
                    # Pin BLAS to one thread for a test built on matrix algebra: its p-values then
                    # match the (1-thread) parallel skeleton workers, where it has them, whatever
                    # the host thread count.
                    with backend.one_blas_thread() if pins_blas(test) else nullcontext():
                        removals = scan_pairs(
                            test, pairs, neighbors, size, alpha, record_pvalues=False
                        )
                for x, y, separating, _ in removals:
                    remove.append((x, y))
                    sepsets[(x, y)] = separating
                    sepsets[(y, x)] = separating
            for x, y in remove:
                support[x, y] = False
                support[y, x] = False
    finally:
        if pool is not None:
            pool.close()

    if apart is not None:
        for x, y, separating, _ in separate_apart(
            test, support, apart, alpha, max_depth=max_depth, record_pvalues=False
        ):
            sepsets[(x, y)] = separating
            sepsets[(y, x)] = separating
    return support, sepsets


# ---- mark-matrix helpers ------------------------------------------------------------------------
# ``marks[i, j]`` is the mark at node ``i`` on edge ``i-j``, as in :mod:`andrey.core.structure`.


def _circle_pag(support: np.ndarray) -> np.ndarray:
    """An all-circle PAG over the given adjacency support (every present edge becomes ``o-o``)."""
    marks = np.where(support != NULL, np.int8(CIRCLE), np.int8(NULL))
    np.fill_diagonal(marks, np.int8(NULL))
    return marks


def _knowledge_marks(marks: np.ndarray, constraints: Constraints | None) -> None:
    """Set the endpoints ``constraints`` decide on the adjacent pairs, in place.

    A forbidden ``i -> j`` whose reverse is not forbidden puts an arrowhead at ``i`` (the MAG has no
    ``i -> j``), and a required ``i -> j`` is ``i -> j``: a tail at ``i`` and, as its reverse is
    forbidden, an arrowhead at ``j``. ``None`` leaves ``marks`` as they are.
    """
    if constraints is None:
        return
    adjacent = marks != NULL
    marks[adjacent & constraints.forbidden & ~constraints.forbidden.T] = ARROW
    marks[adjacent & constraints.required] = TAIL


def _adjacent(marks: np.ndarray, a: int, b: int) -> bool:
    return bool(marks[a, b] != NULL)


def _is_def_collider(marks: np.ndarray, a: int, b: int, c: int) -> bool:
    """True when ``a *-> b <-* c`` (arrowheads at ``b`` from both ``a`` and ``c``)."""
    return bool(marks[b, a] == ARROW and marks[b, c] == ARROW)


def _arrow_allowed(marks: np.ndarray, a: int, b: int) -> bool:
    """Whether an arrowhead may be placed at ``b`` on edge ``a-b`` (no background knowledge).

    The endpoint at ``b`` must currently be an arrowhead or a circle; a tail is never overwritten.
    """
    return bool(marks[b, a] in (ARROW, CIRCLE))


def _is_parent_of(marks: np.ndarray, a: int, b: int) -> bool:
    """True when ``a -> b`` is a fully directed edge (tail at ``a``, arrowhead at ``b``)."""
    return bool(marks[b, a] == ARROW and marks[a, b] == TAIL)


def _parents(marks: np.ndarray, b: int) -> set[int]:
    """The directed parents of ``b`` (nodes ``p`` with ``p -> b``)."""
    return {p for p in range(marks.shape[0]) if _is_parent_of(marks, p, b)}


def _neighbor_lists(marks: np.ndarray) -> list[list[int]]:
    """Each node's ascending list of the nodes sharing an edge with it, read from ``marks``."""
    return [np.flatnonzero(row).tolist() for row in marks != NULL]


def _into(marks: np.ndarray, b: int, mark: int) -> list[int]:
    """Ascending nodes ``a`` whose endpoint at ``b`` (on edge ``a-b``) equals ``mark``."""
    return [a for a in range(marks.shape[0]) if marks[b, a] == mark]


def _sepset(
    sepsets: Mapping[tuple[int, int], tuple[int, ...]], a: int, c: int
) -> frozenset[int] | None:
    """Return the separating set for ``(a, c)`` from either key order.

    An empty tuple is a recorded set. ``None`` means neither key order is present.
    """
    value = sepsets.get((a, c))
    if value is None:
        value = sepsets.get((c, a))
    if value is None:
        return None
    return frozenset(int(v) for v in value)


def _largest_size(candidates: int, max_depth: int | None) -> int:
    """The largest conditioning-set size drawn from ``candidates`` nodes under ``max_depth``."""
    return candidates if max_depth is None else min(candidates, max_depth)


def _fits(cond: list[int], max_depth: int | None) -> bool:
    """Whether ``cond`` has at most ``max_depth`` members (always, when ``max_depth`` is None)."""
    return max_depth is None or len(cond) <= max_depth


# ---- R0: unshielded colliders -------------------------------------------------------------------


def _validate_collider_rule(collider_rule: str) -> None:
    if collider_rule not in ("sepsets", "majority"):
        raise ValueError(f"collider_rule must be 'sepsets' or 'majority', got {collider_rule!r}")


def _validate_possible_dsep(possible_dsep: str) -> None:
    if possible_dsep not in ("order_free", "fast"):
        raise ValueError(f"possible_dsep must be 'order_free' or 'fast', got {possible_dsep!r}")


def _triple(a: int, b: int, c: int) -> tuple[int, int, int]:
    """Canonical key for a triple with middle node ``b``."""
    return (a, b, c) if a < c else (c, b, a)


# Conditioning sets per test call in the majority vote. The vote needs only two counts, so a
# triple's memory is bounded by one chunk however large the endpoint degrees are.
_MAJORITY_CHUNK = 64


def _majority_sepsets(
    marks: np.ndarray,
    test: CITest,
    alpha: float,
    sepsets: dict[tuple[int, int], tuple[int, ...]],
    *,
    max_depth: int | None = None,
    skip: frozenset[tuple[int, int, int]] = frozenset(),
) -> frozenset[tuple[int, int, int]]:
    """Rewrite separating sets by majority vote on the final skeleton; return ambiguous triples.

    Every subset of each endpoint's adjacency votes separately, including the empty set and
    subsets shared by both endpoints. Independence is ``p >= alpha``. The recorded sets do not
    vote. A missing key direction inherits the reverse entry before membership is rewritten.
    Fewer than half containing the middle node (or no separating subset) makes a collider;
    more than half makes a noncollider. An exact half leaves both recorded sets unchanged.
    Cost grows exponentially with endpoint degree; the subsets are capped at ``max_depth`` members
    when one is given, and have no size cap otherwise. A triple in ``skip`` holds no vote and keeps
    its recorded sets. Subsets are tested in chunks of ``_MAJORITY_CHUNK``, so memory per triple
    stays bounded.
    """
    neighbors = _neighbor_lists(marks)
    ambiguous = set()
    batched = getattr(test, "batched_call", None)
    for b, adjacent in enumerate(neighbors):
        for a, c in combinations(adjacent, 2):
            if _adjacent(marks, a, c) or _triple(a, b, c) in skip:
                continue
            subsets = (
                subset
                for endpoint in (a, c)
                for size in range(_largest_size(len(neighbors[endpoint]), max_depth) + 1)
                for subset in combinations(neighbors[endpoint], size)
            )
            separating = containing = 0
            while chunk := list(islice(subsets, _MAJORITY_CHUNK)):
                pvalues = (
                    batched(a, c, chunk)
                    if callable(batched)
                    else [test(a, c, subset) for subset in chunk]
                )
                for subset, p in zip(chunk, pvalues):
                    if p >= alpha:
                        separating += 1
                        containing += b in subset
            if separating and 2 * containing == separating:
                ambiguous.add(_triple(a, b, c))
                continue
            noncollider = 2 * containing > separating
            for pair in ((a, c), (c, a)):
                recorded = sepsets.get(pair)
                if recorded is None:
                    recorded = sepsets.get(pair[::-1], ())
                if noncollider:
                    sepsets[pair] = recorded if b in recorded else (*recorded, b)
                else:
                    sepsets[pair] = tuple(node for node in recorded if node != b)
    return frozenset(ambiguous)


def _orient_colliders(
    marks: np.ndarray,
    sepsets: Mapping[tuple[int, int], tuple[int, ...]],
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
) -> None:
    """Orient every unshielded collider ``a *-> b <-* c`` from the separating sets.

    For each node ``b`` and each non-adjacent pair ``(a, c)`` of its neighbors (visited in
    ascending order), ``b`` is a collider iff a separating set for ``a`` and ``c`` is recorded and
    does not contain ``b``; the endpoint at ``b`` on both edges is then set to an arrowhead. An
    already-directed tail at ``b`` blocks the orientation.
    """
    for b, adj in enumerate(_neighbor_lists(marks)):
        for i in range(len(adj)):
            for k in range(i + 1, len(adj)):
                a, c = adj[i], adj[k]
                if _adjacent(marks, a, c) or _is_def_collider(marks, a, b, c):
                    continue
                if _triple(a, b, c) in ambiguous:
                    continue
                sep = _sepset(sepsets, a, c)
                if sep is None or b in sep:
                    continue
                if not _arrow_allowed(marks, a, b) or not _arrow_allowed(marks, c, b):
                    continue
                marks[b, a] = ARROW
                marks[b, c] = ARROW


# ---- possible-d-separation refinement -----------------------------------------------------------


def _semidirected_reach(marks: np.ndarray) -> Callable[[int], int]:
    """``reach(src)``: the bitmask of every node ``dst`` with a semidirected path
    ``src o-> ... o-> dst`` of one or more steps, searched once per ``src``.

    A step ``u -> v`` is allowed when the endpoint at ``u`` is a tail or a circle, so ``src`` is in
    its own set only when such a path returns to it.
    """
    step = (marks == TAIL) | (marks == CIRCLE)
    succ = row_masks(step)
    pred = row_masks(step.T)

    @cache
    def reach(src: int) -> int:
        seen = semidirected_reach_set(succ, src, 0)
        # The search counts src as reached; keep it only when a reached node steps back into it.
        return seen if seen & pred[src] else seen & ~(1 << src)

    return reach


def _exists_possible_parent_path(
    previous: Mapping[int, set[int]], node: int, x: int, reach: Callable[[int], int]
) -> bool:
    """The possible-d-sep reachability check over the ``previous`` predecessor sets, with the
    semidirected paths of :func:`_semidirected_reach`."""
    if node == x:
        return True
    preds = previous.get(node)
    if preds is None:
        return False
    for r in preds:
        if r == node or r == x:
            continue
        if reach(r) & (1 << x | 1 << node):
            return True
    return False


def _possible_dsep(
    marks: np.ndarray,
    x: int,
    y: int,
    neighbors: list[list[int]],
    reach: Callable[[int], int],
) -> list[int]:
    """Possible-d-separating nodes for the pair ``(x, y)`` (descending), from a reachability BFS.

    ``neighbors`` and ``reach`` are the :func:`_neighbor_lists` and :func:`_semidirected_reach` of
    ``marks``.

    Grows a frontier of node pairs from ``x``, extending through definite colliders and adjacent
    triples, and collects every intermediate node reachable by a possible-parent path. The frontier
    holds ``(a, c)`` pairs rather than the last edge of each path, so the set can be smaller than
    Possible-D-Sep as Spirtes, Glymour, and Scheines (2000) define it; :func:`andrey.fci` gives the
    reasons for keeping it. A node's possible-parent check runs when the frontier reaches it, over
    the predecessors recorded so far, so the set depends on the order of the node indices.
    """
    dsep: set[int] = set()
    queue: deque[tuple[int, int]] = deque()
    visited: set[tuple[int, int]] = set()
    previous: dict[int, set[int]] = {}
    first: tuple[int, int] | None = None
    distance = 0

    for b in neighbors[x]:
        if b == y:
            continue
        edge = (x, b)
        if first is None:
            first = edge
        queue.append(edge)
        visited.add(edge)
        previous.setdefault(x, set()).add(b)
        dsep.add(b)

    while queue:
        t = queue.popleft()
        if first == t:
            first = None
            distance += 1
            if distance > _UNBOUNDED_PATH:
                break
        a, b = t
        if _exists_possible_parent_path(previous, b, x, reach):
            dsep.add(b)
        for c in neighbors[b]:
            if c in (a, x, y):
                continue
            previous.setdefault(c, set()).add(b)
            if _is_def_collider(marks, a, b, c) or _adjacent(marks, a, c):
                u = (a, c)
                if u in visited:
                    continue
                visited.add(u)
                queue.append(u)
                if first is None:
                    first = u

    dsep.discard(x)
    dsep.discard(y)
    # Sort by descending labels "X{i+1}". Lexicographic order differs from numeric order
    # at two digits; this order decides which separating subset is found and recorded first.
    return sorted(dsep, key=lambda v: f"X{v + 1}", reverse=True)


def _iter_dsep_choices(
    dsep: list[int], adj_a: set[int], adj_b: set[int], max_depth: int | None = None
) -> Iterator[tuple[int, ...]]:
    """Every non-empty subset of ``dsep``, of at most ``max_depth`` members when one is given, that
    lies inside neither endpoint's adjacency.

    Subsets come by increasing size, each size in ``combinations`` order, one at a time. The
    adjacency search already tested the empty set and every subset of either endpoint's
    neighbors: those neighborhoods only shrink, so each such subset was a candidate at its size
    and left the pair dependent. Testing it again cannot remove the edge.
    """

    for size in range(1, _largest_size(len(dsep), max_depth) + 1):
        yield from (c for c in combinations(dsep, size) if _outside(c, adj_a, adj_b))


def _outside(values: tuple[int, ...], adj_a: set[int], adj_b: set[int]) -> bool:
    """``values`` lie inside neither endpoint's adjacency."""
    return not (all(v in adj_a for v in values) or all(v in adj_b for v in values))


def _try_remove_by_dsep(
    marks: np.ndarray,
    x: int,
    y: int,
    dsep: list[int],
    adj_a: set[int],
    adj_b: set[int],
    test: CITest,
    alpha: float,
    sepsets: dict[tuple[int, int], tuple[int, ...]],
    max_depth: int | None = None,
) -> bool:
    """Remove edge ``x-y`` if some subset of ``dsep``, of at most ``max_depth`` members when one is
    given, makes ``x`` and ``y`` independent."""
    for cond in _iter_dsep_choices(dsep, adj_a, adj_b, max_depth):
        if test(x, y, cond) > alpha:
            marks[x, y] = NULL
            marks[y, x] = NULL
            sepsets[(x, y)] = tuple(int(v) for v in cond)
            return True
    return False


def _edge_order(marks: np.ndarray, i: int, j: int) -> tuple[int, int]:
    """Return the endpoint order used by the possible-d-sep search.

    A left-pointing edge uses ``(j, i)``. Every other edge uses ``(i, j)``, so its search order
    follows the node indices.
    """
    if marks[i, j] == ARROW and marks[j, i] in (TAIL, CIRCLE):
        return j, i
    return i, j


def _remove_by_possible_dsep(
    marks: np.ndarray,
    test: CITest,
    alpha: float,
    sepsets: dict[tuple[int, int], tuple[int, ...]],
    by_rank: np.ndarray | None = None,
    *,
    keep: np.ndarray | None = None,
    max_depth: int | None = None,
) -> None:
    """Prune edges the plain skeleton kept but a possible-d-sep conditioning set separates.

    Visits the pairs one at a time and searches each endpoint's possible-d-sep set for a
    conditioning subset outside either endpoint's adjacency, of at most ``max_depth`` members when
    one is given; the first subset found removes the pair at once, so each removal changes the sets
    of the pairs after it, and is stored under the endpoint order returned by ``_edge_order``. A
    pair marked in the boolean ``keep`` (the required edges, both ways) is never tested. The pairs,
    their endpoints, and the nodes of each set are visited in column order. ``by_rank``, when given,
    is the column order of ``test``'s data, as :func:`_test_in_rank_order` returns them: the step
    then runs on the nodes in that order, with ``keep`` permuted the same way, and maps its removals
    and separating sets back to the column labels once.
    """
    if by_rank is None:
        _remove_in_node_order(marks, test, alpha, sepsets, keep, max_depth)
        return
    ranked = marks[np.ix_(by_rank, by_rank)]
    ranked_keep = None if keep is None else keep[np.ix_(by_rank, by_rank)]
    found: dict[tuple[int, int], tuple[int, ...]] = {}
    _remove_in_node_order(ranked, test, alpha, found, ranked_keep, max_depth)
    position = np.argsort(by_rank)
    marks[:] = ranked[np.ix_(position, position)]
    label = by_rank.tolist()
    for (x, y), cond in found.items():
        sepsets[(label[x], label[y])] = tuple(label[v] for v in cond)


def _test_in_rank_order(
    X: np.ndarray, indep_test: str, categorical: np.ndarray | None
) -> tuple[np.ndarray, CITest]:
    """The column order that sorts ``X`` by :func:`~andrey.search.hc.tie_ranks`, and the
    ``indep_test`` test over ``X``'s columns in that order.

    Each rank moves with its column, so every column order of ``X`` gives the same test.
    """
    by_rank = np.argsort(tie_ranks(X, categorical))
    mask = None if categorical is None else np.asarray(categorical)[by_rank]
    return by_rank, make_indep_test(
        indep_test, np.ascontiguousarray(X[:, by_rank]), categorical=mask
    )


def _remove_in_node_order(
    marks: np.ndarray,
    test: CITest,
    alpha: float,
    sepsets: dict[tuple[int, int], tuple[int, ...]],
    keep: np.ndarray | None,
    max_depth: int | None,
) -> None:
    """:func:`_remove_by_possible_dsep` in the order of the node indices."""
    n = marks.shape[0]
    edges = [_edge_order(marks, i, j) for i in range(n) for j in range(i + 1, n) if marks[i, j]]
    # The lists and the reach read the marks, which only a removal changes.
    neighbors, reach = _neighbor_lists(marks), _semidirected_reach(marks)
    for na, nb in edges:
        if not _adjacent(marks, na, nb) or (keep is not None and keep[na, nb]):
            continue
        adj_a, adj_b = set(neighbors[na]), set(neighbors[nb])
        for a, b in ((na, nb), (nb, na)):
            dsep = _possible_dsep(marks, a, b, neighbors, reach)
            if _try_remove_by_dsep(
                marks, na, nb, dsep, adj_a, adj_b, test, alpha, sepsets, max_depth
            ):
                neighbors, reach = _neighbor_lists(marks), _semidirected_reach(marks)
                break


# ---- FCI orientation rules ----------------------------------------------------------------------


class _Implied:
    """The endpoint marks the orientation rules imply in one round, without changing ``marks``.

    A rule records ``out[i, j] = mark``, the mark at ``i`` on edge ``i-j``, as it would set it on
    the mark matrix itself; :meth:`apply` then sets the marks.
    """

    def __init__(self) -> None:
        self.marks: dict[tuple[int, int], set[int]] = {}

    def __setitem__(self, endpoint: tuple[int, int], mark: int) -> None:
        self.marks.setdefault(endpoint, set()).add(int(mark))

    def apply(self, marks: np.ndarray) -> bool:
        """Set each circle implied one mark alone to that mark; return whether any changed."""
        changed = False
        for endpoint, implied in self.marks.items():
            if marks[endpoint] == CIRCLE and len(implied) == 1:
                marks[endpoint] = implied.pop()
                changed = True
        return changed


def _fci_orient(
    marks: np.ndarray,
    test: CITest,
    alpha: float,
    sepsets: Mapping[tuple[int, int], tuple[int, ...]],
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
    *,
    max_depth: int | None = None,
    in_column_order: bool = False,
) -> None:
    """Apply the FCI orientation rules R1-R10 to a fixed point, in place over ``marks``.

    Circles become tails or arrowheads only where the current marks force it; no rule changes a
    tail or an arrowhead, so one present when the rules start, such as one :func:`_knowledge_marks`
    set, stays. ``max_depth`` caps the conditioning sets of R4's tests (:func:`_do_ddp`). The rules
    run in rounds. In a round, every rule reads the marks as they stand when the round starts and
    records the marks it implies (:class:`_Implied`). A circle then takes the mark the rules imply
    for it when they imply only that one; a circle they imply both an arrowhead and a tail for stays
    a circle in that round. The rounds repeat until one changes no endpoint, and stop because each
    endpoint changes at most once. R4 decides by every
    shortest discriminating path, not the first one found (:func:`_rule_r4`), and R5 orients every
    edge of each uncovered circle cycle it finds, which the same round also finds from each of that
    cycle's edges. A round reads only the marks, never a node's position, so relabeling the nodes
    relabels its marks: the result does not depend on the order in which the rules visit the
    nodes, nor on the column order, except where R4's or R5's path search stops at its bound.

    With ``in_column_order``, each sweep instead runs the arrowhead rules (R1/R2 cycle, R3, and the
    R4 discriminating-path rule) followed by the tail and possibly-directed-path rules R5-R10, each
    rule setting its marks at once and visiting the nodes in column order, so of two firings that
    conflict the first decides; sweeps repeat until no endpoint changes.
    """
    if not in_column_order:
        test = _memoized(test)  # R4 repeats its tests in every round
        while True:
            implied = _Implied()
            _rule_r1r2_cycle(marks, False, ambiguous, out=implied)
            _rule_r3(marks, sepsets, False, ambiguous, out=implied)
            _rule_r4(marks, test, alpha, sepsets, False, max_depth=max_depth, out=implied)
            _rule_r5(marks, False, ambiguous, out=implied)
            _rule_r6(marks, False, out=implied)
            _rule_r7(marks, False, ambiguous, out=implied)
            _rule_r8(marks, False, out=implied)
            _rule_r9(marks, False, ambiguous, out=implied)
            _rule_r10(marks, False, ambiguous, out=implied)
            if not implied.apply(marks):
                return
    change = True
    while change:
        change = False
        change = _rule_r1r2_cycle(marks, change, ambiguous)
        change = _rule_r3(marks, sepsets, change, ambiguous)
        # R4 runs every sweep because a discriminating path can orient an otherwise unchanged graph.
        change = _rule_r4(marks, test, alpha, sepsets, change, max_depth=max_depth)
        change = _rule_r5(marks, change, ambiguous)
        change = _rule_r6(marks, change)
        change = _rule_r7(marks, change, ambiguous)
        change = _rule_r8(marks, change)
        change = _rule_r9(marks, change, ambiguous)
        change = _rule_r10(marks, change, ambiguous)


def _memoized(test: CITest) -> CITest:
    """``test``, answering a query it has answered before from memory."""
    answered = cache(lambda x, y, condition_set: test(x, y, condition_set))

    def memoized(x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        return answered(x, y, tuple(condition_set))

    return memoized


def _rule_r1(
    marks: np.ndarray,
    a: int,
    b: int,
    c: int,
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
    *,
    out: np.ndarray | _Implied | None = None,
) -> bool:
    """R1: ``a *-> b o-* c`` with ``a``, ``c`` non-adjacent  =>  ``b -> c``.

    Every rule reads ``marks`` and sets its marks in ``out``: ``marks`` itself when ``out`` is
    ``None``, or an :class:`_Implied` that records them.
    """
    out = marks if out is None else out
    if _adjacent(marks, a, c) or _triple(a, b, c) in ambiguous:
        return False
    if marks[b, a] == ARROW and marks[b, c] == CIRCLE and _arrow_allowed(marks, b, c):
        out[c, b] = ARROW
        out[b, c] = TAIL
        return True
    return False


def _rule_r2(
    marks: np.ndarray, a: int, b: int, c: int, *, out: np.ndarray | _Implied | None = None
) -> bool:
    """R2: ``a -> b *-> c`` or ``a *-> b -> c`` with ``a o-* c``  =>  arrowhead at ``c``."""
    out = marks if out is None else out
    if not (_adjacent(marks, a, c) and marks[c, a] == CIRCLE):
        return False
    directed = marks[b, a] == ARROW and marks[c, b] == ARROW
    if directed and (marks[a, b] == TAIL or marks[b, c] == TAIL) and _arrow_allowed(marks, a, c):
        out[c, a] = ARROW
        return True
    return False


def _rule_r1r2_cycle(
    marks: np.ndarray,
    change: bool,
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
    *,
    out: np.ndarray | _Implied | None = None,
) -> bool:
    """Sweep R1 and R2 over every node ``b`` and each ordered pair of its neighbors."""
    for b, adj in enumerate(_neighbor_lists(marks)):
        for i in range(len(adj)):
            for k in range(i + 1, len(adj)):
                a, c = adj[i], adj[k]
                change = _rule_r1(marks, a, b, c, ambiguous, out=out) or change
                change = _rule_r1(marks, c, b, a, ambiguous, out=out) or change
                change = _rule_r2(marks, a, b, c, out=out) or change
                change = _rule_r2(marks, c, b, a, out=out) or change
    return change


def _rule_r3(
    marks: np.ndarray,
    sepsets: Mapping[tuple[int, int], tuple[int, ...]],
    change: bool,
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
    *,
    out: np.ndarray | _Implied | None = None,
) -> bool:
    """R3 (double triangle): a shared circle child of two colliders gets an arrowhead."""
    out = marks if out is None else out
    n = marks.shape[0]
    for b in range(n):
        into_arrows = _into(marks, b, ARROW)
        into_circles = _into(marks, b, CIRCLE)
        if len(into_arrows) < 2:
            continue
        for d in into_circles:
            for i in range(len(into_arrows)):
                for k in range(i + 1, len(into_arrows)):
                    a, c = into_arrows[i], into_arrows[k]
                    if _adjacent(marks, a, c):
                        continue
                    if not (_adjacent(marks, a, d) and _adjacent(marks, c, d)):
                        continue
                    if _triple(a, d, c) in ambiguous:
                        continue
                    sep = _sepset(sepsets, a, c)
                    if sep is None or d not in sep:
                        continue
                    if marks[d, a] != CIRCLE or marks[d, c] != CIRCLE:
                        continue
                    if not _arrow_allowed(marks, d, b):
                        continue
                    out[b, d] = ARROW
                    change = True
    return change


def _rule_r4(
    marks: np.ndarray,
    test: CITest,
    alpha: float,
    sepsets: Mapping[tuple[int, int], tuple[int, ...]],
    change: bool,
    *,
    max_depth: int | None = None,
    out: _Implied | None = None,
) -> bool:
    """R4: orient the endpoints closing a definite discriminating path into ``b``.

    Without ``out``, the first discriminating path found that decides (:func:`_ddp_orient`) sets
    the marks in ``marks``. With ``out``, every shortest discriminating path that decides records
    its marks in ``out`` (:func:`_discriminating_paths`), so paths that decide differently leave
    the endpoint at ``b`` a circle.
    """
    n = marks.shape[0]
    for b in range(n):
        poss_a = [a for a in range(n) if marks[a, b] == ARROW]  # b *-> a
        poss_c = _into(marks, b, CIRCLE)  # c *-o b
        for a in poss_a:
            for c in poss_c:
                if not _is_parent_of(marks, a, c):
                    continue
                if marks[c, b] != ARROW:
                    continue
                if out is None:
                    change = _ddp_orient(
                        marks, a, b, c, test, alpha, sepsets, change, max_depth=max_depth
                    )
                    continue
                for paths in _discriminating_paths(marks, a, b, c):
                    decided = False
                    for path in paths:
                        previous = dict(zip(path, path[1:]))
                        oriented, _ = _do_ddp(
                            marks,
                            path[0],
                            a,
                            b,
                            c,
                            previous,
                            test,
                            alpha,
                            sepsets,
                            False,
                            max_depth=max_depth,
                            out=out,
                        )
                        decided = oriented or decided
                    if decided:
                        change = True
                        break
    return change


def _discriminating_paths(marks: np.ndarray, a: int, b: int, c: int) -> Iterator[list[list[int]]]:
    """The discriminating paths ``d, ..., a, b`` for ``b`` toward ``c``, shortest first.

    The paths grow backward from ``a`` one node at a time, every inner node a collider on the path
    and a parent of ``c``, as :func:`_ddp_orient` grows them, and end at a node ``d`` non-adjacent
    to ``c``. Each yield lists every path of one length, from ``d`` to ``b``. The search stops at
    ``_UNBOUNDED_PATH`` nodes, or when one length holds more than ``_R4_PARTIAL_PATHS`` partial
    paths.
    """
    c_parents = _parents(marks, c)
    layer = [[b, a]]  # each partial path read from b backward
    for _ in range(_UNBOUNDED_PATH):
        found, extended = [], []
        for path in layer:
            t, p = path[-1], path[-2]
            for d in _into(marks, t, ARROW):  # d *-> t
                if d in path or d == c or not _is_def_collider(marks, d, t, p):
                    continue
                if not _adjacent(marks, d, c):
                    found.append([*path, d][::-1])
                elif d in c_parents:
                    extended.append([*path, d])
        if found:
            yield found
        if not extended or len(extended) > _R4_PARTIAL_PATHS:
            return
        layer = extended


def _get_path(node: int, previous: Mapping[int, int]) -> list[int]:
    """The predecessor chain of ``node`` under ``previous`` (nearest predecessor first)."""
    path: list[int] = []
    p = previous.get(node)
    while p is not None:
        path.append(p)
        p = previous.get(p)
    return path


def _ddp_orient(
    marks: np.ndarray,
    a: int,
    b: int,
    c: int,
    test: CITest,
    alpha: float,
    sepsets: Mapping[tuple[int, int], tuple[int, ...]],
    change: bool,
    *,
    max_depth: int | None = None,
) -> bool:
    """Search back from ``a`` for a definite discriminating path ending at ``c`` and orient it."""
    queue: deque[int] = deque([a])
    visited: set[int] = {a, b}
    previous: dict[int, int] = {a: b}
    c_parents = _parents(marks, c)
    frontier: int | None = None
    distance = 0

    while queue:
        t = queue.popleft()
        if frontier is None or frontier == t:
            frontier = t
            distance += 1
            if distance > _UNBOUNDED_PATH:
                return change
        for d in _into(marks, t, ARROW):  # d *-> t
            if d in visited:
                continue
            previous[d] = t
            p = previous[t]
            if not _is_def_collider(marks, d, t, p):
                continue
            if not _adjacent(marks, d, c) and d != c:
                oriented, change = _do_ddp(
                    marks, d, a, b, c, previous, test, alpha, sepsets, change, max_depth=max_depth
                )
                if oriented:
                    return change
            if d in c_parents:
                queue.append(d)
                visited.add(d)
    return change


def _do_ddp(
    marks: np.ndarray,
    d: int,
    a: int,
    b: int,
    c: int,
    previous: Mapping[int, int],
    test: CITest,
    alpha: float,
    sepsets: Mapping[tuple[int, int], tuple[int, ...]],
    change: bool,
    *,
    max_depth: int | None = None,
    out: np.ndarray | _Implied | None = None,
) -> tuple[bool, bool]:
    """Resolve the discriminating-path endpoint at ``b``: tail if ``b`` separates ``d``/``c``.

    ``d`` and ``c`` are tested given the path's inner nodes, with and without ``b``; a set larger
    than ``max_depth`` is not tested and counts as dependent. When neither test separates them, the
    recorded separating set decides.
    """
    out = marks if out is None else out
    path = _get_path(d, previous)
    ind = _fits(path, max_depth) and test(d, c, tuple(path)) > alpha
    path2 = [p for p in path if p != b]
    ind2 = _fits(path2, max_depth) and test(d, c, tuple(path2)) > alpha

    if not ind and not ind2:
        sep = _sepset(sepsets, d, c)
        if sep is None:
            return False, change
        ind = b in sep

    if ind:
        out[b, c] = TAIL
        return True, True
    if not _arrow_allowed(marks, a, b) or not _arrow_allowed(marks, c, b):
        return False, change
    out[b, a] = ARROW
    out[b, c] = ARROW
    return True, True


def _traverse_circle(marks: np.ndarray, node: int, other: int) -> bool:
    """True when edge ``node-other`` is a full circle edge ``o-o`` (circle at both endpoints)."""
    return bool(marks[node, other] == CIRCLE and marks[other, node] == CIRCLE)


def _is_uncovered(marks: np.ndarray, path: list[int]) -> bool:
    """True when no two nodes two apart on ``path`` are adjacent (an uncovered path)."""
    return all(not _adjacent(marks, path[i], path[i + 2]) for i in range(len(path) - 2))


def _circle_reaches(marks: np.ndarray, target: int, avoid: int) -> set[int]:
    """Nodes joined to ``target`` by a circle walk (every edge ``o-o``) that avoids ``avoid``."""
    reachable = {target}
    queue = deque([target])
    while queue:
        node = queue.popleft()
        for other in range(marks.shape[0]):
            if other != avoid and other not in reachable and _traverse_circle(marks, node, other):
                reachable.add(other)
                queue.append(other)
    return reachable


def _circle_walk_closes(
    marks: np.ndarray,
    a: int,
    b: int,
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
) -> bool:
    """True when a circle walk ``a, c, ..., d, b`` meets every R5 condition except simplicity.

    The search runs over ``(previous, current)`` edges, so it takes polynomial time. Every
    uncovered circle path is such a walk, so ``False`` rules out an R5 cycle for ``a o-o b``.
    """
    n = marks.shape[0]
    start = [
        (a, c)
        for c in range(n)
        if c != b
        and _traverse_circle(marks, a, c)
        and not _adjacent(marks, b, c)
        and _triple(b, a, c) not in ambiguous
    ]
    seen = set(start)
    queue = deque(start)
    while queue:
        previous, current = queue.popleft()
        for node in range(n):
            if node in (previous, a) or not _traverse_circle(marks, current, node):
                continue
            if _adjacent(marks, previous, node) or _triple(previous, current, node) in ambiguous:
                continue
            if node == b:
                if not _adjacent(marks, a, current) and _triple(current, b, a) not in ambiguous:
                    return True
            elif (current, node) not in seen:
                seen.add((current, node))
                queue.append((current, node))
    return False


def _uncovered_circle_cycle(
    marks: np.ndarray,
    a: int,
    b: int,
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
) -> list[int] | None:
    """Inner nodes ``c, ..., d`` of an uncovered circle cycle ``a, c, ..., d, b, a``, or ``None``.

    The circle path ``a, c, ..., d, b`` is simple and uncovered, and ``a``-``d`` and ``b``-``c``
    are non-adjacent, so the cycle closed by ``a o-o b`` is uncovered too. No triple on the cycle
    is ambiguous. Search state belongs to each path: reaching a node through one predecessor must
    not block a different route. Reverse reachability prunes nodes with no circle walk to ``b``
    avoiding ``a``.

    Finding such a simple path is NP-complete in general [1]_. A polynomial walk check
    (:func:`_circle_walk_closes`) rules out most pairs first; the path search then stops after
    ``_R5_PARTIAL_PATHS`` partial paths and returns ``None``.

    References
    ----------
    .. [1] Szeider, S. (2003). Finding paths in graphs avoiding forbidden transitions. Discrete
       Applied Mathematics 126(2-3), 261-273.
    """
    if not _circle_walk_closes(marks, a, b, ambiguous):
        return None
    reachable = _circle_reaches(marks, b, a)
    stack = [
        [a, c]
        for c in range(marks.shape[0])
        if c != b
        and c in reachable
        and _traverse_circle(marks, a, c)
        and not _adjacent(marks, b, c)
        and _triple(b, a, c) not in ambiguous
    ]
    for _ in range(_R5_PARTIAL_PATHS):
        if not stack:
            return None
        path = stack.pop()
        last = path[-1]
        for node in range(marks.shape[0]):
            if node not in reachable or node in path or not _traverse_circle(marks, last, node):
                continue
            if _adjacent(marks, path[-2], node) or _triple(path[-2], last, node) in ambiguous:
                continue
            if node != b:
                stack.append([*path, node])
            elif not _adjacent(marks, a, last) and _triple(last, b, a) not in ambiguous:
                return path[1:]
    return None


def _rule_r5(
    marks: np.ndarray,
    change: bool,
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
    *,
    out: np.ndarray | _Implied | None = None,
) -> bool:
    """R5: orient an uncovered circle cycle to tails when every cycle triple is unambiguous.

    The cycle is ``a o-o b`` closed by a circle path ``a, c, ..., d, b`` with ``a``-``d`` and
    ``b``-``c`` non-adjacent; the whole path, not only ``c, ..., d``, must be uncovered
    (Zhang 2008). Every simple path is a candidate, so a covered route through a node never hides
    an uncovered route through the same node. A pair whose search reaches ``_R5_PARTIAL_PATHS``
    keeps its circles.
    """
    n = marks.shape[0]
    for a in range(n):
        for b in range(a + 1, n):
            if not _traverse_circle(marks, a, b):
                continue
            path = _uncovered_circle_cycle(marks, a, b, ambiguous)
            if path is not None:
                _orient_tails(marks if out is None else out, a, b, path)
                change = True
    return change


def _orient_tails(marks: np.ndarray | _Implied, a: int, b: int, path: list[int]) -> None:
    """Set every endpoint on ``a - path - b`` to a tail (the R5 double-tail orientation)."""
    marks[a, b] = TAIL
    marks[b, a] = TAIL
    marks[a, path[0]] = TAIL
    marks[path[0], a] = TAIL
    marks[b, path[-1]] = TAIL
    marks[path[-1], b] = TAIL
    for i in range(len(path) - 1):
        marks[path[i], path[i + 1]] = TAIL
        marks[path[i + 1], path[i]] = TAIL


def _rule_r6(marks: np.ndarray, change: bool, *, out: np.ndarray | _Implied | None = None) -> bool:
    """R6: ``a --- b o-* c``  =>  tail at ``b`` on edge ``b-c``."""
    out = marks if out is None else out
    n = marks.shape[0]
    for b in range(n):
        if not any(marks[b, a] == TAIL and marks[a, b] == TAIL for a in range(n)):
            continue
        for c in _into(marks, b, CIRCLE):
            out[b, c] = TAIL
            change = True
    return change


def _rule_r7(
    marks: np.ndarray,
    change: bool,
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
    *,
    out: np.ndarray | _Implied | None = None,
) -> bool:
    """R7: ``a --o b o-* c`` with ``a``, ``c`` non-adjacent  =>  tail at ``b`` on edge ``b-c``."""
    out = marks if out is None else out
    n = marks.shape[0]
    for b in range(n):
        into_circles = _into(marks, b, CIRCLE)
        tail_side = [a for a in into_circles if marks[a, b] == TAIL]  # a --o b
        for c in into_circles:
            for a in tail_side:
                if a == c or _adjacent(marks, a, c) or _triple(a, b, c) in ambiguous:
                    continue
                out[b, c] = TAIL
                change = True
    return change


def _rule_r8(marks: np.ndarray, change: bool, *, out: np.ndarray | _Implied | None = None) -> bool:
    """R8: ``a -> b -> c`` (or ``a --o b -> c``) with ``a o-> c``  =>  ``a -> c``."""
    out = marks if out is None else out
    for b, adj in enumerate(_neighbor_lists(marks)):
        # Both orders of each neighbor pair, so the rule does not depend on node labels.
        for x, z in permutations(adj, 2):
            b_to_z = marks[z, b] == ARROW and marks[b, z] == TAIL
            x_to_b_directed = marks[b, x] == ARROW and marks[x, b] == TAIL
            x_to_b_circle = marks[b, x] == CIRCLE and marks[x, b] == TAIL
            x_o_arrow_z = _adjacent(marks, x, z) and marks[z, x] == ARROW and marks[x, z] == CIRCLE
            if (x_to_b_directed or x_to_b_circle) and b_to_z and x_o_arrow_z:
                out[x, z] = TAIL
                out[z, x] = ARROW
                change = True
    return change


def _pd_reaches_target(
    marks: np.ndarray, target: int, a: int, neighbors: list[list[int]]
) -> set[int]:
    """Nodes with a possibly-directed walk to ``target`` that avoids ``a``.

    ``neighbors`` holds the :func:`_neighbor_lists` of ``marks``.
    """
    reachable = {target}
    queue = deque([target])
    while queue:
        node = queue.popleft()
        for previous in neighbors[node]:
            if (
                previous not in reachable
                and previous != a
                and marks[previous, node] != ARROW
                and marks[node, previous] != TAIL
            ):
                reachable.add(previous)
                queue.append(previous)
    return reachable


def _exists_uncovered_pd_path(
    marks: np.ndarray,
    a: int,
    first: int,
    target: int,
    neighbors: list[list[int]],
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
) -> bool:
    """True when an uncovered possibly-directed simple path ``a, first, ..., target`` exists.

    Each step has no arrowhead at its near end and no tail at its far end. Search state belongs
    to each path: reaching a node through one predecessor must not block a different route.
    Reverse reachability prunes branches with no p.d. walk to the target avoiding ``a``.
    Ambiguity checks cover interior triples of the open path, with no wraparound at its target.
    ``neighbors`` holds the :func:`_neighbor_lists` of ``marks``.
    """
    if marks[a, first] not in (TAIL, CIRCLE) or marks[first, a] == TAIL:
        return False
    reachable = _pd_reaches_target(marks, target, a, neighbors)
    if first not in reachable:
        return False
    stack = [[a, first]]
    while stack:
        path = stack.pop()
        last = path[-1]
        if last == target:
            return True
        for node in neighbors[last]:
            if (
                node not in reachable
                or node in path
                or marks[last, node] == ARROW
                or marks[node, last] == TAIL
            ):
                continue
            # The prefix is already uncovered; only the new triple needs checking.
            if (
                _is_uncovered(marks, [path[-2], last, node])
                and _triple(path[-2], last, node) not in ambiguous
            ):
                stack.append([*path, node])
    return False


def _rule_r9(
    marks: np.ndarray,
    change: bool,
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
    *,
    out: np.ndarray | _Implied | None = None,
) -> bool:
    """R9: ``a o-> c`` with an uncovered possibly-directed path ``a .. b .. c``  =>  ``a -> c``."""
    out = marks if out is None else out
    n = marks.shape[0]
    neighbors = _neighbor_lists(marks)
    for c in range(n):
        for a in _into(marks, c, ARROW):
            if marks[a, c] != CIRCLE:  # need a o-> c
                continue
            candidates = [
                v for v in range(n) if v not in (a, c) and _is_possible_child(marks, a, v)
            ]
            for b in candidates:
                if _adjacent(marks, b, c):
                    continue
                if _exists_uncovered_pd_path(marks, a, b, c, neighbors, ambiguous):
                    out[a, c] = TAIL
                    out[c, a] = ARROW
                    change = True
                    break
    return change


def _is_possible_child(marks: np.ndarray, parent: int, child: int) -> bool:
    """True when ``parent`` may be a parent of ``child`` (adjacent, no arrowhead at ``parent``)."""
    if parent == child or not _adjacent(marks, parent, child):
        return False
    return bool(marks[parent, child] != ARROW)


def _rule_r10(
    marks: np.ndarray,
    change: bool,
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
    *,
    out: np.ndarray | _Implied | None = None,
) -> bool:
    """R10: complete ``a o-> c`` via two uncovered possibly-directed paths into ``b -> c <- d``.

    The paths reach the tails ``b`` and ``d`` from two non-adjacent possible children of ``a``.
    """
    out = marks if out is None else out
    n = marks.shape[0]
    neighbors = _neighbor_lists(marks)
    for c in range(n):
        into_arrows = _into(marks, c, ARROW)
        if len(into_arrows) < 2:
            continue
        a_nodes = [a for a in into_arrows if marks[a, c] == CIRCLE]  # a o-> c
        for a in a_nodes:
            children = [
                node for node in range(n) if node != c and _is_possible_child(marks, a, node)
            ]
            if len(children) < 2:
                continue
            for i in range(len(into_arrows)):
                for k in range(i + 1, len(into_arrows)):
                    b, d = into_arrows[i], into_arrows[k]
                    if marks[b, c] != TAIL or marks[d, c] != TAIL:  # need b -> c <- d
                        continue
                    if _r10_connects(marks, children, b, d, a, neighbors, ambiguous):
                        out[a, c] = TAIL
                        out[c, a] = ARROW
                        change = True
    return change


def _r10_connects(
    marks: np.ndarray,
    children: list[int],
    b: int,
    d: int,
    a: int,
    neighbors: list[list[int]],
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
) -> bool:
    """True when uncovered p.d. paths from ``a`` reach ``b``/``d`` via non-adjacent children.

    Either child of a pair may lead to either tail. Each child is searched at most once per tail.
    ``neighbors`` holds the :func:`_neighbor_lists` of ``marks``.
    """
    reaches: dict[tuple[int, int], bool] = {}

    def reach(child: int, target: int) -> bool:
        if (child, target) not in reaches:
            reaches[child, target] = _exists_uncovered_pd_path(
                marks, a, child, target, neighbors, ambiguous
            )
        return reaches[child, target]

    for one, two in combinations(children, 2):
        if _adjacent(marks, one, two) or _triple(one, a, two) in ambiguous:
            continue
        if (reach(one, b) and reach(two, d)) or (reach(one, d) and reach(two, b)):
            return True
    return False
