"""FCI (Fast Causal Inference) causal discovery, the engine behind :func:`andrey.fci`.

The algorithm, its settings, and its references are described on :func:`andrey.fci`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping
from itertools import combinations, islice

import numpy as np
from threadpoolctl import threadpool_limits

from andrey.constraint._parallel_skeleton import SkeletonWorkerPool
from andrey.constraint._skeleton import (
    adjacent_pairs,
    first_order_separating,
    marginal_removals,
    prescreen_capable,
    scan_pairs,
)
from andrey.core import ARROW, CIRCLE, NULL, TAIL, GraphStructure
from andrey.core.ci import CITest, make_indep_test

# Bound discriminating and possible-d-sep paths by this finite cap.
_UNBOUNDED_PATH = 1000


def fci(
    data: np.ndarray,
    *,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    collider_rule: str = "sepsets",
) -> GraphStructure:
    """Discover a PAG from ``data`` with the FCI algorithm and the ``indep_test`` CI test.

    ``data`` is an ``(n_samples, n_features)`` array; ``alpha`` is the independence-test
    significance level in ``(0, 1)``. Skeleton pruning requires a p-value above ``alpha``;
    the optional majority collider rule also counts equality as a separating vote.
    Returns the PAG as a ``GraphStructure`` of kind ``"pag"`` (circle marks and all).
    """
    X = np.asarray(data, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"data must be a 2-D (n_samples, n_features) array, got ndim={X.ndim}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in the open interval (0, 1), got {alpha}")

    _validate_collider_rule(collider_rule)
    test = make_indep_test(indep_test, X)
    support, sepsets = _fast_adjacency_search(X, test, alpha)

    marks = _circle_pag(support)
    _orient_colliders(marks, sepsets)
    _remove_by_possible_dsep(marks, test, alpha, sepsets)
    marks = _circle_pag(marks != NULL)
    ambiguous = (
        _majority_sepsets(marks, test, alpha, sepsets)
        if collider_rule == "majority"
        else frozenset()
    )
    _orient_colliders(marks, sepsets, ambiguous)
    _fci_orient(marks, test, alpha, sepsets, ambiguous)

    return GraphStructure.from_numpy(marks, kind="pag")


# ---- skeleton (fast adjacency search) -----------------------------------------------------------


def _fast_adjacency_search(
    X: np.ndarray, test: CITest, alpha: float, *, prescreen: bool = True, workers: int = 1
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, ...]]]:
    """Learn the undirected skeleton and every removed pair's separating set (stable, FAS-style).

    Grows the conditioning size from zero over a complete graph; at each size, every still-adjacent
    pair ``(x, y)`` is tested against subsets of a start-of-size snapshot of *either* endpoint's
    neighbourhood, and pairs found independent are removed together after the size completes.
    A removed pair's separating set is the union of every conditioning set that made it independent.

    Either endpoint supplies the conditioning candidates, so a separating set is reachable wherever
    it lives and the skeleton does not depend on how the caller ordered the columns.

    When ``prescreen`` is set and the test exposes the vectorized closed forms, the size-0 and
    size-1 passes read the correlation matrix directly instead of the dense scalar scan -- same
    removals and separating sets. With ``workers > 1`` the ``|S| >= 2`` scan fans out across worker
    processes, reducing per-chunk removals by union, so the skeleton stays identical to the serial
    scan.
    """
    d = X.shape[1]
    support = np.ones((d, d), dtype=bool)
    np.fill_diagonal(support, False)
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {}

    can_prescreen = prescreen and prescreen_capable(test)
    parallel = workers > 1 and callable(getattr(test, "to_seed", None))
    pool = SkeletonWorkerPool(test, workers) if parallel else None
    try:
        size = -1
        while True:
            degrees = support.sum(axis=1)
            if int(degrees.max(initial=0)) - 1 <= size:
                break
            size += 1
            neighbours = [np.flatnonzero(support[i]) for i in range(d)]
            remove: list[tuple[int, int]] = []
            if can_prescreen and size == 0:
                for x, y in marginal_removals(test, np.triu(support, 1), alpha):
                    remove.append((x, y))
                    sepsets[(x, y)] = ()
                    sepsets[(y, x)] = ()  # skeleton separations are recorded both ways
            elif can_prescreen and size == 1:
                for x, y in adjacent_pairs(support):
                    candidates = sorted(
                        (set(neighbours[x].tolist()) | set(neighbours[y].tolist())) - {x, y}
                    )
                    separating = first_order_separating(test, x, y, candidates, alpha)
                    if separating:
                        remove.append((x, y))
                        key = tuple(separating)
                        sepsets[(x, y)] = key
                        sepsets[(y, x)] = key
            else:
                pairs = adjacent_pairs(support)
                if pool is not None:
                    removals = pool.scan(pairs, support.copy(), size, alpha)
                else:
                    # Pin BLAS to one thread: the conditioning-submatrix inversions then match the
                    # (1-thread) parallel skeleton workers regardless of the host thread count.
                    with threadpool_limits(limits=1):
                        removals = scan_pairs(test, pairs, neighbours, size, alpha)
                for x, y, separating in removals:
                    remove.append((x, y))
                    sepsets[(x, y)] = separating
                    sepsets[(y, x)] = separating
            for x, y in remove:
                support[x, y] = False
                support[y, x] = False
    finally:
        if pool is not None:
            pool.close()

    return support, sepsets


# ---- mark-matrix helpers ------------------------------------------------------------------------
# ``marks[i, j]`` is the mark at node ``i`` on edge ``i-j``, as in :mod:`andrey.core.structure`.


def _circle_pag(support: np.ndarray) -> np.ndarray:
    """An all-circle PAG over the given adjacency support (every present edge becomes ``o-o``)."""
    marks = np.where(support != NULL, np.int8(CIRCLE), np.int8(NULL))
    np.fill_diagonal(marks, np.int8(NULL))
    return marks


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


def _adjacent_nodes(marks: np.ndarray, b: int) -> list[int]:
    """Ascending list of nodes sharing an edge with ``b``."""
    return [j for j in range(marks.shape[0]) if marks[b, j] != NULL]


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


# ---- R0: unshielded colliders -------------------------------------------------------------------


def _validate_collider_rule(collider_rule: str) -> None:
    if collider_rule not in ("sepsets", "majority"):
        raise ValueError(f"collider_rule must be 'sepsets' or 'majority', got {collider_rule!r}")


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
) -> frozenset[tuple[int, int, int]]:
    """Rewrite separating sets by majority vote on the final skeleton; return ambiguous triples.

    Every subset of each endpoint's adjacency votes separately, including the empty set and
    subsets shared by both endpoints. Independence is ``p >= alpha``. The recorded sets do not
    vote. A missing key direction inherits the reverse entry before membership is rewritten.
    Fewer than half containing the middle node (or no separating subset) makes a collider;
    more than half makes a noncollider. An exact half leaves both recorded sets unchanged.
    Cost grows exponentially with endpoint degree; conditioning sets have no size cap. Subsets are
    tested in chunks of ``_MAJORITY_CHUNK``, so memory per triple stays bounded.
    """
    neighbours = [_adjacent_nodes(marks, node) for node in range(len(marks))]
    ambiguous = set()
    batched = getattr(test, "batched_call", None)
    for b, adjacent in enumerate(neighbours):
        for a, c in combinations(adjacent, 2):
            if _adjacent(marks, a, c):
                continue
            subsets = (
                subset
                for endpoint in (a, c)
                for size in range(len(neighbours[endpoint]) + 1)
                for subset in combinations(neighbours[endpoint], size)
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

    For each node ``b`` and each non-adjacent pair ``(a, c)`` of its neighbours (visited in
    ascending order), ``b`` is a collider iff a separating set for ``a`` and ``c`` is recorded and
    does not contain ``b``; the endpoint at ``b`` on both edges is then set to an arrowhead. An
    already-directed tail at ``b`` blocks the orientation.
    """
    n = marks.shape[0]
    for b in range(n):
        adj = _adjacent_nodes(marks, b)
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


def _exists_semidirected_path(marks: np.ndarray, src: int, dst: int) -> bool:
    """True when a semidirected path ``src o-> ... o-> dst`` exists.

    A step ``src -> u`` is allowed when the endpoint at ``src`` is a tail or a circle.
    """
    n = marks.shape[0]
    seen: set[int] = set()
    queue: deque[int] = deque()
    for u in range(n):
        if marks[src, u] in (TAIL, CIRCLE) and u not in seen:
            seen.add(u)
            queue.append(u)
    while queue:
        node = queue.popleft()
        if node == dst:
            return True
        for u in range(n):
            if marks[node, u] in (TAIL, CIRCLE) and u not in seen:
                seen.add(u)
                queue.append(u)
    return False


def _exists_possible_parent_path(
    previous: Mapping[int, set[int]], node: int, x: int, marks: np.ndarray
) -> bool:
    """The possible-d-sep reachability check over the ``previous`` predecessor sets."""
    if node == x:
        return True
    preds = previous.get(node)
    if preds is None:
        return False
    for r in preds:
        if r == node or r == x:
            continue
        if _exists_semidirected_path(marks, r, x) or _exists_semidirected_path(marks, r, node):
            return True
    return False


def _possible_dsep(marks: np.ndarray, x: int, y: int) -> list[int]:
    """Possible-d-separating nodes for the pair ``(x, y)`` (descending), from a reachability BFS.

    Grows a frontier of node pairs from ``x``, extending through definite colliders and adjacent
    triples, and collects every intermediate node reachable by a possible-parent path.
    """
    n = marks.shape[0]
    dsep: set[int] = set()
    queue: deque[tuple[int, int]] = deque()
    visited: set[tuple[int, int]] = set()
    previous: dict[int, set[int]] = {}
    first: tuple[int, int] | None = None
    distance = 0

    for b in range(n):
        if not _adjacent(marks, x, b) or b == y:
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
        if _exists_possible_parent_path(previous, b, x, marks):
            dsep.add(b)
        for c in range(n):
            if not _adjacent(marks, b, c) or c in (a, x, y):
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
    # at two digits; this order determines which conditioning subsets are skipped.
    return sorted(dsep, key=lambda v: f"X{v + 1}", reverse=True)


def _iter_dsep_choices(
    dsep: list[int], adj_a: set[int], adj_b: set[int]
) -> Iterator[tuple[int, ...]]:
    """Conditioning subsets of ``dsep`` (size >= 2) outside either endpoint's adjacency.

    Candidate subsets are enumerated by increasing size. For each size, the first combination
    is skipped and the last is emitted twice; a size with a single combination emits it once.
    Subsets are generated one at a time and filtered against the endpoint adjacencies.
    """

    def outside(values: tuple[int, ...]) -> bool:
        return not (all(v in adj_a for v in values) or all(v in adj_b for v in values))

    for size in range(2, len(dsep) + 1):
        combos = combinations(dsep, size)
        last = next(combos)  # skip the first combination of each size
        for values in combos:
            if outside(values):
                yield values
            last = values
        if outside(last):  # and emits the last one twice
            yield last


def _try_remove_by_dsep(
    marks: np.ndarray,
    a: int,
    b: int,
    x: int,
    y: int,
    adj_a: set[int],
    adj_b: set[int],
    test: CITest,
    alpha: float,
    sepsets: dict[tuple[int, int], tuple[int, ...]],
) -> bool:
    """Remove edge ``x-y`` if some possible-d-sep subset makes ``x`` and ``y`` independent."""
    dsep = _possible_dsep(marks, a, b)
    if len(dsep) < 2:
        return False
    for cond in _iter_dsep_choices(dsep, adj_a, adj_b):
        if test(x, y, cond) > alpha:
            marks[x, y] = NULL
            marks[y, x] = NULL
            sepsets[(x, y)] = tuple(int(v) for v in cond)
            return True
    return False


def _edge_order(marks: np.ndarray, i: int, j: int) -> tuple[int, int]:
    """Return the endpoint order used by the possible-d-sep search.

    A left-pointing edge uses ``(j, i)``. Every other edge uses ``(i, j)``, so its search order
    follows the column indices.
    """
    if marks[i, j] == ARROW and marks[j, i] in (TAIL, CIRCLE):
        return j, i
    return i, j


def _remove_by_possible_dsep(
    marks: np.ndarray,
    test: CITest,
    alpha: float,
    sepsets: dict[tuple[int, int], tuple[int, ...]],
) -> None:
    """Prune edges the plain skeleton kept but a possible-d-sep conditioning set separates.

    Search each endpoint's possible-d-sep set for a conditioning subset outside either endpoint's
    adjacency. Store a successful subset under the endpoint order returned by ``_edge_order``.
    """
    n = marks.shape[0]
    edges = [_edge_order(marks, i, j) for i in range(n) for j in range(i + 1, n) if marks[i, j]]
    for na, nb in edges:
        if not _adjacent(marks, na, nb):
            continue
        adj_a = set(_adjacent_nodes(marks, na))
        adj_b = set(_adjacent_nodes(marks, nb))
        _try_remove_by_dsep(marks, na, nb, na, nb, adj_a, adj_b, test, alpha, sepsets)
        if _adjacent(marks, na, nb):
            _try_remove_by_dsep(marks, nb, na, na, nb, adj_a, adj_b, test, alpha, sepsets)


# ---- FCI orientation rules ----------------------------------------------------------------------


def _fci_orient(
    marks: np.ndarray,
    test: CITest,
    alpha: float,
    sepsets: Mapping[tuple[int, int], tuple[int, ...]],
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
) -> None:
    """Apply the FCI orientation rules R1-R10 to a fixed point, in place over ``marks``.

    Each sweep runs the arrowhead rules (R1/R2 cycle, R3, and the R4 discriminating-path rule)
    followed by the tail and possibly-directed-path rules R5-R10; sweeps repeat until no endpoint
    changes. Circles become tails or arrowheads only where the current marks force it.
    """
    change = True
    while change:
        change = False
        change = _rule_r1r2_cycle(marks, change, ambiguous)
        change = _rule_r3(marks, sepsets, change, ambiguous)
        # R4 runs every sweep because a discriminating path can orient an otherwise unchanged graph.
        change = _rule_r4(marks, test, alpha, sepsets, change)
        change = _rule_r5(marks, change, ambiguous)
        change = _rule_r6(marks, change)
        change = _rule_r7(marks, change, ambiguous)
        change = _rule_r8(marks, change)
        change = _rule_r9(marks, change, ambiguous)
        change = _rule_r10(marks, change, ambiguous)


def _rule_r1(
    marks: np.ndarray,
    a: int,
    b: int,
    c: int,
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
) -> bool:
    """R1: ``a *-> b o-* c`` with ``a``, ``c`` non-adjacent  =>  ``b -> c``."""
    if _adjacent(marks, a, c) or _triple(a, b, c) in ambiguous:
        return False
    if marks[b, a] == ARROW and marks[b, c] == CIRCLE and _arrow_allowed(marks, b, c):
        marks[c, b] = ARROW
        marks[b, c] = TAIL
        return True
    return False


def _rule_r2(marks: np.ndarray, a: int, b: int, c: int) -> bool:
    """R2: ``a -> b *-> c`` or ``a *-> b -> c`` with ``a o-* c``  =>  arrowhead at ``c``."""
    if not (_adjacent(marks, a, c) and marks[c, a] == CIRCLE):
        return False
    directed = marks[b, a] == ARROW and marks[c, b] == ARROW
    if directed and (marks[a, b] == TAIL or marks[b, c] == TAIL) and _arrow_allowed(marks, a, c):
        marks[c, a] = ARROW
        return True
    return False


def _rule_r1r2_cycle(
    marks: np.ndarray, change: bool, ambiguous: frozenset[tuple[int, int, int]] = frozenset()
) -> bool:
    """Sweep R1 and R2 over every node ``b`` and each ordered pair of its neighbours."""
    n = marks.shape[0]
    for b in range(n):
        adj = _adjacent_nodes(marks, b)
        for i in range(len(adj)):
            for k in range(i + 1, len(adj)):
                a, c = adj[i], adj[k]
                change = _rule_r1(marks, a, b, c, ambiguous) or change
                change = _rule_r1(marks, c, b, a, ambiguous) or change
                change = _rule_r2(marks, a, b, c) or change
                change = _rule_r2(marks, c, b, a) or change
    return change


def _rule_r3(
    marks: np.ndarray,
    sepsets: Mapping[tuple[int, int], tuple[int, ...]],
    change: bool,
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
) -> bool:
    """R3 (double triangle): a shared circle child of two colliders gets an arrowhead."""
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
                    marks[b, d] = ARROW
                    change = True
    return change


def _rule_r4(
    marks: np.ndarray,
    test: CITest,
    alpha: float,
    sepsets: Mapping[tuple[int, int], tuple[int, ...]],
    change: bool,
) -> bool:
    """R4: orient the endpoints closing a definite discriminating path into ``b``."""
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
                change = _ddp_orient(marks, a, b, c, test, alpha, sepsets, change)
    return change


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
                    marks, d, a, b, c, previous, test, alpha, sepsets, change
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
) -> tuple[bool, bool]:
    """Resolve the discriminating-path endpoint at ``b``: tail if ``b`` separates ``d``/``c``."""
    path = _get_path(d, previous)
    ind = test(d, c, tuple(path)) > alpha
    path2 = [p for p in path if p != b]
    ind2 = test(d, c, tuple(path2)) > alpha

    if not ind and not ind2:
        sep = _sepset(sepsets, d, c)
        if sep is None:
            return False, change
        ind = b in sep

    if ind:
        marks[b, c] = TAIL
        return True, True
    if not _arrow_allowed(marks, a, b) or not _arrow_allowed(marks, c, b):
        return False, change
    marks[b, a] = ARROW
    marks[b, c] = ARROW
    return True, True


def _traverse_circle(marks: np.ndarray, node: int, other: int) -> bool:
    """True when edge ``node-other`` is a full circle edge ``o-o`` (circle at both endpoints)."""
    return bool(marks[node, other] == CIRCLE and marks[other, node] == CIRCLE)


def _is_uncovered(marks: np.ndarray, path: list[int]) -> bool:
    """True when no two nodes two apart on ``path`` are adjacent (an uncovered path)."""
    return all(not _adjacent(marks, path[i], path[i + 2]) for i in range(len(path) - 2))


def _uncovered_circle_paths(
    marks: np.ndarray,
    src: int,
    dst: int,
    exclude: tuple[int, int],
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
) -> Iterator[list[int]]:
    """Yield uncovered circle paths from a breadth-first search with one shared visited set.

    Excluded endpoints cannot occur on the path. Each node is enqueued at most once, so alternate
    prefixes are not enumerated. Ambiguity filters completed paths without changing the search.
    """
    n = marks.shape[0]
    queue: deque[tuple[int, list[int]]] = deque()
    visited: set[int] = set()
    for u in range(n):
        if u in exclude or not _traverse_circle(marks, src, u):
            continue
        if u not in visited:
            visited.add(u)
            queue.append((u, [src, u]))
    while queue:
        node, path = queue.popleft()
        if (
            node == dst
            and _is_uncovered(marks, path)
            and not any(_triple(*path[i : i + 3]) in ambiguous for i in range(len(path) - 2))
        ):
            yield path
        for u in range(n):
            if u in exclude or not _traverse_circle(marks, node, u):
                continue
            if u in visited:
                continue
            visited.add(u)
            queue.append((u, path + [u]))


def _rule_r5(
    marks: np.ndarray, change: bool, ambiguous: frozenset[tuple[int, int, int]] = frozenset()
) -> bool:
    """R5: orient an uncovered circle cycle to tails when every cycle triple is unambiguous.

    The cycle is ``a o-o b`` closed by a circle path ``a, c, ..., d, b`` with ``a``-``d`` and
    ``b``-``c`` non-adjacent; the whole path, not only ``c, ..., d``, must be uncovered
    (Zhang 2008).
    """
    n = marks.shape[0]
    for b in range(n):
        for a in _into(marks, b, CIRCLE):
            if marks[a, b] != CIRCLE:  # need a o-o b
                continue
            a_circle = [c for c in range(n) if c not in (a, b) and _traverse_circle(marks, c, a)]
            b_circle = [d for d in range(n) if d not in (a, b) and _traverse_circle(marks, d, b)]
            for c in a_circle:
                if _adjacent(marks, b, c):
                    continue
                for d in b_circle:
                    if _adjacent(marks, a, d):
                        continue
                    for path in _uncovered_circle_paths(marks, c, d, (a, b), ambiguous):
                        if not _is_uncovered(marks, [a, *path, b]):
                            continue
                        # Include both triples crossing the closing edge a-b.
                        full_path = [a, *path, b, a, path[0]]
                        if any(
                            _triple(*full_path[i : i + 3]) in ambiguous
                            for i in range(len(full_path) - 2)
                        ):
                            continue
                        change = True
                        _orient_tails(marks, a, b, path)
    return change


def _orient_tails(marks: np.ndarray, a: int, b: int, path: list[int]) -> None:
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


def _rule_r6(marks: np.ndarray, change: bool) -> bool:
    """R6: ``a --- b o-* c``  =>  tail at ``b`` on edge ``b-c``."""
    n = marks.shape[0]
    for b in range(n):
        if not any(marks[b, a] == TAIL and marks[a, b] == TAIL for a in range(n)):
            continue
        for c in _into(marks, b, CIRCLE):
            marks[b, c] = TAIL
            change = True
    return change


def _rule_r7(
    marks: np.ndarray, change: bool, ambiguous: frozenset[tuple[int, int, int]] = frozenset()
) -> bool:
    """R7: ``a --o b o-* c`` with ``a``, ``c`` non-adjacent  =>  tail at ``b`` on edge ``b-c``."""
    n = marks.shape[0]
    for b in range(n):
        into_circles = _into(marks, b, CIRCLE)
        tail_side = [a for a in into_circles if marks[a, b] == TAIL]  # a --o b
        for c in into_circles:
            for a in tail_side:
                if not _adjacent(marks, a, c) and _triple(a, b, c) not in ambiguous:
                    marks[b, c] = TAIL
                    change = True
    return change


def _rule_r8(marks: np.ndarray, change: bool) -> bool:
    """R8: ``a -> b -> c`` (or ``a --o b -> c``) with ``a o-> c``  =>  ``a -> c``."""
    n = marks.shape[0]
    for b in range(n):
        adj = _adjacent_nodes(marks, b)
        for i in range(len(adj)):
            for k in range(i + 1, len(adj)):
                # Test only ascending endpoint pairs (x = adj[i], z = adj[k], i < k).
                x, z = adj[i], adj[k]
                b_to_z = marks[z, b] == ARROW and marks[b, z] == TAIL
                x_to_b_directed = marks[b, x] == ARROW and marks[x, b] == TAIL
                x_to_b_circle = marks[b, x] == CIRCLE and marks[x, b] == TAIL
                x_o_arrow_z = (
                    _adjacent(marks, x, z) and marks[z, x] == ARROW and marks[x, z] == CIRCLE
                )
                if (x_to_b_directed or x_to_b_circle) and b_to_z and x_o_arrow_z:
                    marks[x, z] = TAIL
                    marks[z, x] = ARROW
                    change = True
    return change


def _pd_reaches_target(marks: np.ndarray, target: int, a: int) -> set[int]:
    """Nodes with a possibly-directed walk to ``target`` that avoids ``a``."""
    reachable = {target}
    queue = deque([target])
    while queue:
        node = queue.popleft()
        for previous in _adjacent_nodes(marks, node):
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
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
) -> bool:
    """True when an uncovered possibly-directed simple path ``a, first, ..., target`` exists.

    Each step has no arrowhead at its near end and no tail at its far end. Search state belongs
    to each path: reaching a node through one predecessor must not block a different route.
    Reverse reachability prunes branches with no p.d. walk to the target avoiding ``a``.
    Ambiguity checks cover interior triples of the open path, with no wraparound at its target.
    """
    if marks[a, first] not in (TAIL, CIRCLE) or marks[first, a] == TAIL:
        return False
    reachable = _pd_reaches_target(marks, target, a)
    if first not in reachable:
        return False
    stack = [[a, first]]
    while stack:
        path = stack.pop()
        last = path[-1]
        if last == target:
            return True
        for node in _adjacent_nodes(marks, last):
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
    marks: np.ndarray, change: bool, ambiguous: frozenset[tuple[int, int, int]] = frozenset()
) -> bool:
    """R9: ``a o-> c`` with an uncovered possibly-directed path ``a .. b .. c``  =>  ``a -> c``."""
    n = marks.shape[0]
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
                if _exists_uncovered_pd_path(marks, a, b, c, ambiguous):
                    marks[a, c] = TAIL
                    marks[c, a] = ARROW
                    change = True
                    break
    return change


def _is_possible_child(marks: np.ndarray, parent: int, child: int) -> bool:
    """True when ``parent`` may be a parent of ``child`` (adjacent, no arrowhead at ``parent``)."""
    if parent == child or not _adjacent(marks, parent, child):
        return False
    return bool(marks[parent, child] != ARROW)


def _rule_r10(
    marks: np.ndarray, change: bool, ambiguous: frozenset[tuple[int, int, int]] = frozenset()
) -> bool:
    """R10: complete ``a o-> c`` via two uncovered possibly-directed paths into ``b -> c <- d``.

    The paths reach the tails ``b`` and ``d`` from two non-adjacent possible children of ``a``.
    """
    n = marks.shape[0]
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
                    if _r10_connects(marks, children, b, d, a, ambiguous):
                        marks[a, c] = TAIL
                        marks[c, a] = ARROW
                        change = True
    return change


def _r10_connects(
    marks: np.ndarray,
    children: list[int],
    b: int,
    d: int,
    a: int,
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
) -> bool:
    """True when uncovered p.d. paths from ``a`` reach ``b``/``d`` via non-adjacent children.

    Either child of a pair may lead to either tail. Each child is searched at most once per tail.
    """
    reaches: dict[tuple[int, int], bool] = {}

    def reach(child: int, target: int) -> bool:
        if (child, target) not in reaches:
            reaches[child, target] = _exists_uncovered_pd_path(marks, a, child, target, ambiguous)
        return reaches[child, target]

    for one, two in combinations(children, 2):
        if _adjacent(marks, one, two) or _triple(one, a, two) in ambiguous:
            continue
        if (reach(one, b) and reach(two, d)) or (reach(one, d) and reach(two, b)):
            return True
    return False
