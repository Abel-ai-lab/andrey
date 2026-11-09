"""FCI (Fast Causal Inference) causal discovery, the engine behind :func:`andrey.fci`.

The algorithm, its settings, and its references are described on :func:`andrey.fci`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from contextlib import nullcontext
from functools import cache
from itertools import combinations, islice, permutations
from math import comb
from typing import TYPE_CHECKING

import numpy as np

from andrey.constraint._parallel_skeleton import SkeletonWorkerPool
from andrey.constraint._skeleton import (
    CHUNK,
    MASK_BITS,
    blocks,
    canonical_capable,
    check_collider_rule,
    first_separating_subset,
    unseparated_triples,
    vote_triples,
)
from andrey.constraint.pc import adjacency_search, check_knowledge_cycle, knowledge_masks
from andrey.core import ARROW, CIRCLE, NULL, TAIL, GraphStructure, backend
from andrey.core._bitset import row_masks, semidirected_reach_set
from andrey.core.ci import CITest, make_indep_test, pins_blas
from andrey.core.knowledge import KnowledgeError
from andrey.core.orient import separating_set
from andrey.core.ties import tie_mask, tie_ranks

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
    seed: int | None = None,
    _ambiguous: set[tuple[int, int, int]] | None = None,
) -> GraphStructure:
    """Discover a PAG from ``data`` with the FCI algorithm and the ``indep_test`` CI test.

    ``data`` is an ``(n_samples, n_features)`` array; ``alpha`` is the independence-test
    significance level in ``(0, 1)``. Skeleton pruning requires a p-value above ``alpha``;
    the conservative and majority collider rules also count equality as a separating vote.
    ``collider_rule`` (``"sepsets"``, ``"conservative"``, or ``"majority"``) decides the unshielded
    triples after the possible-d-sep step: by the recorded separating sets, or by a vote over the
    subsets of either endpoint's neighbors (:func:`~andrey.constraint._skeleton.vote_triples`),
    which leaves some triples ambiguous.
    ``categorical``, a boolean mask of the categorical columns, reaches a mixed-data test.
    ``possible_dsep`` (``"order_free"`` or ``"fast"``) selects the order the possible-d-sep step
    visits the nodes in: the :func:`~andrey.core.ties.tie_ranks` order or the column order.
    ``in_column_order`` applies the orientation rules one firing at a time in column order instead
    of in order-free rounds (:func:`fci_orient`).
    ``max_depth`` caps the size of every conditioning set a test uses: the adjacency search, the
    possible-d-sep subsets, the collider votes, and the discriminating-path tests. With
    ``constraints``, the adjacency search starts from the pairs they allow, no step tests or removes
    a required pair, a pair they make non-adjacent gets a separating set when the search finds one
    (:func:`~andrey.constraint.pc.adjacency_search`), and :func:`knowledge_marks` sets their
    endpoints before the orientation rules, which change circles only. A triple over a non-adjacent
    pair with no separating set is neither a collider nor a non-collider to any rule
    (:func:`~andrey.constraint._skeleton.unseparated_triples`). An arrowhead the knowledge puts at
    ``x`` on an edge ``x - y`` while its required edges lead from ``x`` to ``y`` raises
    ``ValueError`` (:func:`possible_dsep_step`). ``workers`` worker processes share the adjacency
    search's larger conditioning-size passes (:func:`~andrey.constraint.pc.adjacency_search`) and
    the larger possible-d-sep searches (:func:`possible_dsep_step`); ``None`` reads
    ``andrey.config.n_jobs``. ``seed`` fixes the draws of a stochastic
    ``indep_test`` (:data:`~andrey.core.ci.STOCHASTIC_INDEP_TESTS`); ``None`` uses
    ``ANDREY_SEED``. Returns the PAG as a ``GraphStructure`` of kind ``"pag"`` (circle marks and
    all). The private ``_ambiguous`` sink receives the triples no orientation rule reads as a
    collider or a non-collider, keyed ``(a, b, c)`` with ``a < c``.
    """
    X = np.asarray(data, dtype=np.float64)
    workers = backend.worker_count(workers)

    check_collider_rule(collider_rule)
    validate_possible_dsep(possible_dsep)
    test = make_indep_test(indep_test, X, categorical=categorical, seed=seed)
    allowed, keep, apart = knowledge_masks(constraints)
    skeleton, sepsets, _ = adjacency_search(
        test,
        X.shape[1],
        alpha,
        support=allowed,
        keep=keep,
        apart=apart,
        max_depth=max_depth,
        workers=workers,
    )

    marks = circle_pag(skeleton)
    knowledge_marks(marks, constraints)
    orient_colliders(marks, sepsets)
    marks = possible_dsep_step(
        marks,
        X,
        test,
        alpha,
        sepsets,
        possible_dsep=possible_dsep,
        indep_test=indep_test,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
        constraints=constraints,
        workers=workers,
    )
    ambiguous = unseparated_triples(marks != NULL, sepsets, constraints)
    if collider_rule != "sepsets":
        ambiguous |= vote_triples(
            marks, test, alpha, sepsets, collider_rule, max_depth=max_depth, skip=ambiguous
        )
    if _ambiguous is not None:
        _ambiguous.update(ambiguous)
    orient_colliders(marks, sepsets, ambiguous)
    fci_orient(
        marks, test, alpha, sepsets, ambiguous, max_depth=max_depth, in_column_order=in_column_order
    )

    return GraphStructure.from_numpy(marks, kind="pag")


# ---- mark-matrix helpers ------------------------------------------------------------------------
# ``marks[i, j]`` is the mark at node ``i`` on edge ``i-j``, as in :mod:`andrey.core.structure`.


def circle_pag(support: np.ndarray) -> np.ndarray:
    """An all-circle PAG over the given adjacency support (every present edge becomes ``o-o``)."""
    marks = np.where(support != NULL, np.int8(CIRCLE), np.int8(NULL))
    np.fill_diagonal(marks, np.int8(NULL))
    return marks


def knowledge_marks(marks: np.ndarray, constraints: Constraints | None) -> None:
    """Set the endpoints ``constraints`` decide on the adjacent pairs, in place.

    A forbidden ``i -> j`` whose reverse is not forbidden puts an arrowhead at ``i`` (the MAG has no
    ``i -> j``), and a required ``i -> j`` is ``i -> j``: a tail at ``i`` and, as its reverse is
    forbidden, an arrowhead at ``j``. A non-ancestor claim ``(i, j)`` puts an arrowhead at ``i``,
    and an ancestor claim ``(i, j)`` makes the edge ``i -> j``: in a maximal ancestral graph an
    edge whose end ``i`` is an ancestor of ``j`` has a tail at ``i`` and an arrowhead at ``j``
    (Richardson and Spirtes 2002). An ancestor claim on an edge the forbidden edges or the tiers put
    an arrowhead at its cause raises :class:`~andrey.core.knowledge.KnowledgeError`. ``None``
    leaves ``marks`` as they are.
    """
    if constraints is None:
        return
    adjacent = marks != NULL
    marks[adjacent & constraints.forbidden & ~constraints.forbidden.T] = ARROW
    if constraints.non_ancestor is not None:
        marks[adjacent & constraints.non_ancestor] = ARROW
    marks[adjacent & constraints.required] = TAIL
    if constraints.ancestor is not None:
        claimed = adjacent & constraints.ancestor
        if (clash := claimed & (marks == ARROW)).any():
            i, j = np.argwhere(clash)[0].tolist()
            raise KnowledgeError(
                f"knowledge lists ({i}, {j}) in ancestors, but forbids the edge {i} -> {j} one way "
                f"or by the tiers, which puts an arrowhead at {i} on the edge it keeps between them"
            )
        marks[claimed] = TAIL
        marks[claimed.T] = ARROW


def check_path_claims(
    marks: np.ndarray,
    ancestor: np.ndarray | None,
    non_ancestor: np.ndarray | None,
    label: Callable[[int], str] = str,
) -> None:
    """Raise :class:`~andrey.core.knowledge.KnowledgeError` when the PAG ``marks`` contradict a
    path claim.

    ``ancestor[a, b]`` claims that ``a`` is an ancestor of ``b``, which needs a possibly directed
    path from ``a`` to ``b``: one whose every edge ``u *-* v`` is not into ``u`` and not out of
    ``v`` (Zhang 2008). Without one, ``a`` is an ancestor of ``b`` in no maximal ancestral graph the
    PAG stands for. ``non_ancestor[a, b]`` claims the opposite, which a directed path
    ``a -> ... -> b`` denies in every such graph. ``label`` names a node in the message.
    """
    for claims, steps, holds, what in (
        (ancestor, (marks != NULL) & (marks != ARROW) & (marks.T != TAIL), False, "no possibly"),
        (non_ancestor, (marks == TAIL) & (marks.T == ARROW), True, "a"),
    ):
        if claims is None or not claims.any():
            continue
        for a in np.flatnonzero(claims.any(axis=1)).tolist():
            reached = reach(steps, a)
            if (found := claims[a] & (reached == holds)).any():
                b = int(np.flatnonzero(found)[0])
                part = "ancestors" if claims is ancestor else "non_ancestors"
                raise KnowledgeError(
                    f"knowledge lists ({label(a)}, {label(b)}) in {part}, but the PAG has {what} "
                    f"directed path from {label(a)} to {label(b)}"
                )


def reach(steps: np.ndarray, source: int) -> np.ndarray:
    """The nodes a path of one or more ``steps[u, v]`` leads to from ``source``, as a mask."""
    reached = np.zeros(len(steps), dtype=bool)
    frontier = steps[source].copy()
    while frontier.any():
        reached |= frontier
        frontier = steps[frontier].any(axis=0) & ~reached
    return reached


def _pattern_decides(pattern: np.ndarray, a: int, b: int, c: int) -> bool:
    """Whether ``pattern`` holds the triple ``a - b - c`` with ``a`` and ``c`` non-adjacent."""
    return not _adjacent(pattern, a, c) and _adjacent(pattern, a, b) and _adjacent(pattern, c, b)


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
    return np.flatnonzero(marks[b] == mark).tolist()


def _largest_size(candidates: int, max_depth: int | None) -> int:
    """The largest conditioning-set size drawn from ``candidates`` nodes under ``max_depth``."""
    return candidates if max_depth is None else min(candidates, max_depth)


def _fits(cond: list[int], max_depth: int | None) -> bool:
    """Whether ``cond`` has at most ``max_depth`` members (always, when ``max_depth`` is None)."""
    return max_depth is None or len(cond) <= max_depth


# ---- R0: unshielded colliders -------------------------------------------------------------------


def validate_possible_dsep(possible_dsep: str) -> None:
    if possible_dsep not in ("order_free", "fast"):
        raise ValueError(f"possible_dsep must be 'order_free' or 'fast', got {possible_dsep!r}")


def _triple(a: int, b: int, c: int) -> tuple[int, int, int]:
    """Canonical key for a triple with middle node ``b``."""
    return (a, b, c) if a < c else (c, b, a)


def orient_colliders(
    marks: np.ndarray,
    sepsets: Mapping[tuple[int, int], tuple[int, ...]],
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
    pattern: np.ndarray | None = None,
) -> None:
    """Orient every unshielded collider ``a *-> b <-* c`` from the separating sets.

    For each node ``b`` and each non-adjacent pair ``(a, c)`` of its neighbors (visited in
    ascending order), ``b`` is a collider iff a separating set for ``a`` and ``c`` is recorded and
    does not contain ``b``, and the triple is not ``ambiguous``. With ``pattern``, the CPDAG of a
    score search, a pair the pattern leaves non-adjacent is decided by the pattern instead: ``b``
    is a collider iff the pattern has ``a -> b <- c``, when the pattern also joins ``b`` to both:
    a triple over an edge the pattern lacks, such as a latent pair the knowledge adds, is decided
    by the separating sets. The endpoint at ``b`` on both edges is then set to an arrowhead; an
    already-directed tail at ``b`` blocks the orientation.
    """
    for b, adj in enumerate(_neighbor_lists(marks)):
        for i in range(len(adj)):
            for k in range(i + 1, len(adj)):
                a, c = adj[i], adj[k]
                if _adjacent(marks, a, c) or _is_def_collider(marks, a, b, c):
                    continue
                if pattern is not None and _pattern_decides(pattern, a, b, c):
                    collider = pattern[b, a] == ARROW and pattern[b, c] == ARROW
                else:
                    sep = separating_set(sepsets, a, c)
                    collider = (
                        _triple(a, b, c) not in ambiguous and sep is not None and b not in sep
                    )
                if (
                    not collider
                    or not _arrow_allowed(marks, a, b)
                    or not _arrow_allowed(marks, c, b)
                ):
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


# The most subsets one possible-d-sep run asks a test for at once (see _separating_choice).
_DSEP_RUN = 256


def _iter_dsep_choices(
    dsep: list[int],
    adj_a: set[int],
    adj_b: set[int],
    max_depth: int | None = None,
    tested: Collection[int] = (),
) -> Iterator[tuple[int, ...]]:
    """Every non-empty subset of ``dsep``, of at most ``max_depth`` members when one is given, that
    lies inside neither endpoint's adjacency nor inside ``tested``.

    Subsets come by increasing size, each size in ``combinations`` order, one at a time. The
    adjacency search already tested the empty set and every subset of either endpoint's
    neighbors: those neighborhoods only shrink, so each such subset was a candidate at its size
    and left the pair dependent. Testing it again cannot remove the edge. ``tested`` is the pair's
    other possible-d-sep set when its search found no separating subset: that search tested every
    subset of it outside both adjacencies and found each dependent.
    """
    within = (adj_a, adj_b, tested) if tested else (adj_a, adj_b)
    for size in range(1, _largest_size(len(dsep), max_depth) + 1):
        yield from (c for c in combinations(dsep, size) if _outside(c, within))


def _outside(values: tuple[int, ...], within: Iterable[Collection[int]]) -> bool:
    """``values`` lie inside none of the sets ``within``."""
    return not any(all(v in nodes for v in values) for nodes in within)


# The fewest subsets, counted over every size a search tests, of a search a FisherZ, StudentT, or
# NormalScores test asks as arrays (see _first_separating); a smaller one asks tuples, which costs
# less there.
_ARRAY_MIN_SUBSETS = 2048


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
    pool: SkeletonWorkerPool | None = None,
    tested: Collection[int] = (),
) -> bool:
    """Remove edge ``x-y`` if some subset of ``dsep``, of at most ``max_depth`` members when one is
    given, makes ``x`` and ``y`` independent.

    The subsets are tested in :func:`_iter_dsep_choices` order, less those inside ``tested``, and
    the first that separates the pair is recorded (:func:`_first_separating`).
    """
    found = _first_separating(test, x, y, dsep, adj_a, adj_b, alpha, max_depth, tested, pool)
    if found is None:
        return False
    marks[x, y] = NULL
    marks[y, x] = NULL
    sepsets[(x, y)] = tuple(int(v) for v in found)
    return True


def _first_separating(
    test: CITest,
    x: int,
    y: int,
    dsep: list[int],
    adj_a: set[int],
    adj_b: set[int],
    alpha: float,
    max_depth: int | None,
    tested: Collection[int],
    pool: SkeletonWorkerPool | None,
) -> tuple[int, ...] | None:
    """The first subset in :func:`_iter_dsep_choices` order that separates ``x`` and ``y``, or
    ``None``.

    With a FisherZ, StudentT, or NormalScores test
    (:func:`~andrey.constraint._skeleton.canonical_capable`) and at most
    :data:`~andrey.constraint._skeleton.MASK_BITS` candidates, a search whose subsets, counted over
    every size it tests, number at least the ``pool``'s cutoff
    (:meth:`~andrey.constraint._parallel_skeleton.SkeletonWorkerPool.takes_search`, which warns
    once on the default cutoff) runs on the pool
    (:meth:`~andrey.constraint._parallel_skeleton.SkeletonWorkerPool.first_separating`),
    and one of at least ``_ARRAY_MIN_SUBSETS`` enumerates its subsets as arrays in this process
    (:func:`~andrey.constraint._skeleton.first_separating_subset`). Another search asks them as
    tuples (:func:`_separating_choice`). Every path tests the subsets in the same order and stops
    at the first that separates the pair; on the pool, tests after it may run, but the subset found
    is the serial one.
    """
    sizes = range(1, _largest_size(len(dsep), max_depth) + 1)
    if canonical_capable(test) and len(dsep) <= MASK_BITS:
        subsets = sum(comb(len(dsep), size) for size in sizes)
        within = tuple(
            sum(1 << i for i, v in enumerate(dsep) if v in nodes)
            for nodes in (adj_a, adj_b, tested)
        )
        if pool is not None and pool.takes_search(subsets):
            return pool.first_separating(x, y, dsep, within, sizes, alpha)
        if subsets >= _ARRAY_MIN_SUBSETS:
            order = (block for size in sizes for block in blocks(len(dsep), size, CHUNK))
            return first_separating_subset(test, x, y, dsep, within, order, alpha)
    choices = _iter_dsep_choices(dsep, adj_a, adj_b, max_depth, tested)
    return _separating_choice(test, x, y, choices, alpha)


def _separating_choice(
    test: CITest, x: int, y: int, choices: Iterator[tuple[int, ...]], alpha: float
) -> tuple[int, ...] | None:
    """The first of ``choices`` given which ``x`` and ``y`` test independent, or ``None``.

    A test with ``separating_prefix`` answers them in runs of 1, 2, 4, ... up to ``_DSEP_RUN``
    subsets, each run cut at its first separating subset, so it answers the queries the
    one-at-a-time loop asks, in its order, and no more; another test takes one call per subset.
    """
    prefix = getattr(test, "separating_prefix", None)
    if prefix is None:
        return next((cond for cond in choices if test(x, y, cond) > alpha), None)
    found, run = None, 1
    while found is None and (chunk := list(islice(choices, run))):
        pvalues = prefix(x, y, chunk, alpha)
        if pvalues and pvalues[-1] > alpha:
            found = chunk[len(pvalues) - 1]
        run = min(2 * run, _DSEP_RUN)
    return found


def _edge_order(marks: np.ndarray, i: int, j: int) -> tuple[int, int]:
    """Return the endpoint order used by the possible-d-sep search.

    A left-pointing edge uses ``(j, i)``. Every other edge uses ``(i, j)``, so its search order
    follows the node indices.
    """
    if marks[i, j] == ARROW and marks[j, i] in (TAIL, CIRCLE):
        return j, i
    return i, j


def possible_dsep_step(
    marks: np.ndarray,
    X: np.ndarray,
    test: CITest,
    alpha: float,
    sepsets: dict[tuple[int, int], tuple[int, ...]],
    *,
    possible_dsep: str,
    indep_test: str,
    categorical: np.ndarray | None,
    seed: int | None,
    max_depth: int | None,
    constraints: Constraints | None,
    keep: np.ndarray | None = None,
    workers: int = 1,
) -> np.ndarray:
    """Remove the pairs a possible-d-sep subset separates; return the PAG re-circled for the
    orientation rules.

    The step FCI, FCI-JCI, and GFCI run after their first collider pass, on ``marks`` in place
    (:func:`_remove_by_possible_dsep`). ``possible_dsep="fast"`` visits the nodes in column order
    with ``test``; ``"order_free"`` visits them in the :func:`~andrey.core.ties.tie_ranks` order,
    with an ``indep_test`` test over ``X``'s columns in that order (:func:`_test_in_rank_order`).
    The pairs the required edges of ``constraints`` join, and those marked in the boolean ``keep``
    (both ways), are never tested. ``workers`` worker processes share the larger searches. Every
    kept edge then becomes ``o-o``, and
    :func:`knowledge_marks` sets the endpoints ``constraints`` decide. An arrowhead the knowledge
    puts at ``x`` on a kept edge ``x - y`` while its required edges lead from ``x`` to ``y`` raises
    :class:`~andrey.core.knowledge.KnowledgeError`
    (:func:`~andrey.constraint.pc.check_knowledge_cycle`).
    """
    required = knowledge_masks(constraints)[1]
    if required is not None:
        keep = required if keep is None else keep | required
    if possible_dsep == "fast":
        _remove_by_possible_dsep(
            marks, test, alpha, sepsets, keep=keep, max_depth=max_depth, workers=workers
        )
    else:
        by_rank, ranked_test = _test_in_rank_order(X, indep_test, categorical, seed)
        _remove_by_possible_dsep(
            marks,
            ranked_test,
            alpha,
            sepsets,
            by_rank,
            keep=keep,
            max_depth=max_depth,
            workers=workers,
        )
    marks = circle_pag(marks != NULL)
    knowledge_marks(marks, constraints)
    if constraints is not None:
        check_knowledge_cycle((marks == TAIL) & (marks.T == ARROW), closing=(marks == ARROW).T)
    return marks


def _remove_by_possible_dsep(
    marks: np.ndarray,
    test: CITest,
    alpha: float,
    sepsets: dict[tuple[int, int], tuple[int, ...]],
    by_rank: np.ndarray | None = None,
    *,
    keep: np.ndarray | None = None,
    max_depth: int | None = None,
    workers: int = 1,
) -> None:
    """Prune edges the plain skeleton kept but a possible-d-sep conditioning set separates.

    Visits the pairs one at a time and searches each endpoint's possible-d-sep set for a
    conditioning subset outside either endpoint's adjacency, of at most ``max_depth`` members when
    one is given; the first subset found removes the pair at once, so each removal changes the sets
    of the pairs after it, and is stored under the endpoint order returned by ``_edge_order``. A
    pair marked in the boolean ``keep`` (the required edges and, under GFCI's
    ``edge_removal="triangles"``, the edges in no triangle of its CPDAG; both ways) is never tested.
    The pairs, their endpoints, and the nodes of each set are visited in column order. ``by_rank``,
    when given, is the column order of ``test``'s data, as :func:`_test_in_rank_order` returns them:
    the step then runs on the nodes in that order, with ``keep`` permuted the same way, and maps its
    removals and separating sets back to the column labels once.

    When the first endpoint's search finds no separating subset, the second endpoint's leaves out
    the subsets of the first endpoint's set: each was tested there and left the pair dependent.

    With ``workers > 1`` and a FisherZ, StudentT, or NormalScores test
    (:func:`~andrey.constraint._skeleton.canonical_capable`), a pair's search whose subsets, counted
    over every size it tests, reach
    :data:`~andrey.constraint._parallel_skeleton.MIN_POOL_DSEP_SUBSETS`, or the
    ``ANDREY_POSSIBLE_DSEP_PARALLEL_MIN_WORK`` that overrides it, runs on that many worker processes
    (:func:`_first_separating`), started on the first such search. The pairs are still visited one
    at a time, in the same order: a removal changes the possible-d-sep sets and adjacencies of the
    pairs after it, so their searches cannot run at once. The workers test a pair's subsets in the
    serial order and the first that separates it removes it, so the removals, their order, and the
    separating sets are the serial ones. The step runs with BLAS pinned to one thread, as on the
    workers, for a test built on matrix algebra (:func:`~andrey.core.ci.pins_blas`).
    """
    pool = SkeletonWorkerPool(test, workers) if workers > 1 and canonical_capable(test) else None
    try:
        with backend.one_blas_thread() if pins_blas(test) else nullcontext():
            if by_rank is None:
                _remove_in_node_order(marks, test, alpha, sepsets, keep, max_depth, pool)
                return
            ranked = marks[np.ix_(by_rank, by_rank)]
            ranked_keep = None if keep is None else keep[np.ix_(by_rank, by_rank)]
            found: dict[tuple[int, int], tuple[int, ...]] = {}
            _remove_in_node_order(ranked, test, alpha, found, ranked_keep, max_depth, pool)
    finally:
        if pool is not None:
            pool.close()
    position = np.argsort(by_rank)
    marks[:] = ranked[np.ix_(position, position)]
    label = by_rank.tolist()
    for (x, y), cond in found.items():
        sepsets[(label[x], label[y])] = tuple(label[v] for v in cond)


def _test_in_rank_order(
    X: np.ndarray, indep_test: str, categorical: np.ndarray | None, seed: int | None = None
) -> tuple[np.ndarray, CITest]:
    """The column order that sorts ``X`` by :func:`~andrey.core.ties.tie_ranks`, and the
    ``indep_test`` test, seeded by ``seed``, over ``X``'s columns in that order.

    The columns the test reads as categories rank as categories
    (:func:`~andrey.core.ties.tie_mask`), so relabeling a column's categories keeps the order. Each
    rank moves with its column, so every column order of ``X`` gives the same test.
    """
    by_rank = np.argsort(tie_ranks(X, tie_mask(indep_test, categorical, X.shape[1])))
    mask = None if categorical is None else np.asarray(categorical)[by_rank]
    return by_rank, make_indep_test(
        indep_test, np.ascontiguousarray(X[:, by_rank]), categorical=mask, seed=seed
    )


def _remove_in_node_order(
    marks: np.ndarray,
    test: CITest,
    alpha: float,
    sepsets: dict[tuple[int, int], tuple[int, ...]],
    keep: np.ndarray | None,
    max_depth: int | None,
    pool: SkeletonWorkerPool | None = None,
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
        tested: set[int] = set()
        for a, b in ((na, nb), (nb, na)):
            dsep = _possible_dsep(marks, a, b, neighbors, reach)
            if _try_remove_by_dsep(
                marks, na, nb, dsep, adj_a, adj_b, test, alpha, sepsets, max_depth, pool, tested
            ):
                neighbors, reach = _neighbor_lists(marks), _semidirected_reach(marks)
                break
            tested = set(dsep)


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


def fci_orient(
    marks: np.ndarray,
    test: CITest,
    alpha: float,
    sepsets: Mapping[tuple[int, int], tuple[int, ...]],
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
    *,
    max_depth: int | None = None,
    in_column_order: bool = False,
    selection_bias: bool = True,
) -> None:
    """Apply the FCI orientation rules R1-R10 to a fixed point, in place over ``marks``.

    Circles become tails or arrowheads only where the current marks force it; no rule changes a
    tail or an arrowhead, so one present when the rules start, such as one :func:`knowledge_marks`
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

    ``selection_bias=False`` skips R5-R7, the rules for selection bias, leaving R1-R4 and R8-R10,
    Zhang's complete set for PAGs without selection bias.
    """
    if not in_column_order:
        test = _memoized(test)  # R4 repeats its tests in every round
        while True:
            implied = _Implied()
            _rule_r1r2_cycle(marks, False, ambiguous, out=implied)
            _rule_r3(marks, sepsets, False, ambiguous, out=implied)
            _rule_r4(marks, test, alpha, sepsets, False, max_depth=max_depth, out=implied)
            if selection_bias:
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
        if selection_bias:
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
                    sep = separating_set(sepsets, a, c)
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
        sep = separating_set(sepsets, d, c)
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
    *,
    reaching: dict[tuple[int, int], set[int]] | None = None,
) -> bool:
    """True when an uncovered possibly-directed simple path ``a, first, ..., target`` exists.

    Each step has no arrowhead at its near end and no tail at its far end. Search state belongs
    to each path: reaching a node through one predecessor must not block a different route.
    Reverse reachability prunes branches with no p.d. walk to the target avoiding ``a``.
    Ambiguity checks cover interior triples of the open path, with no wraparound at its target.
    ``neighbors`` holds the :func:`_neighbor_lists` of ``marks``. ``reaching``, when given, keeps
    each reverse reachability set by ``(target, a)`` for later calls on the same marks.
    """
    if marks[a, first] not in (TAIL, CIRCLE) or marks[first, a] == TAIL:
        return False
    reachable = None if reaching is None else reaching.get((target, a))
    if reachable is None:
        reachable = _pd_reaches_target(marks, target, a, neighbors)
        if reaching is not None:
            reaching[target, a] = reachable
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
    """R9: ``a o-> c`` with an uncovered possibly-directed path ``a .. b .. c``  =>  ``a -> c``.

    One reverse reachability set from ``c`` avoiding ``a`` serves every ``b``.
    """
    out = marks if out is None else out
    neighbors = _neighbor_lists(marks)
    for c in range(marks.shape[0]):
        for a in _into(marks, c, ARROW):
            if marks[a, c] != CIRCLE:  # need a o-> c
                continue
            candidates = [v for v in neighbors[a] if v != c and _is_possible_child(marks, a, v)]
            reaching: dict[tuple[int, int], set[int]] = {}
            for b in candidates:
                if _adjacent(marks, b, c):
                    continue
                if _exists_uncovered_pd_path(
                    marks, a, b, c, neighbors, ambiguous, reaching=reaching
                ):
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
    A path search depends on ``a``, the child, and the tail, not on ``c``, so each runs once per
    call, until a mark R10 sets on ``marks`` directly clears the answers.
    """
    out = marks if out is None else out
    neighbors = _neighbor_lists(marks)
    searched: dict[tuple[int, int, int], bool] = {}
    reaching: dict[tuple[int, int], set[int]] = {}

    def reach(a: int, child: int, tail: int) -> bool:
        if (a, child, tail) not in searched:
            searched[a, child, tail] = _exists_uncovered_pd_path(
                marks, a, child, tail, neighbors, ambiguous, reaching=reaching
            )
        return searched[a, child, tail]

    for c in range(marks.shape[0]):
        into_arrows = _into(marks, c, ARROW)
        if len(into_arrows) < 2:
            continue
        a_nodes = [a for a in into_arrows if marks[a, c] == CIRCLE]  # a o-> c
        for a in a_nodes:
            children = [v for v in neighbors[a] if v != c and _is_possible_child(marks, a, v)]
            if len(children) < 2:
                continue
            tails = [t for t in into_arrows if marks[t, c] == TAIL]  # t -> c
            if _r10_connects(marks, children, tails, a, reach, ambiguous):
                out[a, c] = TAIL
                out[c, a] = ARROW
                change = True
                if out is marks:
                    searched.clear()
                    reaching.clear()
    return change


def _r10_connects(
    marks: np.ndarray,
    children: list[int],
    tails: list[int],
    a: int,
    reach: Callable[[int, int, int], bool],
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
) -> bool:
    """True when uncovered p.d. paths from ``a`` reach two of ``tails`` via non-adjacent children.

    Either child of a pair may lead to either tail. ``reach(a, child, tail)`` tells whether an
    uncovered p.d. path ``a, child, ..., tail`` exists; only children in such a pair are searched.
    """
    if len(tails) < 2:
        return False
    pairs = [
        (one, two)
        for one, two in combinations(children, 2)
        if not _adjacent(marks, one, two) and _triple(one, a, two) not in ambiguous
    ]
    paired = sorted({child for pair in pairs for child in pair})
    reached = [{child for child in paired if reach(a, child, tail)} for tail in tails]
    reached = [hits for hits in reached if hits]
    return any(
        (one in to_b and two in to_d) or (one in to_d and two in to_b)
        for to_b, to_d in combinations(reached, 2)
        for one, two in pairs
    )
