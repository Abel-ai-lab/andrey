"""GFCI, the engine behind :func:`andrey.gfci`.

The algorithm, its settings, and its references are described on :func:`andrey.gfci`.

The steps follow Algorithm 1 of Ogarrio, Spirtes, and Ramsey (2016). GES returns a CPDAG, the
pattern. FCI's adjacency search, started from the pattern's adjacencies instead of a complete graph,
removes the pairs a subset of either endpoint's neighbors separates. An unshielded triple of the
result is a collider when it is one in the pattern, or, when the pattern shields it, when the
separating set of its endpoints leaves out its middle node. The possible-d-separation and R0-R10
passes of :mod:`andrey.constraint.fci` follow, with its rule implementations,
conditional-independence tests, and mark convention.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from itertools import combinations, islice
from typing import TYPE_CHECKING

import numpy as np

from andrey.constraint._skeleton import _CHUNK, _canonical_chunk, _canonical_route
from andrey.constraint.fci import (
    _adjacent,
    _arrow_allowed,
    _circle_pag,
    _fast_adjacency_search,
    _fci_orient,
    _is_def_collider,
    _knowledge_marks,
    _largest_size,
    _majority_sepsets,
    _neighbor_lists,
    _remove_by_possible_dsep,
    _sepset,
    _test_in_rank_order,
    _triple,
    _validate_collider_rule,
    _validate_possible_dsep,
)
from andrey.core import ARROW, NULL, GraphStructure, backend
from andrey.core.ci import CITest, make_indep_test
from andrey.core.orient import from_structure
from andrey.search.ges import ges

if TYPE_CHECKING:
    from andrey.core.knowledge import Constraints


def gfci(
    data: np.ndarray,
    *,
    score_func: str = "local_score_BIC",
    lambda_value: float = 1.0,
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
    """Discover a PAG with GES adjacency and FCI orientation.

    ``categorical``, a boolean mask of the categorical columns, reaches a mixed-data score and test.
    ``possible_dsep`` (``"order_free"`` or ``"fast"``) selects the order the possible-d-sep step
    visits the nodes in: the :func:`~andrey.search.hc.tie_ranks` order or the column order.
    ``in_column_order`` applies the orientation rules one firing at a time in column order instead
    of in order-free rounds (:func:`~andrey.constraint.fci._fci_orient`).
    ``max_depth`` caps the size of every conditioning set a test uses, as in
    :func:`~andrey.constraint.fci.fci`, and of the separating sets :func:`_derive_sepsets` searches.
    ``constraints`` reach the GES phase, and the test phase honors them as FCI does: no step tests
    or removes a required pair, and their endpoints are set before the orientation rules. GES never
    joins a pair they make non-adjacent, so its colliders over that pair come from the pattern, and
    its separating set from :func:`_derive_sepsets`, as for any pair GES leaves non-adjacent.
    ``workers`` worker processes share the GES passes and the adjacency search's larger
    conditioning-size passes, each above its own work threshold; ``None`` reads
    ``andrey.config.num_workers``.
    """
    X = np.asarray(data, dtype=np.float64)
    workers = backend.worker_count(workers)

    _validate_collider_rule(collider_rule)
    _validate_possible_dsep(possible_dsep)
    cpdag_structure, _score = ges(
        X,
        score_func=score_func,
        lambda_value=lambda_value,
        categorical=categorical,
        constraints=constraints,
        workers=workers,
    )
    pattern = from_structure(cpdag_structure)

    test = make_indep_test(indep_test, X, categorical=categorical)
    keep = None if constraints is None else constraints.required | constraints.required.T
    support, sepsets = _fast_adjacency_search(
        X, test, alpha, workers=workers, support=pattern != NULL, keep=keep, max_depth=max_depth
    )
    sepsets.update(
        _derive_sepsets(pattern, test, alpha, max_depth=max_depth, in_column_order=in_column_order)
    )

    marks = _circle_pag(support)
    _knowledge_marks(marks, constraints)
    _orient_colliders_from_pattern(marks, pattern, sepsets)
    if possible_dsep == "fast":
        _remove_by_possible_dsep(marks, test, alpha, sepsets, keep=keep, max_depth=max_depth)
    else:
        by_rank, ranked_test = _test_in_rank_order(X, indep_test, categorical)
        _remove_by_possible_dsep(
            marks, ranked_test, alpha, sepsets, by_rank, keep=keep, max_depth=max_depth
        )
    marks = _circle_pag(marks != NULL)
    _knowledge_marks(marks, constraints)
    ambiguous = (
        frozenset(
            triple
            for triple in _majority_sepsets(marks, test, alpha, sepsets, max_depth=max_depth)
            if _adjacent(pattern, triple[0], triple[2])
        )
        if collider_rule == "majority"
        else frozenset()
    )
    _orient_colliders_from_pattern(marks, pattern, sepsets, ambiguous)
    _fci_orient(
        marks, test, alpha, sepsets, ambiguous, max_depth=max_depth, in_column_order=in_column_order
    )

    return GraphStructure.from_numpy(marks, kind="pag")


def _orient_colliders_from_pattern(
    marks: np.ndarray,
    pattern: np.ndarray,
    sepsets: Mapping[tuple[int, int], tuple[int, ...]],
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
) -> None:
    """Orient the unshielded colliders ``a *-> b <-* c`` of ``marks`` with the GES pattern.

    For each node ``b`` and each non-adjacent pair ``(a, c)`` of its neighbors (visited in
    ascending order): when ``a`` and ``c`` are not adjacent in ``pattern``, ``b`` is a collider iff
    the pattern has ``a -> b <- c``; when they are, ``b`` is a collider iff a separating set for
    ``a`` and ``c`` is recorded and does not contain ``b``, and the triple is not ``ambiguous``. An
    already-directed tail at ``b`` blocks the orientation.
    """
    for b, adj in enumerate(_neighbor_lists(marks)):
        for i in range(len(adj)):
            for k in range(i + 1, len(adj)):
                a, c = adj[i], adj[k]
                if _adjacent(marks, a, c) or _is_def_collider(marks, a, b, c):
                    continue
                if _adjacent(pattern, a, c):
                    if _triple(a, b, c) in ambiguous:
                        continue
                    sep = _sepset(sepsets, a, c)
                    if sep is None or b in sep:
                        continue
                elif not (pattern[b, a] == ARROW and pattern[b, c] == ARROW):
                    continue
                if not _arrow_allowed(marks, a, b) or not _arrow_allowed(marks, c, b):
                    continue
                marks[b, a] = ARROW
                marks[b, c] = ARROW


def _derive_sepsets(
    cpdag: np.ndarray,
    test: CITest,
    alpha: float,
    *,
    max_depth: int | None = None,
    in_column_order: bool = False,
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Separating sets for every non-adjacent CPDAG pair, from the pooled two-endpoint adjacency.

    For each non-adjacent ``(x, y)`` (``x < y``, row-major), searches subsets of
    ``adj(x) | adj(y) - {x, y}``, of at most ``max_depth`` members when one is given, by increasing
    size and records, both ways, the union of every subset of the first size that makes the pair
    independent, as the adjacency search records its sets; a pair with no such subset gets no
    entry. The union does not depend on the order of the tests. With ``in_column_order``, the first
    subset found in column order is recorded instead, and no subset after it is tested. The
    orientation rules that read a separating set use these for the pairs GES left non-adjacent.

    A FisherZ or StudentT test evaluates each nonempty size's subsets in chunks of canonical query
    arrays (:func:`_separating_subsets`), with the p-values of the scalar calls; the size-0 query
    and every query of another test, or of ``in_column_order``, is a scalar call.
    """
    adjacent = cpdag != NULL
    canonical = None if in_column_order else _canonical_route(test)
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {}
    xs, ys = np.nonzero(np.triu(~adjacent, 1))
    for x, y in zip(xs.tolist(), ys.tolist()):
        pool = [v for v in np.flatnonzero(adjacent[x] | adjacent[y]).tolist() if v != x and v != y]
        for size in range(_largest_size(len(pool), max_depth) + 1):
            subsets = combinations(pool, size)
            if in_column_order:
                first = next((cond for cond in subsets if test(x, y, cond) > alpha), None)
                found = [] if first is None else [first]
            elif canonical is None or size == 0:
                found = [cond for cond in subsets if test(x, y, cond) > alpha]
            else:
                found = _separating_subsets(canonical, test, x, y, size, subsets, alpha)
            if found:
                union = tuple(sorted(set().union(*found)))
                sepsets[(x, y)] = union
                sepsets[(y, x)] = union
                break
    return sepsets


def _separating_subsets(
    canonical: Callable[..., np.ndarray],
    test: CITest,
    x: int,
    y: int,
    size: int,
    subsets: Iterator[tuple[int, ...]],
    alpha: float,
) -> list[tuple[int, ...]]:
    """The ``subsets`` (sorted, distinct, of ``size`` members) whose test separates ``x < y``.

    The subsets are evaluated in chunks of ``_CHUNK`` canonical queries, in order. A chunk whose
    evaluation raises is tested again subset by subset with scalar ``test`` calls, so the error is
    the one the scalar loop raises, after the same queries.
    """
    found: list[tuple[int, ...]] = []
    while chunk := list(islice(subsets, _CHUNK)):
        try:
            pvalues = _canonical_chunk(canonical, size, x, y, chunk)
        except Exception:
            pvalues = [test(x, y, cond) for cond in chunk]
        found.extend(cond for cond, p in zip(chunk, pvalues) if p > alpha)
    return found
