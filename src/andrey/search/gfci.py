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

from collections.abc import Mapping
from itertools import combinations

import numpy as np

from andrey.constraint.fci import (
    _adjacent,
    _adjacent_nodes,
    _arrow_allowed,
    _circle_pag,
    _fast_adjacency_search,
    _fci_orient,
    _is_def_collider,
    _majority_sepsets,
    _remove_by_possible_dsep,
    _sepset,
    _test_in_rank_order,
    _triple,
    _validate_collider_rule,
    _validate_possible_dsep,
)
from andrey.core import ARROW, NULL, GraphStructure
from andrey.core.ci import CITest, make_indep_test
from andrey.core.orient import from_structure
from andrey.search.ges import ges


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
    in_column_order: bool = False,
) -> GraphStructure:
    """Discover a PAG with GES adjacency and FCI orientation.

    ``categorical``, a boolean mask of the categorical columns, reaches a mixed-data score and test.
    ``possible_dsep`` (``"order_free"`` or ``"fast"``) selects the order the possible-d-sep step
    visits the nodes in: the :func:`~andrey.search.hc.tie_ranks` order or the column order.
    ``in_column_order`` takes the column order there too, and applies the orientation rules one
    firing at a time in column order instead of in order-free rounds
    (:func:`~andrey.constraint.fci._fci_orient`).
    """
    X = np.asarray(data, dtype=np.float64)

    _validate_collider_rule(collider_rule)
    _validate_possible_dsep(possible_dsep)
    cpdag_structure, _score = ges(
        X, score_func=score_func, lambda_value=lambda_value, categorical=categorical
    )
    pattern = from_structure(cpdag_structure)

    test = make_indep_test(indep_test, X, categorical=categorical)
    support, sepsets = _fast_adjacency_search(X, test, alpha, support=pattern != NULL)
    sepsets.update(_derive_sepsets(pattern, test, alpha, in_column_order=in_column_order))

    marks = _circle_pag(support)
    _orient_colliders_from_pattern(marks, pattern, sepsets)
    if possible_dsep == "fast" or in_column_order:
        _remove_by_possible_dsep(marks, test, alpha, sepsets)
    else:
        by_rank, ranked_test = _test_in_rank_order(X, indep_test, categorical)
        _remove_by_possible_dsep(marks, ranked_test, alpha, sepsets, by_rank)
    marks = _circle_pag(marks != NULL)
    ambiguous = (
        frozenset(
            triple
            for triple in _majority_sepsets(marks, test, alpha, sepsets)
            if _adjacent(pattern, triple[0], triple[2])
        )
        if collider_rule == "majority"
        else frozenset()
    )
    _orient_colliders_from_pattern(marks, pattern, sepsets, ambiguous)
    _fci_orient(marks, test, alpha, sepsets, ambiguous, in_column_order=in_column_order)

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
    n = marks.shape[0]
    for b in range(n):
        adj = _adjacent_nodes(marks, b)
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
    cpdag: np.ndarray, test: CITest, alpha: float, *, in_column_order: bool = False
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Separating sets for every non-adjacent CPDAG pair, from the pooled two-endpoint adjacency.

    For each non-adjacent ``(x, y)`` (``x < y``), searches subsets of ``adj(x) | adj(y) - {x, y}``
    by increasing size and records, both ways, the union of every subset of the first size that
    makes the pair independent, as the adjacency search records its sets; a pair with no such
    subset gets no entry. The union does not depend on the order of the tests. With
    ``in_column_order``, the first subset found in column order is recorded instead. The
    orientation rules that read a separating set use these for the pairs GES left non-adjacent.
    """
    n = cpdag.shape[0]
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {}
    for x, y in combinations(range(n), 2):
        if _adjacent(cpdag, x, y):
            continue
        pool = sorted((set(_adjacent_nodes(cpdag, x)) | set(_adjacent_nodes(cpdag, y))) - {x, y})
        for size in range(len(pool) + 1):
            separating = (cond for cond in combinations(pool, size) if test(x, y, cond) > alpha)
            if in_column_order:
                first = next(separating, None)
                found = [] if first is None else [first]
            else:
                found = list(separating)
            if found:
                union = tuple(sorted(set().union(*found)))
                sepsets[(x, y)] = union
                sepsets[(y, x)] = union
                break
    return sepsets
