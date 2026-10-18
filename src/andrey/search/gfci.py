"""GFCI, the engine behind :func:`andrey.gfci`.

The algorithm, its settings, and its references are described on :func:`andrey.gfci`.

The GES CPDAG's adjacencies, reset to circle marks, go through the collider,
possible-d-separation, and R0-R10 passes of :mod:`andrey.constraint.fci`, with its rule
implementations, conditional-independence tests, and mark convention.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from andrey.constraint.fci import (
    _adjacent,
    _adjacent_nodes,
    _circle_pag,
    _fci_orient,
    _majority_sepsets,
    _orient_colliders,
    _remove_by_possible_dsep,
    _validate_collider_rule,
    _validate_possible_dsep,
)
from andrey.core import NULL, GraphStructure
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
    possible_dsep: str = "fast",
) -> GraphStructure:
    """Discover a PAG with GES adjacency and FCI orientation.

    ``categorical``, a boolean mask of the categorical columns, reaches a mixed-data score and test.
    """
    X = np.asarray(data, dtype=np.float64)

    _validate_collider_rule(collider_rule)
    _validate_possible_dsep(possible_dsep)
    cpdag_structure, _score = ges(
        X, score_func=score_func, lambda_value=lambda_value, categorical=categorical
    )
    cpdag = from_structure(cpdag_structure)

    test = make_indep_test(indep_test, X, categorical=categorical)
    sepsets = _derive_sepsets(cpdag, test, alpha)

    marks = _circle_pag(cpdag != NULL)
    _orient_colliders(marks, sepsets)
    _remove_by_possible_dsep(marks, test, alpha, sepsets, order_free=possible_dsep == "order_free")
    marks = _circle_pag(marks != NULL)
    ambiguous = (
        _majority_sepsets(marks, test, alpha, sepsets)
        if collider_rule == "majority"
        else frozenset()
    )
    _orient_colliders(marks, sepsets, ambiguous)
    _fci_orient(marks, test, alpha, sepsets, ambiguous)

    return GraphStructure.from_numpy(marks, kind="pag")


def _derive_sepsets(
    cpdag: np.ndarray, test: CITest, alpha: float
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Separating sets for every non-adjacent CPDAG pair, from the pooled two-endpoint adjacency.

    For each non-adjacent ``(x, y)`` (``x < y``), searches subsets of ``adj(x) | adj(y) - {x, y}``
    by increasing size and keeps the first one found independent, recorded both ways; a pair with no
    such subset gets no entry. Skips the FAS adjacency search FCI normally runs first, since the
    CPDAG already supplies the skeleton.
    """
    n = cpdag.shape[0]
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {}
    for x, y in combinations(range(n), 2):
        if _adjacent(cpdag, x, y):
            continue
        pool = sorted((set(_adjacent_nodes(cpdag, x)) | set(_adjacent_nodes(cpdag, y))) - {x, y})
        for size in range(len(pool) + 1):
            found = False
            for cond in combinations(pool, size):
                if test(x, y, cond) > alpha:
                    sepsets[(x, y)] = cond
                    sepsets[(y, x)] = cond
                    found = True
                    break
            if found:
                break
    return sepsets
