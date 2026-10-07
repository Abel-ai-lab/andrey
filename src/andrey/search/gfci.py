"""GFCI, the engine behind :func:`andrey.gfci`.

The algorithm, its settings, and its references are described on :func:`andrey.gfci`.

The GES CPDAG's adjacencies, reset to circle marks, go through the collider,
possible-d-separation, and R0-R10 passes of :mod:`andrey.constraint.fci`, with its rule
implementations, Fisher-Z test, and mark convention.
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
) -> GraphStructure:
    """Discover a PAG with GES adjacency and FCI orientation."""
    X = np.asarray(data, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"data must be a 2-D (n_samples, n_features) array, got ndim={X.ndim}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in the open interval (0, 1), got {alpha}")

    _validate_collider_rule(collider_rule)
    cpdag_structure, _score = ges(X, score_func=score_func, lambda_value=lambda_value)
    cpdag = from_structure(cpdag_structure)

    test = make_indep_test(indep_test, X)
    sepsets = _derive_sepsets(cpdag, test, alpha)

    marks = _circle_pag(cpdag != NULL)
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
