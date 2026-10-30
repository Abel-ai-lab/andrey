"""DirectLiNGAM: linear non-Gaussian acyclic causal discovery (Shimizu et al. 2011).

Discovers the causal order of a linear structural-equation model with independent non-Gaussian
errors by peeling the most-exogenous variable one at a time (a pairwise likelihood-ratio measure of
:mod:`pwling`: ``"pwling"``, ``"skew"``, or ``"tanh"``), then estimates the weighted adjacency ``B``
(``x_i = sum_j B[i, j] x_j``, so ``B[i, j]`` weights edge ``j -> i``) by adaptive Lasso on that
order (:mod:`adjacency`). Deterministic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from andrey.lingam import adjacency, pwling

if TYPE_CHECKING:
    from andrey.core.knowledge import Constraints


def direct_lingam(
    X: np.ndarray, *, measure: str = "pwling", constraints: Constraints | None = None
) -> tuple[list[int], np.ndarray]:
    """Fit DirectLiNGAM: return ``(causal_order, B)`` for ``x_i = sum_j B[i, j] x_j``.

    ``measure`` is one of :data:`~andrey.lingam.pwling.MEASURES` and raises ``ValueError``
    otherwise; every measure is deterministic. With background knowledge ``constraints``, the order
    keeps
    :meth:`~andrey.core.knowledge.Constraints.precedence` (required edges and tiers), and the
    regression drops forbidden parents and keeps required ones unpenalized
    (:func:`~andrey.lingam.adjacency.estimate`), so ``B`` has no forbidden edge. A required edge's
    least-squares weight can be ``0``; :func:`andrey.direct_lingam` still puts the edge in its DAG.
    """
    X = np.asarray(X, dtype=np.float64)
    if constraints is None:
        order = pwling.find_causal_order(X, measure=measure)
        return order, adjacency.estimate(X, order)
    order = pwling.find_causal_order(X, precedence=constraints.precedence(), measure=measure)
    B = adjacency.estimate(X, order, forbidden=constraints.forbidden, required=constraints.required)
    return order, B
