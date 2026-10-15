"""DirectLiNGAM: linear non-Gaussian acyclic causal discovery (Shimizu et al. 2011).

Discovers the causal order of a linear structural-equation model with independent non-Gaussian
errors by peeling the most-exogenous variable one at a time (the pairwise-likelihood ``pwling``
measure, :mod:`pwling`), then estimates the weighted adjacency ``B`` (``x_i = sum_j B[i, j] x_j``,
so ``B[i, j]`` weights edge ``j -> i``) by adaptive Lasso on that order (:mod:`adjacency`).
Deterministic; ``measure="pwling"`` only.
"""

from __future__ import annotations

import numpy as np

from andrey.lingam import adjacency, pwling


def direct_lingam(X: np.ndarray, *, measure: str = "pwling") -> tuple[list[int], np.ndarray]:
    """Fit DirectLiNGAM: return ``(causal_order, B)`` for ``x_i = sum_j B[i, j] x_j``.

    ``measure="pwling"`` is the only supported measure; it is deterministic.
    """
    if measure != "pwling":
        raise NotImplementedError(f"DirectLiNGAM supports measure='pwling', not {measure!r}")
    X = np.asarray(X, dtype=np.float64)
    order = pwling.find_causal_order(X)
    return order, adjacency.estimate(X, order)
