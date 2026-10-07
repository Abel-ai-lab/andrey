"""MultiGroupDirectLiNGAM: one shared causal order across datasets sharing features (Shimizu 2012).

Given ``G >= 2`` datasets that share features but not samples, jointly estimate a single causal
order, then a per-group weighted adjacency. The order is peeled one exogenous variable at a time
(as in :mod:`direct`), but each candidate's exogeneity score is the sample-size-weighted sum of the
per-group ``pwling`` scores; residualization is applied to each group independently. Reuses the
shared LiNGAM foundation (:mod:`pwling`, :mod:`adjacency`); deterministic.

References
----------
S. Shimizu. Joint estimation of linear non-Gaussian acyclic models. Neurocomputing, 81, 2012.
"""

from __future__ import annotations

import numpy as np

from andrey.lingam import adjacency, pwling


def multi_group_direct_lingam(
    X_list: list[np.ndarray],
) -> tuple[list[int], list[np.ndarray]]:
    """Fit MultiGroupDirectLiNGAM: return ``(causal_order, [B_g])``.

    ``causal_order`` is the single order shared by all groups; ``B_g`` is group ``g``'s weighted
    adjacency for ``x_i = sum_j B_g[i, j] x_j`` (so ``B_g[i, j]`` weights edge ``j -> i``). Requires
    ``len(X_list) >= 2`` arrays with matching ``n_features``.
    """
    if not isinstance(X_list, list):
        raise ValueError("X_list must be a list.")
    if len(X_list) < 2:
        raise ValueError("X_list must be a list containing at least two items")

    groups = [np.asarray(X, dtype=np.float64) for X in X_list]
    for X in groups:
        if X.ndim != 2:
            raise ValueError(
                f"each group must be a 2-D (n_samples, n_features) array, got ndim={X.ndim}"
            )
        if not np.isfinite(X).all():
            raise ValueError("group data contains NaN or Inf")
    n_features = groups[0].shape[1]
    for X in groups:
        if X.shape[1] != n_features:
            raise ValueError("X_list must be a list with the same number of features")

    order = _find_shared_causal_order(groups, n_features)
    adjacency_matrices = [adjacency.estimate(X, order) for X in groups]
    return order, adjacency_matrices


def _find_shared_causal_order(groups: list[np.ndarray], n_features: int) -> list[int]:
    """Peel exogenous variables by size-weighted sum of per-group ``pwling`` scores.

    Mirrors :func:`pwling.find_causal_order` but sums the candidate scores across groups (each
    weighted by ``n_g / total_n``) and residualizes every group independently against the pick.
    """
    total_n = sum(len(X) for X in groups)
    remaining = list(range(n_features))
    order: list[int] = []
    # Copy + float64 so the in-place residual write-back never truncates.
    residualized = [np.array(X, dtype=np.float64) for X in groups]

    while remaining:
        if len(remaining) == 1:
            m = remaining[0]
        else:
            neg_mg = np.zeros(len(remaining), dtype=np.float64)
            for X in residualized:
                neg_mg += (len(X) / total_n) * pwling.score_candidates(X, remaining)
            m = remaining[int(np.argmax(neg_mg))]
        for X in residualized:
            for i in remaining:
                if i != m:
                    X[:, i] = pwling.compute_residual(X[:, i], X[:, m])
        order.append(m)
        remaining = [u for u in remaining if u != m]
    return order
