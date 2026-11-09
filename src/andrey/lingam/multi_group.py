"""MultiGroupDirectLiNGAM: one shared causal order across datasets sharing features (Shimizu 2012).

Given ``G >= 2`` datasets that share features but not samples, jointly estimate a single causal
order, then a per-group weighted adjacency. The order is peeled one exogenous variable at a time
(as in :mod:`direct`), but each candidate's exogeneity score is the sample-size-weighted sum of the
per-group scores of the ``measure``; residualization is applied to each group independently. Reuses
the shared LiNGAM foundation (:mod:`pwling`, :mod:`adjacency`); deterministic.

References
----------
S. Shimizu. Joint estimation of linear non-Gaussian acyclic models. Neurocomputing, 81, 2012.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from andrey.core.ties import tie_ranks
from andrey.lingam import adjacency, pwling
from andrey.lingam.direct import forbidden_parents

if TYPE_CHECKING:
    from andrey.core.knowledge import Constraints


def multi_group_direct_lingam(
    X_list: list[np.ndarray], *, measure: str = "pwling", constraints: Constraints | None = None
) -> tuple[list[int], list[np.ndarray]]:
    """Fit MultiGroupDirectLiNGAM: return ``(causal_order, [B_g])``.

    ``causal_order`` is the single order shared by all groups; ``B_g`` is group ``g``'s weighted
    adjacency for ``x_i = sum_j B_g[i, j] x_j`` (so ``B_g[i, j]`` weights edge ``j -> i``). Requires
    ``len(X_list) >= 2`` arrays with matching ``n_features``. ``measure`` is one of
    :data:`~andrey.lingam.pwling.MEASURES`, as in :func:`~andrey.lingam.direct.direct_lingam`; each
    group is standardized, and for ``"skew"`` sign-corrected, on its own. With background knowledge
    ``constraints``, the order keeps :meth:`~andrey.core.knowledge.Constraints.precedence` and
    reads ``forbidden_paths``, and each group's regression drops forbidden parents and forbidden
    paths and keeps required ones unpenalized, as
    :func:`~andrey.lingam.direct.direct_lingam` does.
    """
    groups = [np.asarray(X, dtype=np.float64) for X in X_list]
    n_features = groups[0].shape[1]

    if constraints is None:
        order = _find_shared_causal_order(groups, n_features, measure=measure)
        return order, [adjacency.estimate(X, order) for X in groups]
    order = _find_shared_causal_order(
        groups,
        n_features,
        precedence=constraints.precedence(),
        forbidden_paths=constraints.forbidden_paths,
        measure=measure,
    )
    forbidden = forbidden_parents(constraints)
    adjacency_matrices = [
        adjacency.estimate(X, order, forbidden=forbidden, required=constraints.required)
        for X in groups
    ]
    return order, adjacency_matrices


def _find_shared_causal_order(
    groups: list[np.ndarray],
    n_features: int,
    *,
    precedence: np.ndarray | None = None,
    forbidden_paths: np.ndarray | None = None,
    measure: str = "pwling",
) -> list[int]:
    """Peel exogenous variables by size-weighted sum of per-group ``measure`` scores.

    Mirrors :func:`pwling.find_causal_order` but sums the candidate scores across groups (each
    weighted by ``n_g / total_n``) and residualizes every group independently against the pick.
    Ties are broken by :func:`~andrey.core.ties.tie_ranks` of the stacked input groups. Columns
    that are copies in any group form one copy group (:func:`pwling.copy_groups`), scored by its
    representative and followed directly by the other copies. With ``precedence``, as for
    :func:`pwling.find_causal_order`, no copy group is formed, and the pick is the best of
    :func:`pwling.eligible`, which also reads ``forbidden_paths``.
    """
    score = pwling.scorer(measure)
    total_n = sum(len(X) for X in groups)
    remaining = list(range(n_features))
    order: list[int] = []
    # Copy + float64 so the in-place residual write-back never truncates.
    residualized = [np.array(X, dtype=np.float64) for X in groups]
    ranks = tie_ranks(np.vstack(groups))

    while remaining:
        if precedence is None:
            copies = pwling.copy_groups(residualized, remaining, ranks)
        else:
            copies = {u: [] for u in remaining}
        candidates = list(copies)
        if len(candidates) == 1:
            m = candidates[0]
        else:
            neg_mg = np.zeros(len(candidates), dtype=np.float64)
            for X in residualized:
                neg_mg += (len(X) / total_n) * score(X, candidates)
            if precedence is None:
                m = pwling.pick(neg_mg, candidates, ranks)
            else:
                eligible = pwling.eligible(precedence, forbidden_paths, candidates)
                chosen = [c for c, ok in zip(candidates, eligible, strict=True) if ok]
                m = pwling.pick(neg_mg[eligible], chosen, ranks)
        placed = [m, *copies[m]]
        remaining = [u for u in remaining if u not in placed]
        for X in residualized:
            for i in remaining:
                X[:, i] = pwling.compute_residual(X[:, i], X[:, m])
        order += placed
    return order
