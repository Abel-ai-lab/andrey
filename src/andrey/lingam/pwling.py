"""The pairwise-likelihood (pwling) exogeneity measure for LiNGAM ordering.

Scores each candidate variable by how plausibly exogenous it is -- the Hyvarinen maximum-entropy
approximation of the pairwise likelihood ratios (Hyvarinen & Smith 2013) -- and peels the
most-exogenous variable one at a time. Entropy is evaluated in batch over a stack of residuals.
Shared by the LiNGAM ordering methods (DirectLiNGAM, MultiGroupDirectLiNGAM, the temporal LiNGAMs).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from andrey.core import stats


def compute_residual(a: np.ndarray, b: np.ndarray, *, least_squares: bool = False) -> np.ndarray:
    """Residual of ``a`` regressed on ``b``: ``a - (cov(a, b) / var(b)) * b``.

    By default the slope is ``np.cov`` (``ddof=1``) over ``np.var`` (``ddof=0``), ``n / (n - 1)``
    times the least-squares slope. ``least_squares=True`` takes both with ``ddof=0``: the
    least-squares slope, whose residual is uncorrelated with ``b`` in the sample.
    """
    if least_squares:
        return a - (np.cov(a, b, ddof=0)[0, 1] / np.var(b)) * b
    return a - (np.cov(a, b)[0, 1] / np.var(b)) * b


def compute_entropy(u: np.ndarray) -> float:
    """Hyvarinen max-entropy approximation of a standardized 1-D column (via ``core.stats``)."""
    return float(stats.entropy(u)[0])


def score_candidates(X: np.ndarray, U: list[int], *, least_squares: bool = False) -> np.ndarray:
    """Score how plausibly exogenous each column in ``U`` of ``X`` is (higher = more exogenous).

    Returns an array aligned with ``U``: ``score(i) = -sum_j min(0, diff_mi(i, j))^2`` with
    ``diff_mi(i, j) = (H(x_j) + H(r_{i|j})) - (H(x_i) + H(r_{j|i}))`` on standardized columns.
    Entropy is batched over a stack of residuals; DirectLiNGAM takes the ``argmax``, MultiGroup
    sums across groups. Order-exactness needs the float64 entropy path -- on a float32 device
    (atol 1e-3) a near-tie could flip the pick, so pin ``ANDREY_DEVICE=numpy`` for exact order.
    ``least_squares`` picks the residual slope as in :func:`compute_residual`.
    """
    cols = np.asarray(U)
    if len(cols) < 2:
        return np.zeros(len(cols), dtype=np.float64)  # a lone candidate has no pair to score
    sub = X[:, cols]
    x_std = (sub - sub.mean(axis=0)) / sub.std(axis=0)  # (n, m) standardized candidate columns
    ent = stats.entropy(x_std)  # (m,) batched candidate entropy
    cov = np.cov(x_std, rowvar=False, ddof=0) if least_squares else np.cov(x_std, rowvar=False)
    var = np.var(x_std, axis=0)  # (m,), ddof=0
    m = len(cols)
    scores = np.empty(m, dtype=np.float64)
    for a in range(m):
        others = np.arange(m) != a
        xi = x_std[:, a][:, None]  # (n, 1)
        xj = x_std[:, others]  # (n, m-1)
        r_ij = xi - (cov[a, others] / var[others])[None, :] * xj  # residual of i on each j
        r_ji = xj - (cov[others, a] / var[a])[None, :] * xi  # residual of each j on i
        r_ij /= np.std(r_ij, axis=0)
        r_ji /= np.std(r_ji, axis=0)
        diff = (ent[others] + stats.entropy(r_ij)) - (ent[a] + stats.entropy(r_ji))
        scores[a] = -np.sum(np.minimum(0.0, diff) ** 2)
    return scores


def score_candidates_serial(
    X: np.ndarray, U: list[int], *, least_squares: bool = False
) -> np.ndarray:
    """Readable per-pair reference implementation of :func:`score_candidates`."""
    cols = list(U)
    x_std = {k: (X[:, k] - X[:, k].mean()) / X[:, k].std() for k in cols}
    ent = {k: compute_entropy(x_std[k]) for k in cols}
    scores = np.empty(len(cols), dtype=np.float64)
    for a, i in enumerate(cols):
        xi, e_xi = x_std[i], ent[i]
        acc = 0.0
        for j in cols:
            if j == i:
                continue
            xj, e_xj = x_std[j], ent[j]
            r_ij = compute_residual(xi, xj, least_squares=least_squares)
            r_ji = compute_residual(xj, xi, least_squares=least_squares)
            diff = (e_xj + compute_entropy(r_ij / np.std(r_ij))) - (
                e_xi + compute_entropy(r_ji / np.std(r_ji))
            )
            acc += min(0.0, diff) ** 2
        scores[a] = -acc
    return scores


def find_causal_order(
    X: np.ndarray,
    *,
    scores: Callable[..., np.ndarray] = score_candidates,
    least_squares: bool = False,
) -> list[int]:
    """Peel exogenous variables (``argmax`` of ``scores``) one at a time, residualizing the rest.

    ``scores(X, U, least_squares=...) -> array aligned with U`` is injectable. ``least_squares``
    picks the residual slope of the scores and of the residualizing step
    (:func:`compute_residual`). Single-dataset; MultiGroupDirectLiNGAM runs its own group loop
    (per-group residualization), reusing :func:`score_candidates` and :func:`compute_residual`
    directly.
    """
    remaining = list(range(X.shape[1]))
    order: list[int] = []
    residualized = np.array(X, dtype=np.float64)  # copy + float64: the write-back must not truncate
    while remaining:
        # The last variable needs no scoring (and one candidate has no pair to score).
        if len(remaining) == 1:
            m = remaining[0]
        else:
            pick = scores(residualized, remaining, least_squares=least_squares)
            m = remaining[int(np.argmax(pick))]
        for i in remaining:
            if i != m:
                residualized[:, i] = compute_residual(
                    residualized[:, i], residualized[:, m], least_squares=least_squares
                )
        order.append(m)
        remaining = [u for u in remaining if u != m]
    return order
