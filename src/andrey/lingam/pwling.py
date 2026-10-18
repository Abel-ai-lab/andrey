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
from andrey.search.hc import tie_ranks

# A score is ``-sum_j min(0, diff_j)^2``, so ``sqrt(-score)`` is the norm of the negative entropy
# differences, and rounding moves it by at most the norm of their rounding errors. Mathematically
# tied candidates measure < 1e-12 apart in this norm and distinct ones > 5e-10 apart.
_TIE_TOL = 1e-10


def compute_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares residual of ``a`` regressed on ``b``: ``a - (cov(a, b) / var(b)) * b``.

    Covariance and variance share ``ddof=0``, so the residual is uncorrelated with ``b`` in the
    sample, as the order search assumes.
    """
    return a - (np.cov(a, b, ddof=0)[0, 1] / np.var(b)) * b


def compute_entropy(u: np.ndarray) -> float:
    """Hyvarinen max-entropy approximation of a standardized 1-D column (via ``core.stats``)."""
    return float(stats.entropy(u)[0])


def score_candidates(X: np.ndarray, U: list[int]) -> np.ndarray:
    """Score how plausibly exogenous each column in ``U`` of ``X`` is (higher = more exogenous).

    Returns an array aligned with ``U``: ``score(i) = -sum_j min(0, diff_mi(i, j))^2`` with
    ``diff_mi(i, j) = (H(x_j) + H(r_{i|j})) - (H(x_i) + H(r_{j|i}))`` on standardized columns.
    Entropy is batched over a stack of residuals; :func:`pick` chooses the best, MultiGroup sums
    across groups first. Order-exactness needs the float64 entropy path -- on a float32 device
    (atol 1e-3) a near-tie could flip the pick, so pin ``ANDREY_DEVICE=numpy`` for exact order.
    """
    cols = np.asarray(U)
    if len(cols) < 2:
        return np.zeros(len(cols), dtype=np.float64)  # a lone candidate has no pair to score
    sub = X[:, cols]
    x_std = (sub - sub.mean(axis=0)) / sub.std(axis=0)  # (n, m) standardized candidate columns
    ent = stats.entropy(x_std)  # (m,) batched candidate entropy
    cov = np.cov(x_std, rowvar=False, ddof=0)  # (m, m), same ddof as var: least-squares slopes
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


def score_candidates_serial(X: np.ndarray, U: list[int]) -> np.ndarray:
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
            r_ij = compute_residual(xi, xj)
            r_ji = compute_residual(xj, xi)
            diff = (e_xj + compute_entropy(r_ij / np.std(r_ij))) - (
                e_xi + compute_entropy(r_ji / np.std(r_ji))
            )
            acc += min(0.0, diff) ** 2
        scores[a] = -acc
    return scores


def pick(scores: np.ndarray, candidates: list[int], ranks: np.ndarray) -> int:
    """The candidate with the highest score; the lowest ``ranks`` entry among tied candidates.

    ``scores`` are aligned with ``candidates`` and ``<= 0``; two tie when their ``sqrt(-score)``
    values both lie within ``_TIE_TOL`` of the best one. ``ranks`` is indexed by column, as
    :func:`~andrey.search.hc.tie_ranks` returns for the input data.
    """
    norm = np.sqrt(-scores)
    tied = np.flatnonzero(norm <= norm.min() + _TIE_TOL)
    return min((candidates[t] for t in tied), key=lambda c: ranks[c])


def find_causal_order(
    X: np.ndarray,
    *,
    scores: Callable[[np.ndarray, list[int]], np.ndarray] = score_candidates,
) -> list[int]:
    """Peel exogenous variables (:func:`pick` of ``scores``) one at a time, residualizing the rest.

    ``scores(X, U) -> array aligned with U`` is injectable. Ties are broken by
    :func:`~andrey.search.hc.tie_ranks` of the input columns, which are identical on every machine
    and keep their ranks when the columns are reordered; residualized columns differ in the last
    bits across machines. Single-dataset; MultiGroupDirectLiNGAM runs its own group loop
    (per-group residualization), reusing :func:`score_candidates` and :func:`pick` directly.
    """
    remaining = list(range(X.shape[1]))
    order: list[int] = []
    residualized = np.array(X, dtype=np.float64)  # copy + float64: the write-back must not truncate
    ranks = tie_ranks(residualized)
    while remaining:
        # The last variable needs no scoring (and one candidate has no pair to score).
        if len(remaining) == 1:
            m = remaining[0]
        else:
            m = pick(scores(residualized, remaining), remaining, ranks)
        for i in remaining:
            if i != m:
                residualized[:, i] = compute_residual(residualized[:, i], residualized[:, m])
        order.append(m)
        remaining = [u for u in remaining if u != m]
    return order
