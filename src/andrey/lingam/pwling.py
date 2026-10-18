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
# Two columns are degenerate when the residual of one on the other keeps at most this fraction of
# its variance (1 - corr^2), so that its standardized entropy is undefined (0 / 0) or rounding
# noise, and which of the two a CPU's rounding gives differs by kernel. Copies, exact or perturbed
# by 1e-12, measure < 2e-15 at every step; rank-transformed near-copies with |corr| up to 0.999999
# measure > 7e-6, and orders differed between CPUs only for pairs below 1e-14.
_DEGENERATE_TOL = 1e-12


def compute_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares residual of ``a`` regressed on ``b``: ``a - (cov(a, b) / var(b)) * b``.

    Covariance and variance share ``ddof=0``, so the residual is uncorrelated with ``b`` in the
    sample, as the order search assumes. A constant ``b`` explains nothing and leaves ``a`` as it
    is. When ``b`` leaves at most ``_DEGENERATE_TOL`` of the variance of ``a`` unexplained
    (``1 - corr^2``, as for a copy up to scale), the residual is exact zeros, which
    :func:`score_candidates` marks degenerate.
    """
    c = np.cov(a, b, ddof=0)
    if c[1, 1] == 0.0:
        return a.copy()
    if c[0, 1] ** 2 >= (1.0 - _DEGENERATE_TOL) * c[0, 0] * c[1, 1]:
        return np.zeros_like(a)
    return a - (c[0, 1] / np.var(b)) * b


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

    A degenerate candidate scores NaN, which :func:`pick` ranks highest: a constant column (one
    :func:`compute_residual` zeroed), or one of a pair with ``1 - corr^2 <= _DEGENERATE_TOL``.
    Other candidates are scored against the non-constant columns only.
    """
    cols = np.asarray(U)
    if len(cols) < 2:
        return np.zeros(len(cols), dtype=np.float64)  # a lone candidate has no pair to score
    sub = X[:, cols]
    mean, std = sub.mean(axis=0), sub.std(axis=0)
    scores = np.full(len(cols), np.nan)
    live = np.flatnonzero(std > 0.0)
    if len(live) < 2:
        scores[live] = 0.0  # a lone live candidate has no pair to score
        return scores
    if len(live) < len(cols):
        sub, mean, std = sub[:, live], mean[live], std[live]
    x_std = (sub - mean) / std  # (n, k) standardized live columns
    ent = stats.entropy(x_std)  # (k,) batched candidate entropy
    cov = np.cov(x_std, rowvar=False, ddof=0)  # (k, k), same ddof as var: least-squares slopes
    var = np.var(x_std, axis=0)  # (k,), ddof=0
    unexplained = 1.0 - cov * cov.T / np.outer(var, var)  # symmetric 1 - corr^2
    np.fill_diagonal(unexplained, np.inf)
    degenerate = (unexplained <= _DEGENERATE_TOL).any(axis=0)
    k = len(live)
    for a in np.flatnonzero(~degenerate):
        others = np.arange(k) != a
        xi = x_std[:, a][:, None]  # (n, 1)
        xj = x_std[:, others]  # (n, k-1)
        r_ij = xi - (cov[a, others] / var[others])[None, :] * xj  # residual of i on each j
        r_ji = xj - (cov[others, a] / var[a])[None, :] * xi  # residual of each j on i
        r_ij /= np.std(r_ij, axis=0)
        r_ji /= np.std(r_ji, axis=0)
        diff = (ent[others] + stats.entropy(r_ij)) - (ent[a] + stats.entropy(r_ji))
        scores[live[a]] = -np.sum(np.minimum(0.0, diff) ** 2)
    return scores


def score_candidates_serial(X: np.ndarray, U: list[int]) -> np.ndarray:
    """Readable per-pair reference implementation of :func:`score_candidates`."""
    cols = [k for k in U if X[:, k].std() > 0.0]
    x_std = {k: (X[:, k] - X[:, k].mean()) / X[:, k].std() for k in cols}
    ent = {k: compute_entropy(x_std[k]) for k in cols}
    scores = np.full(len(U), np.nan)
    for a, i in enumerate(U):
        if i not in x_std:
            continue
        xi, e_xi = x_std[i], ent[i]
        acc = 0.0
        for j in cols:
            if j == i:
                continue
            xj, e_xj = x_std[j], ent[j]
            r_ij = compute_residual(xi, xj)
            r_ji = compute_residual(xj, xi)
            if not (r_ij.any() and r_ji.any()):  # a degenerate pair
                acc = np.nan
                break
            diff = (e_xj + compute_entropy(r_ij / np.std(r_ij))) - (
                e_xi + compute_entropy(r_ji / np.std(r_ji))
            )
            acc += min(0.0, diff) ** 2
        scores[a] = -acc
    return scores


def pick(scores: np.ndarray, candidates: list[int], ranks: np.ndarray) -> int:
    """The candidate with the highest score; the lowest ``ranks`` entry among tied candidates.

    ``scores`` are aligned with ``candidates`` and ``<= 0`` or NaN; two tie when their
    ``sqrt(-score)`` values both lie within ``_TIE_TOL`` of the best one. A NaN score, from a
    degenerate candidate (:func:`score_candidates`), ranks highest, as under ``np.argmax``; all NaN
    candidates then tie. ``ranks`` is indexed by column, as
    :func:`~andrey.search.hc.tie_ranks` returns for the input data.
    """
    norm = np.sqrt(-np.asarray(scores, dtype=np.float64))
    nan = np.isnan(norm)
    tied = np.flatnonzero(nan) if nan.any() else np.flatnonzero(norm <= norm.min() + _TIE_TOL)
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
