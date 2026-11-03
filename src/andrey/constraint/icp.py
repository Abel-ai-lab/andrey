"""ICP, the engine behind :func:`andrey.icp`.

The algorithm, its settings, and its references are described on :func:`andrey.icp`.

For each set ``S`` of the target's candidate predictors, of at most ``max_parents`` of them, the
engine tests whether the regression of the target on ``S`` (with an intercept) is the same in every
environment (Peters, Buhlmann and Meinshausen 2016, Section 3.1):

- ``"approximate"`` (Method II): one regression on the pooled rows; for each environment, Welch's
  t-test of the residual means and the two-sided F-test of the residual variances, in the
  environment against the other rows, each p-value twice the smaller of the two.
- ``"exact"`` (Method I): for each environment, the regression on the other rows predicts the
  environment's rows, and Chow's statistic ``(RSS_all - RSS_out) / n_e`` over the residual
  variance ``RSS_out / (n_out - |S| - 1)`` is ``F(n_e, n_out - |S| - 1)`` under the null (the
  prediction errors' quadratic form of eq. 17 equals ``RSS_all - RSS_out``).

A set's p-value is the smallest environment's times the number of distinct comparisons, ``K``,
capped at 1. With two environments the approximate test's two comparisons are one test, since
Welch's statistic and the two-sided F p-value do not depend on which group comes first, so the
factor is 1. A set is accepted when its p-value exceeds ``alpha``; the estimate is the intersection
of the accepted sets (eq. 12).

Every regression reads Gram matrices of ``[1, x, y]`` per environment, with each column centered on
its pooled mean, computed once: a set costs ``O(K s^3)`` in its size ``s``, whatever the number of
rows, and the sets of one size run in batches. The result does not depend on the order in which
the sets are tested.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ["INVARIANCE_TESTS", "Invariance", "icp"]

#: The invariance tests: the paper's Method II and Method I.
INVARIANCE_TESTS = ("approximate", "exact")
#: The most sets one batch holds.
BATCH = 2048


@dataclass(frozen=True)
class Invariance:
    """The result of one ICP fit, by 0-based column position.

    ``parents`` is the intersection of the accepted sets, empty when none is accepted;
    ``accepted`` lists the accepted sets by size, then by position. ``p_values`` holds each
    column's p-value, the largest p-value of the tested sets without it (1 when no set is
    accepted), and ``NaN`` at the target. ``intervals`` holds each column's ``(low, high)``, the
    union over the accepted sets of the per-set intervals, with 0 for a set without the column, and
    ``NaN`` at the target; ``None`` when no set is accepted. ``best_p_value`` is the largest p-value
    of any tested set.
    """

    parents: tuple[int, ...]
    accepted: tuple[tuple[int, ...], ...]
    p_values: np.ndarray
    intervals: np.ndarray | None
    best_p_value: float


def icp(
    X: np.ndarray,
    codes: np.ndarray,
    target: int,
    *,
    alpha: float,
    invariance_test: str,
    max_parents: int | None,
    environments: Sequence[object],
) -> Invariance:
    """Test every set of at most ``max_parents`` predictors of column ``target`` for invariance.

    ``X`` is the checked ``(n_samples, n_variables)`` data and ``codes`` each row's environment,
    ``0..K-1`` with ``K >= 2``, labeled ``environments`` in messages. The caller checks that every
    environment has at least 3 rows and that each regression has more rows than coefficients.
    Raises ``ValueError`` when the rows of a regression leave its coefficients undetermined.
    """
    n, d = X.shape
    others = np.array([j for j in range(d) if j != target], dtype=np.intp)
    p = others.size
    K = int(codes.max()) + 1
    columns = np.column_stack([X[:, others], X[:, target]])
    Z = np.column_stack([np.ones(n), columns - columns.mean(axis=0)])
    grams = np.stack([Z[codes == k].T @ Z[codes == k] for k in range(K)])
    sizes = np.bincount(codes, minlength=K).astype(np.float64)
    largest = p if max_parents is None else min(max_parents, p)

    accepted: list[tuple[int, ...]] = []
    without = np.full(p, -np.inf)  # the largest p-value of a tested set without each predictor
    inside = np.ones(p, dtype=bool)  # in every accepted set so far
    outside = np.zeros(p, dtype=bool)  # outside some accepted set
    low, high = np.full(p, np.inf), np.full(p, -np.inf)
    best = -np.inf
    for size in range(largest + 1):
        for sets in _batches(p, size):
            pooled, beta, rss = _pooled(grams, sets)
            if invariance_test == "exact":
                pv = _exact(grams, sizes, sets, pooled, rss, environments)
            else:
                pv = _approximate(grams, sizes, sets, beta)
            best = max(best, float(pv.max()))
            member = np.zeros((len(sets), p), dtype=bool)
            member[np.arange(len(sets))[:, None], sets] = True
            if p:
                without = np.maximum(without, np.where(member, -np.inf, pv[:, None]).max(axis=0))
            keep = pv > alpha
            if not keep.any():
                continue
            accepted.extend(tuple(int(j) for j in others[row]) for row in sets[keep])
            inside &= member[keep].all(axis=0)
            outside |= ~member[keep].all(axis=0)
            if size:
                lows, highs = _intervals(pooled[keep], beta[keep], rss[keep], n, alpha)
                np.minimum.at(low, sets[keep].ravel(), lows.ravel())
                np.maximum.at(high, sets[keep].ravel(), highs.ravel())

    p_values = np.full(d, np.nan)
    if not accepted:
        p_values[others] = 1.0
        return Invariance((), (), p_values, None, best)
    p_values[others] = without
    low[outside], high[outside] = np.minimum(low[outside], 0.0), np.maximum(high[outside], 0.0)
    intervals = np.full((d, 2), np.nan)
    intervals[others] = np.column_stack([low, high])
    parents = tuple(int(j) for j in others[inside])
    return Invariance(parents, tuple(accepted), p_values, intervals, best)


def _batches(p: int, size: int) -> Iterator[np.ndarray]:
    """Yield the sets of ``size`` of ``p`` predictors, in lexicographic order, as index arrays."""
    combinations = itertools.combinations(range(p), size)
    while chunk := list(itertools.islice(combinations, BATCH)):
        yield np.array(chunk, dtype=np.intp).reshape(len(chunk), size)


def _rows(sets: np.ndarray) -> np.ndarray:
    """The Gram rows of each set's regression: the intercept, then its predictors."""
    return np.column_stack([np.zeros(len(sets), dtype=np.intp), sets + 1])


def _solve(A: np.ndarray, b: np.ndarray, what: str) -> np.ndarray:
    """Solve each system ``A x = b``; ``ValueError`` names ``what`` when one has no solution."""
    try:
        return np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        raise ValueError(
            f"icp: the predictors are linearly dependent on the rows of {what}, so a regression's "
            "coefficients are undetermined; drop a column that is a linear function of others "
            "there, or lower max_parents"
        ) from None


def _pooled(grams: np.ndarray, sets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each set's pooled Gram block, coefficients, and residual sum of squares."""
    G = grams.sum(axis=0)
    rows, y = _rows(sets), G.shape[0] - 1
    A = G[rows[:, :, None], rows[:, None, :]]
    b = G[rows, y]
    beta = _solve(A, b, "every environment")
    return A, beta, G[y, y] - (b * beta).sum(axis=1)


def _approximate(
    grams: np.ndarray, sizes: np.ndarray, sets: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Method II: each set's p-value from its pooled residuals' means and variances."""
    rows, y = _rows(sets), grams.shape[1] - 1
    A = grams[:, rows[:, :, None], rows[:, None, :]]
    # Each environment's residual sum and sum of squares, from its Gram block.
    total = grams[:, 0, y][:, None] - (grams[:, 0, rows] * beta).sum(axis=2)
    squares = (
        grams[:, y, y][:, None]
        - 2.0 * (grams[:, rows, y] * beta).sum(axis=2)
        + np.einsum("mi,kmij,mj->km", beta, A, beta)
    )
    n_in, n_out = sizes[:, None], sizes.sum() - sizes[:, None]
    mean_in = total / n_in
    mean_out = (total.sum(axis=0) - total) / n_out
    var_in = (squares - n_in * mean_in**2) / (n_in - 1.0)
    var_out = (squares.sum(axis=0) - squares - n_out * mean_out**2) / (n_out - 1.0)
    se_in, se_out = var_in / n_in, var_out / n_out
    t = (mean_in - mean_out) / np.sqrt(se_in + se_out)
    df = (se_in + se_out) ** 2 / (se_in**2 / (n_in - 1.0) + se_out**2 / (n_out - 1.0))
    p_mean = 2.0 * stats.t.sf(np.abs(t), df)
    ratio = var_in / var_out
    below = stats.f.cdf(ratio, n_in - 1.0, n_out - 1.0)
    above = stats.f.sf(ratio, n_in - 1.0, n_out - 1.0)
    p_variance = 2.0 * np.minimum(below, above)
    p_env = 2.0 * np.minimum(p_mean, p_variance)
    factor = 1.0 if len(sizes) == 2 else float(len(sizes))
    return np.minimum(1.0, factor * p_env.min(axis=0))


def _exact(
    grams: np.ndarray,
    sizes: np.ndarray,
    sets: np.ndarray,
    pooled: np.ndarray,
    rss: np.ndarray,
    environments: Sequence[object],
) -> np.ndarray:
    """Method I: each set's p-value from Chow's test of every environment against the others."""
    rows, y = _rows(sets), grams.shape[1] - 1
    G = grams.sum(axis=0)
    k = rows.shape[1]
    p_env = np.empty((len(sizes), len(sets)))
    for e in range(len(sizes)):
        out = G - grams[e]
        b = out[rows, y]
        A = pooled - grams[e][rows[:, :, None], rows[:, None, :]]
        beta = _solve(A, b, f"every environment but {environments[e]!r}")
        rss_out = out[y, y] - (b * beta).sum(axis=1)
        n_e, df = sizes[e], sizes.sum() - sizes[e] - k
        statistic = np.maximum(rss - rss_out, 0.0) / n_e / (rss_out / df)
        p_env[e] = stats.f.sf(statistic, n_e, df)
    return np.minimum(1.0, len(sizes) * p_env.min(axis=0))


def _intervals(
    A: np.ndarray, beta: np.ndarray, rss: np.ndarray, n: int, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Each set's rectangular interval for its coefficients (Section 3.1, step 3).

    For a set of ``s`` predictors, ``beta +- t_{1 - alpha/(2s), n - s - 1} * se``, with the
    standard errors of the pooled regression.
    """
    s = beta.shape[1] - 1
    variance = rss / (n - s - 1)
    unscaled = np.diagonal(np.linalg.inv(A), axis1=1, axis2=2)[:, 1:]
    se = np.sqrt(variance[:, None] * unscaled)
    q = stats.t.ppf(1.0 - alpha / (2.0 * s), n - s - 1)
    return beta[:, 1:] - q * se, beta[:, 1:] + q * se
