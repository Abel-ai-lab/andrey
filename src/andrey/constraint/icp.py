"""ICP, the engine behind :func:`andrey.icp`.

The algorithm, its settings, and its references are described on :func:`andrey.icp`.

For each set ``S`` of the target's candidate predictors, of at most ``max_parents`` of them, the
engine tests whether the regression of the target on ``S`` (with an intercept) is the same in every
environment (Peters, Buhlmann and Meinshausen 2016, Section 3.1), by the invariance test
``"icp_approximate"`` (Method II) or ``"icp_exact"`` (Method I) of :mod:`andrey.core.invariance`,
built once through :func:`~andrey.core.invariance.make_invariance_test` with the environments as
its groups. A set is accepted when its p-value exceeds ``alpha``; the estimate is the intersection
of the accepted sets (eq. 12). The sets of one size are tested in batches, and the result does not
depend on the order in which they are tested.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy import stats

from andrey.core.invariance import RegressionInvariance, SingularRegression, make_invariance_test

__all__ = ["TESTS", "Invariance", "icp"]

#: Each ``invariance_test`` of :func:`andrey.icp` and the invariance test it builds: the paper's
#: Method II and Method I.
TESTS = {"approximate": "icp_approximate", "exact": "icp_exact"}
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
    # The regressions are linear-Gaussian; the invariance tests read no indep_test.
    test = cast(
        RegressionInvariance,
        make_invariance_test(TESTS[invariance_test], X, codes, indep_test="fisherz"),
    )
    largest = p if max_parents is None else min(max_parents, p)

    accepted: list[tuple[int, ...]] = []
    without = np.full(p, -np.inf)  # the largest p-value of a tested set without each predictor
    inside = np.ones(p, dtype=bool)  # in every accepted set so far
    outside = np.zeros(p, dtype=bool)  # outside some accepted set
    low, high = np.full(p, np.inf), np.full(p, -np.inf)
    best = -np.inf
    for size in range(largest + 1):
        for sets in _batches(p, size):
            try:
                pv, pooled, beta, rss = test.fit(target, others[sets])
            except SingularRegression as error:
                where = (
                    "every environment"
                    if error.group is None
                    else f"every environment but {environments[error.group]!r}"
                )
                raise ValueError(
                    f"icp: the predictors are linearly dependent on the rows of {where}, so a "
                    "regression's coefficients are undetermined; drop a column that is a linear "
                    "function of others there, or lower max_parents"
                ) from None
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
