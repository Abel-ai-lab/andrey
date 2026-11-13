"""Invariance tests: whether a variable's conditional distribution is the same in groups of rows.

An invariance test is built once over the data and a group per row, and called per query as
``test(y, condition_set)``. It returns the p-value of the null hypothesis that the conditional
distribution of column ``y`` given the columns ``condition_set`` is the same in every group; a
larger p-value means more invariant. The caller compares the p-value with its level, rejects
invariance at ``p <= level`` as a conditional-independence search rejects independence, and keeps
any memo. :data:`INVARIANCE_TESTS` maps each ``invariance_test`` name to the factory that builds
it, and :func:`make_invariance_test` builds one by name.

The ICP tests compare ``K >= 2`` groups, by the regression of ``x_y`` on ``[1, x_S]``
(Peters, Buhlmann, and Meinshausen 2016, Section 3.1):

- ``"icp_approximate"`` (Method II): one regression on the rows of every group; for each group,
  Welch's t-test of the residual means and the two-sided F-test of the residual variances, the
  group's rows against the others', the group's p-value twice the smaller of the two.
- ``"icp_exact"`` (Method I): for each group, the regression on the other rows predicts the
  group's rows, and Chow's statistic ``(RSS_all - RSS_out) / n_g`` over the residual variance
  ``RSS_out / (n_out - |S| - 1)`` is ``F(n_g, n_out - |S| - 1)`` under the null with Gaussian
  errors, upper tail; ``RSS_all - RSS_out`` is the quadratic form of the prediction errors in the
  paper's eq. 17 (Chow 1960).

A query's p-value is the smallest group's times the number of comparisons, ``K``, capped at 1;
with two groups the approximate test's two comparisons are one test, since Welch's statistic and
the two-sided F p-value do not depend on which group comes first, so its factor is 1. Both tests
read Gram matrices of ``[1, x]`` per group, every column centered on its mean over the rows read,
built once per ``y``, so a query costs ``O(K |S|^3)`` whatever the number of rows; ``fit(y, sets)``
answers many sets of one size at once and returns their pooled regressions.

The permutation searches' tests compare two groups:

- ``"gaussian"``: the conditional is linear-Gaussian. ``x_y`` is regressed on ``[1, x_S]`` in each
  group and in the two pooled, and two F-tests compare the groups: the Chow test of equal
  coefficients and intercept, upper tail, and the test of equal residual variances, two-sided
  (Chow 1960). The p-value is twice the smaller of the two, capped at 1, so that rejecting at
  ``level`` rejects either test at ``level / 2`` (Bonferroni). Under ``indep_test="normal_scores"``
  every column is first replaced by its normal scores over all rows.
- ``"indicator"``: the conditional-independence test ``indep_test`` of ``x_y`` and a 0/1 group
  indicator given ``x_S``, on the rows of both groups, with the indicator categorical for the
  categorical and mixed-data tests (Heinze-Deml, Peters, and Meinshausen 2018).

References
----------
Chow, G. C. (1960). Tests of equality between sets of coefficients in two linear regressions.
Econometrica 28(3), 591-605.

Heinze-Deml, C., Peters, J., and Meinshausen, N. (2018). Invariant causal prediction for nonlinear
models. Journal of Causal Inference 6(2).

Peters, J., Buhlmann, P., and Meinshausen, N. (2016). Causal inference by using invariant
prediction: identification and confidence intervals. Journal of the Royal Statistical Society
Series B 78(5), 947-1012.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from scipy import stats

from .ci import MIXED_INDEP_TESTS, make_indep_test
from .stats import normal_scores

__all__ = [
    "GAUSSIAN_INVARIANCE",
    "INVARIANCE_TESTS",
    "TWO_GROUP_INVARIANCE_TESTS",
    "InvarianceTest",
    "RegressionInvariance",
    "SingularRegression",
    "default_invariance_test",
    "make_invariance_test",
]

#: The tests whose conditional is linear-Gaussian, which ``invariance_test=None`` pairs with
#: ``"gaussian"``.
GAUSSIAN_INVARIANCE: tuple[str, ...] = ("fisherz", "student_t", "normal_scores")
#: The shipped two-group tests, the ``invariance_test`` choices of the permutation searches.
TWO_GROUP_INVARIANCE_TESTS: tuple[str, ...] = ("gaussian", "indicator")
# A Gram matrix's direction is left out of a regression when its eigenvalue is at most this
# fraction of the largest: the columns are standardized, so it is singular to working precision.
_RANK_RTOL = 1e-12


class InvarianceTest(Protocol):
    """An invariance test over fixed data and groups, called as ``test(y, condition_set)``.

    Returns the p-value of the null hypothesis that the conditional distribution of column ``y``
    given ``condition_set`` is the same in every group; a larger p-value means more invariant.
    """

    def __call__(self, y: int, condition_set: Iterable[int] = ()) -> float: ...


def default_invariance_test(indep_test: str) -> str:
    """The invariance test ``invariance_test=None`` picks for the conditional-independence test
    ``indep_test``: ``"gaussian"`` for the partial-correlation tests, else ``"indicator"``."""
    return "gaussian" if indep_test in GAUSSIAN_INVARIANCE else "indicator"


def make_invariance_test(
    name: str,
    X: npt.ArrayLike,
    groups: npt.ArrayLike,
    *,
    indep_test: str,
    categorical: npt.ArrayLike | None = None,
    seed: int | None = None,
    degrees_of_freedom: str = "mean",
) -> InvarianceTest:
    """Build the invariance test ``name`` over the rows of ``X`` and their ``groups``.

    ``X`` is the checked ``(n_samples, n_variables)`` data and ``groups[i]`` the group of row ``i``,
    ``0..G-1``, or ``-1`` for a row the test does not read. ``indep_test`` is the fit's
    conditional-independence test, ``categorical`` its mask of the categorical columns (for the
    mixed-data tests), ``seed`` reaches its random draws, and ``degrees_of_freedom`` a categorical
    test, as for :func:`~andrey.core.ci.make_indep_test`. A factory reads what its test needs and
    ignores the rest. Raises ``ValueError`` when ``name`` is not in :data:`INVARIANCE_TESTS`, when a
    two-group test is given other groups than ``0`` and ``1``, and when a ``K``-group test is given
    fewer than two groups.
    """
    try:
        factory = INVARIANCE_TESTS[name]
    except KeyError:
        raise ValueError(
            f"unknown invariance_test {name!r}; registered: {sorted(INVARIANCE_TESTS)}"
        ) from None
    return factory(
        np.asarray(X, dtype=np.float64),
        np.asarray(groups, dtype=np.intp),
        indep_test=indep_test,
        categorical=categorical,
        seed=seed,
        degrees_of_freedom=degrees_of_freedom,
    )


def _two_groups(groups: np.ndarray, name: str) -> None:
    """Raise ``ValueError`` unless ``groups`` holds both groups ``0`` and ``1`` and only ``-1``
    besides."""
    if groups.dtype.kind == "i" and groups.size and groups.min() >= -1 and groups.max() <= 1:
        if np.bincount(groups + 1, minlength=3)[1:].all():
            return
    if set(np.unique(groups).tolist()) - {-1} != {0, 1}:
        raise ValueError(f"invariance_test {name!r} compares two groups, 0 and 1")


class GaussianInvariance:
    """The ``"gaussian"`` test: equal regressions of ``x_y`` on ``[1, x_S]`` in two groups.

    Each column is standardized over the rows read, so the p-values do not change when a column
    is multiplied by a nonzero number and shifted on every row. Each group's Gram matrix of
    ``[1, X]`` is computed once; a query solves its regressions from them, leaving out the
    directions a group's rows do not determine: a regression's rank ``q`` counts the directions it
    keeps, so a column constant within a group costs that group one degree of freedom.

    With ``n_0, n_1`` rows, ranks ``q_0, q_1`` and residual sums of squares ``RSS_0, RSS_1`` in the
    groups and rank ``q_p`` and ``RSS_p`` on the pooled rows, the Chow statistic ``((RSS_p - RSS_0 -
    RSS_1) / df1) / ((RSS_0 + RSS_1) / df2)`` is referred to ``F(df1, df2)``, upper tail, with
    ``df1 = q_0 + q_1 - q_p`` and ``df2 = n_0 + n_1 - q_0 - q_1``; the ratio of residual variances
    ``(RSS_1 / (n_1 - q_1)) / (RSS_0 / (n_0 - q_0))`` is referred to ``F(n_1 - q_1, n_0 - q_0)``,
    two-sided. Both are exact under the null with Gaussian errors. When ``df1 = 0`` the groups share
    no direction to compare, and the variance test alone gives the p-value. A group with no more
    rows than its regression's rank gives ``1``: nothing to compare.
    """

    def __init__(self, X: np.ndarray, groups: np.ndarray, *, indep_test: str, **_: Any) -> None:
        _two_groups(groups, "gaussian")
        if indep_test == "normal_scores":
            X = normal_scores(X)
        read = groups >= 0
        Z = X[read]
        spread = Z.std(axis=0)
        Z = (Z - Z.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
        Z = np.column_stack([np.ones(Z.shape[0]), Z])
        g = groups[read]
        self._grams = [Z[g == k].T @ Z[g == k] for k in (0, 1)]
        self._sizes = [int((g == k).sum()) for k in (0, 1)]

    def __call__(self, y: int, condition_set: Iterable[int] = ()) -> float:
        cols = [0, *(int(s) + 1 for s in sorted(set(condition_set)))]
        target = int(y) + 1
        fits = [_regression(G, cols, target) for G in self._grams]
        (rss0, q0), (rss1, q1) = fits
        n0, n1 = self._sizes
        df2 = n0 + n1 - q0 - q1
        if n0 <= q0 or n1 <= q1 or df2 <= 0:
            return 1.0
        p_var = _two_sided(rss1 / (n1 - q1), rss0 / (n0 - q0), n1 - q1, n0 - q0)
        rss_p, q_p = _regression(self._grams[0] + self._grams[1], cols, target)
        df1 = q0 + q1 - q_p
        if df1 <= 0:
            return min(1.0, 2.0 * p_var)
        within = rss0 + rss1
        between = max(rss_p - within, 0.0)
        if within > 0:
            p_coef = float(stats.f.sf((between / df1) / (within / df2), df1, df2))
        else:
            p_coef = 1.0 if between == 0 else 0.0
        return min(1.0, 2.0 * min(p_coef, p_var))


def _regression(gram: np.ndarray, cols: list[int], target: int) -> tuple[float, int]:
    """The residual sum of squares and the rank of the least-squares fit of column ``target`` on
    ``cols``, from the Gram matrix ``gram`` of the rows."""
    A = gram[np.ix_(cols, cols)]
    b = gram[cols, target]
    w, V = np.linalg.eigh(A)
    keep = w > _RANK_RTOL * max(w.max(), 0.0)
    coef = V[:, keep].T @ b
    rss = gram[target, target] - float(coef @ (coef / w[keep]))
    return max(rss, 0.0), int(keep.sum())


def _two_sided(s1: float, s0: float, df_1: int, df_0: int) -> float:
    """The two-sided p-value of the variance ratio ``s1 / s0`` on ``F(df_1, df_0)``."""
    if s0 == 0 and s1 == 0:
        return 1.0
    ratio = np.inf if s0 == 0 else s1 / s0
    lower, upper = stats.f.cdf(ratio, df_1, df_0), stats.f.sf(ratio, df_1, df_0)
    return float(min(1.0, 2.0 * min(lower, upper)))


class IndicatorInvariance:
    """The ``"indicator"`` test: the conditional-independence test ``indep_test`` of ``x_y`` and a
    0/1 group indicator given ``x_S``, on the rows of both groups.

    The indicator is the last column of the data the test is built over, categorical for the
    mixed-data tests; ``seed`` reaches the randomized tests.
    """

    def __init__(
        self,
        X: np.ndarray,
        groups: np.ndarray,
        *,
        indep_test: str,
        categorical: npt.ArrayLike | None = None,
        seed: int | None = None,
        degrees_of_freedom: str = "mean",
    ) -> None:
        _two_groups(groups, "indicator")
        read = groups >= 0
        data = np.column_stack([X[read], groups[read].astype(np.float64)])
        mask = None
        if indep_test in MIXED_INDEP_TESTS:
            given = np.zeros(X.shape[1], dtype=bool) if categorical is None else categorical
            mask = np.append(np.asarray(given, dtype=bool), True)
        self._test = make_indep_test(
            indep_test, data, categorical=mask, seed=seed, degrees_of_freedom=degrees_of_freedom
        )
        self._indicator = X.shape[1]

    def __call__(self, y: int, condition_set: Iterable[int] = ()) -> float:
        return float(self._test(int(y), self._indicator, [int(s) for s in condition_set]))


class SingularRegression(ValueError):
    """Raised when the rows of a regression leave its coefficients undetermined.

    ``group`` is the group whose rows a regression left out, or ``None`` for the regression on
    every row read.
    """

    group: int | None = None


class RegressionInvariance:
    """The regressions ICP's two tests share: Gram matrices of ``[1, x_others, x_y]`` per group.

    The Gram rows of a query are the intercept, then the columns of ``condition_set``; the Gram
    matrices of one ``y`` are built on its first query and kept.
    """

    name = ""

    def __init__(self, X: np.ndarray, groups: np.ndarray, **_: Any) -> None:
        read = groups >= 0
        self._X, self._groups = X[read], groups[read]
        if np.unique(self._groups).size < 2:
            raise ValueError(f"invariance_test {self.name!r} compares two or more groups")
        self._K = int(self._groups.max()) + 1
        self._sizes = np.bincount(self._groups, minlength=self._K).astype(np.float64)
        self._grams: dict[int, np.ndarray] = {}

    def __call__(self, y: int, condition_set: Iterable[int] = ()) -> float:
        sets = np.array([sorted({int(s) for s in condition_set})], dtype=np.intp)
        return float(self.fit(int(y), sets)[0][0])

    def fit(
        self, y: int, sets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The p-values of ``sets``, an ``(m, s)`` array of columns, with their pooled regressions:
        each set's Gram block of ``[1, x_S]``, coefficients, and residual sum of squares."""
        grams = self._gram(y)
        rows = np.column_stack([np.zeros(len(sets), dtype=np.intp), sets - (sets > y) + 1])
        pooled, beta, rss = _pooled(grams, rows)
        return self._p_values(grams, rows, pooled, beta, rss), pooled, beta, rss

    def _gram(self, y: int) -> np.ndarray:
        if y not in self._grams:
            n, d = self._X.shape
            others = np.array([j for j in range(d) if j != y], dtype=np.intp)
            columns = np.column_stack([self._X[:, others], self._X[:, y]])
            Z = np.column_stack([np.ones(n), columns - columns.mean(axis=0)])
            g = self._groups
            self._grams[y] = np.stack([Z[g == k].T @ Z[g == k] for k in range(self._K)])
        return self._grams[y]

    def _p_values(self, grams, rows, pooled, beta, rss) -> np.ndarray:
        raise NotImplementedError


class ApproximateInvariance(RegressionInvariance):
    """The ``"icp_approximate"`` test, the paper's Method II, from the pooled residuals."""

    name = "icp_approximate"

    def _p_values(self, grams, rows, pooled, beta, rss) -> np.ndarray:
        sizes, y = self._sizes, grams.shape[1] - 1
        A = grams[:, rows[:, :, None], rows[:, None, :]]
        # Each group's residual sum and sum of squares, from its Gram block.
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
        p_group = 2.0 * np.minimum(p_mean, p_variance)
        factor = 1.0 if len(sizes) == 2 else float(len(sizes))
        return np.minimum(1.0, factor * p_group.min(axis=0))


class ExactInvariance(RegressionInvariance):
    """The ``"icp_exact"`` test, the paper's Method I, Chow's test of each group's rows."""

    name = "icp_exact"

    def _p_values(self, grams, rows, pooled, beta, rss) -> np.ndarray:
        sizes, y = self._sizes, grams.shape[1] - 1
        G = grams.sum(axis=0)
        k = rows.shape[1]
        p_group = np.empty((len(sizes), len(rows)))
        for e in range(len(sizes)):
            out = G - grams[e]
            b = out[rows, y]
            A = pooled - grams[e][rows[:, :, None], rows[:, None, :]]
            beta_out = _solve(A, b, e)
            rss_out = out[y, y] - (b * beta_out).sum(axis=1)
            n_e, df = sizes[e], sizes.sum() - sizes[e] - k
            statistic = np.maximum(rss - rss_out, 0.0) / n_e / (rss_out / df)
            p_group[e] = stats.f.sf(statistic, n_e, df)
        return np.minimum(1.0, len(sizes) * p_group.min(axis=0))


def _solve(A: np.ndarray, b: np.ndarray, group: int | None) -> np.ndarray:
    """Solve each system ``A x = b``; :class:`SingularRegression` when one has no solution."""
    try:
        return np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        where = "every row" if group is None else f"the rows outside group {group}"
        error = SingularRegression(
            f"the regressors are linearly dependent on {where}, so a regression's coefficients "
            "are undetermined"
        )
        error.group = group
        raise error from None


def _pooled(grams: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each query's Gram block on every row read, coefficients, and residual sum of squares."""
    G = grams.sum(axis=0)
    y = G.shape[0] - 1
    A = G[rows[:, :, None], rows[:, None, :]]
    b = G[rows, y]
    beta = _solve(A, b, None)
    return A, beta, G[y, y] - (b * beta).sum(axis=1)


#: The registry: each ``invariance_test`` name and the factory that builds it, called as
#: ``factory(X, groups, *, indep_test, categorical, seed)``. A method takes the names it supports
#: as its ``invariance_test`` choices.
INVARIANCE_TESTS: dict[str, Callable[..., InvarianceTest]] = {
    "gaussian": GaussianInvariance,
    "indicator": IndicatorInvariance,
    "icp_approximate": ApproximateInvariance,
    "icp_exact": ExactInvariance,
}
