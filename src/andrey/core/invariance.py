"""Invariance tests: whether a variable's conditional distribution is the same in groups of rows.

An invariance test is built once over the data and a group per row, and called per query as
``test(y, condition_set)``. It returns the p-value of the null hypothesis that the conditional
distribution of column ``y`` given the columns ``condition_set`` is the same in every group; a
larger p-value means more invariant. The caller compares the p-value with its level, rejects
invariance at ``p <= level`` as a conditional-independence search rejects independence, and keeps
any memo. :data:`INVARIANCE_TESTS` maps each ``invariance_test`` name to the factory that builds
it, and :func:`make_invariance_test` builds one by name.

The two shipped tests compare ``K >= 2`` groups, by the regression of ``x_y`` on ``[1, x_S]``
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

References
----------
Chow, G. C. (1960). Tests of equality between sets of coefficients in two linear regressions.
Econometrica 28(3), 591-605.

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

__all__ = [
    "INVARIANCE_TESTS",
    "InvarianceTest",
    "RegressionInvariance",
    "SingularRegression",
    "make_invariance_test",
]


class InvarianceTest(Protocol):
    """An invariance test over fixed data and groups, called as ``test(y, condition_set)``.

    Returns the p-value of the null hypothesis that the conditional distribution of column ``y``
    given ``condition_set`` is the same in every group; a larger p-value means more invariant.
    """

    def __call__(self, y: int, condition_set: Iterable[int] = ()) -> float: ...


def make_invariance_test(
    name: str,
    X: npt.ArrayLike,
    groups: npt.ArrayLike,
    *,
    indep_test: str,
    categorical: npt.ArrayLike | None = None,
    seed: int | None = None,
) -> InvarianceTest:
    """Build the invariance test ``name`` over the rows of ``X`` and their ``groups``.

    ``X`` is the checked ``(n_samples, n_variables)`` data and ``groups[i]`` the group of row ``i``,
    ``0..G-1``, or ``-1`` for a row the test does not read. ``indep_test`` is the fit's
    conditional-independence test, ``categorical`` its mask of the categorical columns (for the
    mixed-data tests), and ``seed`` reaches its random draws, as for
    :func:`~andrey.core.ci.make_indep_test`. A factory reads what its test needs and ignores the
    rest. Raises ``ValueError`` when ``name`` is not in :data:`INVARIANCE_TESTS`, and when a
    ``K``-group test is given fewer than two groups.
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
    )


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
    "icp_approximate": ApproximateInvariance,
    "icp_exact": ExactInvariance,
}
