"""Conditional-independence test substrate: partial-correlation, chi-square, and mixed-data tests.

The constraint-based discovery family (PC / FCI / CDNOD) consumes a conditional-independence test
through the :class:`CITest` contract: an object constructed once over the data and called as
``test(x, y, condition_set)`` returning a p-value, where a larger p-value means more independent.

:class:`PartialCorrelationTest` is the base of the two partial-correlation tests. It precomputes
the sample correlation matrix once, then answers each query from the conditioned partial
correlation of columns ``x`` and ``y`` given ``condition_set``. Empty and single-element
conditioning sets -- the bulk of a constraint-based search -- use the closed-form (first-order)
partial correlation and skip the matrix inversion entirely; larger sets invert the correlation
submatrix. numpy float64 is the oracle. A query is canonicalized (``x``, ``y`` ordered, the
conditioning set sorted unique) so its p-value does not depend on argument order, and nothing is
memoized: a test object holds the correlation matrix and the sample size, whatever it is asked.
The scalar path, the batched path (:meth:`~PartialCorrelationTest.batched_call`), the all-pairs
and first-order closed forms, the many-query path
(:meth:`~PartialCorrelationTest.statistic_pvalues`), and the path for queries already canonical,
which the skeleton search and GFCI's separating-set search take, return the same p-values exactly.

The two partial-correlation tests differ only in how a partial correlation ``r`` and its degrees of
freedom become a p-value. :class:`FisherZ` applies the normal distribution to Fisher's z-transform
of ``r``; :class:`StudentT` applies the Student-t distribution to ``t = r * sqrt(df / (1 - r**2))``.

:class:`ChiSquare` implements the Pearson chi-square and G-square tests on category codes: one
contingency table of ``x`` and ``y`` per observed configuration of the conditioning set. Its
batched path, all-pairs marginal path, and first-order path count many queries' tables together
and return the scalar p-values exactly.

:class:`DGTest` and :class:`CGTest` test mixed data, continuous and categorical columns marked by a
mask, by the likelihood ratio of the degenerate Gaussian and the conditional Gaussian models of
:mod:`andrey.core.mixed`. :class:`CGTest` has the same three many-query paths, fitting their
queries' sets together, with the scalar p-values exactly; :class:`DGTest` answers one query at a
time.

The registry at the end maps each ``indep_test`` name to a factory. The nonlinear tests live in
their own modules (KCI and RCoT in :mod:`andrey.core.ci_kernel`, CMIknn in
:mod:`andrey.core.ci_cmiknn`, GPDC in :mod:`andrey.core.ci_gpdc`) and are imported when first
built. On data with ``NaN``, :func:`make_indep_test` wraps the registered test in the test-wise
deletion of :mod:`andrey.core.ci_missing`, which answers each query on the rows observed in all of
its columns.

References
----------
Spirtes, P., Glymour, C., and Scheines, R. (2000). Causation, Prediction, and Search, 2nd ed.
MIT Press.

Tsamardinos, I., Brown, L. E., and Aliferis, C. F. (2006). The max-min hill-climbing Bayesian
network structure learning algorithm. Machine Learning 65(1), 31-78.
https://doi.org/10.1007/s10994-006-6889-7

Andrews, B., Ramsey, J., and Cooper, G. F. (2018). Scoring Bayesian networks of mixed variables.
International Journal of Data Science and Analytics 6(1), 3-18.
https://doi.org/10.1007/s41060-017-0085-7

Andrews, B., Ramsey, J., and Cooper, G. F. (2019). Learning high-dimensional directed acyclic
graphs with mixed data-types. Proceedings of Machine Learning Research 104, 4-21.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache, partial
from typing import Any, Protocol, Self, cast, runtime_checkable

import numpy as np
import numpy.typing as npt
from scipy.special import chdtrc, ndtr, stdtr

from . import categorical, stats
from .mixed import ConditionalGaussian, DegenerateGaussian
from .score import copies_error, copies_message, copy_groups

__all__ = [
    "CITest",
    "FisherZ",
    "StudentT",
    "register_indep_test",
    "available_indep_tests",
    "make_indep_test",
]

# Partial correlations are clamped just inside (-1, 1) so Fisher's z-transform stays finite on
# deterministic or tiny-sample data.
_ONE_MINUS_EPS = 1.0 - np.finfo(np.float64).eps

# A correlation submatrix is singular when its smallest eigenvalue is at most this fraction of its
# largest.
_SINGULAR_RTOL = 1e-12
# How far the inverse bound in _invert_nonsingular stays above that threshold; the factor absorbs
# the rounding of the computed inverse and of the eigenvalues.
_BOUND_MARGIN = 1e4


class SingularCorrelationError(ValueError):
    """A ``[x, y, *S]`` correlation submatrix is singular to working precision.

    Its smallest eigenvalue is at most ``_SINGULAR_RTOL`` times its largest, so its inverse cannot
    give the partial correlation. :meth:`PartialCorrelationTest.residual_statistic_pvalues` gives it
    from the data instead. When the data hold columns that repeat others up to scale and shift,
    ``groups`` holds them as :func:`~andrey.core.score.copy_groups` returns them, and the message
    gives :func:`~andrey.core.score.copies_message` of those groups as the cause; otherwise
    ``groups`` is empty.
    """

    groups: tuple[tuple[int, ...], ...] = ()


@runtime_checkable
class CITest(Protocol):
    """A conditional-independence test callable over fixed data.

    An implementer is constructed once over the data matrix and invoked per query as
    ``test(x, y, condition_set)``, returning the test's p-value for ``x`` independent of ``y`` given
    ``condition_set``. A larger p-value indicates more independence.
    """

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float: ...


class PartialCorrelationTest:
    """Base of the partial-correlation tests; a subclass maps ``(r, df)`` to a p-value.

    ``__init__`` records the sample size and precomputes the full column correlation matrix once.
    Each call forms the partial correlation ``r`` of ``x`` and ``y`` given the conditioning set --
    directly (``|S| == 0``), by the first-order closed form (``|S| == 1``), or by inverting the
    ``[x, y, *S]`` correlation submatrix (``|S| >= 2``) -- clamps it just inside ``(-1, 1)``, and
    reports the subclass's p-value for ``r`` on ``n - |S| - k`` degrees of freedom. The ``|S| <= 1``
    closed forms do not invert a submatrix, so on exactly-collinear columns they return a finite
    (near-1) partial correlation where an always-invert path would instead raise on a singular
    matrix.

    Queries are canonicalized (``x``, ``y`` ordered, ``condition_set`` sorted unique), so the
    p-value is independent of argument order. No query is memoized: the skeleton tests each
    conditioning set of a pair once, so retained results would grow with the number of subsets
    visited. Later queries are recomputed.

    The scalar path, :meth:`batched_call`, :meth:`statistic_pvalues`, :meth:`marginal_pvalues`,
    :meth:`first_order_pvalues`, and ``_canonical_pvalues``, which the skeleton search and GFCI's
    separating-set search call with queries canonical as built, use identical partial-correlation,
    clip, and p-value arithmetic.
    Their p-values match exactly, so repeated queries take the same ``p > alpha`` decision across
    call paths, including at the threshold.
    """

    # The data's groups of copies, set by the first singular inversion.
    _copies: tuple[tuple[int, ...], ...]

    def __init__(self, data: np.ndarray) -> None:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f"data must be a 2-D (n_samples, n_features) array, got ndim={arr.ndim}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("data contains NaN or Inf; cannot run a partial-correlation test")
        self._n: int = arr.shape[0]
        self._corr: np.ndarray = np.asarray(stats.corrcoef(arr, rowvar=False), dtype=np.float64)

    def _degrees_of_freedom(self, size: int) -> int:
        """The degrees of freedom of a test given ``size`` conditioners; raises when unusable."""
        raise NotImplementedError

    @staticmethod
    def _pvalue(r: np.ndarray | float, df: int) -> np.ndarray | float:
        """The p-value of the clamped partial correlation(s) ``r`` on ``df`` degrees of freedom."""
        raise NotImplementedError

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the p-value for ``x`` independent of ``y`` given ``condition_set``.

        ``condition_set`` is canonicalized to sorted unique ints; ``x`` and ``y`` are ordered so the
        p-value is independent of argument order. A larger p-value means more independence.
        """
        return self.statistic_pvalue(x, y, condition_set)[1]

    def statistic_pvalue(
        self, x: int, y: int, condition_set: Iterable[int] = ()
    ) -> tuple[float, float]:
        """Return the partial correlation and p-value of ``x`` and ``y`` given ``condition_set``.

        The p-value is the one :meth:`__call__` returns. The partial correlation is clamped just
        inside ``(-1, 1)`` and is symmetric in ``x`` and ``y``.
        """
        x_int, y_int, cond = _canonical(x, y, condition_set)
        df = self._degrees_of_freedom(len(cond))
        r = self._partial_correlation(x_int, y_int, cond)
        return r, float(self._pvalue(r, df))

    def _partial_correlation(self, x_int: int, y_int: int, cond: list[int]) -> float:
        """Partial correlation of columns ``x_int`` and ``y_int`` given ``cond``, clamped.

        The arguments are canonical (see :func:`_canonical`). The result lies just inside
        ``(-1, 1)``.
        """
        corr = self._corr
        if not cond:
            r = float(corr[x_int, y_int])  # marginal correlation; no inversion
        elif len(cond) == 1:  # first-order partial correlation, closed form
            z_idx = cond[0]
            rxy = float(corr[x_int, y_int])
            rxz = float(corr[x_int, z_idx])
            ryz = float(corr[y_int, z_idx])
            r = (rxy - rxz * ryz) / math.sqrt((1.0 - rxz * rxz) * (1.0 - ryz * ryz))
        else:
            var = (x_int, y_int, *cond)
            inv = self._invert(corr[np.ix_(var, var)][None])[0]
            r = -float(inv[0, 1]) / math.sqrt(abs(float(inv[0, 0]) * float(inv[1, 1])))
        return (
            -_ONE_MINUS_EPS if r < -_ONE_MINUS_EPS else _ONE_MINUS_EPS if r > _ONE_MINUS_EPS else r
        )

    def copies(self) -> tuple[tuple[int, ...], ...]:
        """The groups of data columns that repeat one another up to scale and shift.

        :func:`~andrey.core.score.copy_groups` of the test's correlation matrix, found once per
        test; a single column's correlation is a scalar and has none.
        """
        if not hasattr(self, "_copies"):
            self._copies = copy_groups(np.atleast_2d(self._corr), self._n)
        return self._copies

    def _invert(self, stack: np.ndarray) -> np.ndarray:
        """:func:`_invert_nonsingular` of ``stack``; a singular error names the data's copies."""
        try:
            return _invert_nonsingular(stack)
        except SingularCorrelationError as error:
            copies = self.copies()
            if not copies:
                raise
            named = SingularCorrelationError(
                f"correlation matrix is singular because {copies_message(copies)}"
            )
            named.groups = copies
            raise named from error

    def batched_call(self, x: int, y: int, condition_sets: Iterable[Iterable[int]]) -> list[float]:
        """Return p-values for one ``(x, y)`` pair over many conditioning sets.

        The result list matches the order of ``condition_sets``. Each set is canonicalized as in
        :meth:`__call__`; queries are grouped by set size so each size is evaluated in one
        vectorized shot, and the p-values match :meth:`__call__` exactly. A single-set
        batch takes the scalar path. Nothing is retained once the list is returned.
        """
        condition_sets = list(condition_sets)
        n = len(condition_sets)
        if n == 0:
            return []
        if n == 1:
            return [self(x, y, condition_sets[0])]

        x_int, y_int = (int(x), int(y)) if x < y else (int(y), int(x))
        results: list[float | None] = [None] * n
        groups: dict[int, list[tuple[int, list[int]]]] = {}
        for i, condition_set in enumerate(condition_sets):
            cond = sorted(set(int(c) for c in condition_set))
            assert x_int not in cond and y_int not in cond, "x, y cannot be in condition_set."
            groups.setdefault(len(cond), []).append((i, cond))

        for members in groups.values():
            pvals = self._pvalues(x_int, y_int, [cond for _, cond in members])
            for (idx, _), p in zip(members, pvals):
                results[idx] = float(p)

        return cast("list[float]", results)

    def separating_prefix(
        self, x: int, y: int, condition_sets: Sequence[Iterable[int]], alpha: float
    ) -> list[float]:
        """Return the p-values of ``condition_sets``, in order, up to and including the first above
        ``alpha``: the answers a loop of scalar calls that stops at the first separating set gets.

        The sets are evaluated together (:func:`_separating_prefix`); each p-value equals
        :meth:`__call__`'s.
        """
        return _separating_prefix(self, x, y, condition_sets, alpha)

    def _pvalues(self, x_int: int, y_int: int, conds: list[list[int]]) -> np.ndarray:
        """P-values for stacked queries on one pair sharing one conditioning-set size.

        ``conds`` are sorted-unique int lists of equal length.
        """
        size = len(conds[0])
        cond_array = np.asarray(conds, dtype=np.intp).reshape(len(conds), size)
        x = np.full(len(conds), x_int, dtype=np.intp)
        return self.statistic_pvalues(x, np.full_like(x, y_int), cond_array)[1]

    def statistic_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Partial correlations and p-values of many queries sharing one conditioning-set size.

        ``x`` and ``y`` are ``(N,)`` int arrays and ``conditions`` the ``(N, m)`` int array whose
        row ``i`` is the conditioning set of query ``i``: distinct columns that exclude ``x[i]``
        and ``y[i]``. Entry ``i`` of each result matches ``statistic_pvalue(x[i], y[i],
        conditions[i])`` exactly. Empty and single conditioners use the closed-form partial
        correlation; larger sets build the ``(N, m + 2, m + 2)`` stack of correlation submatrices
        and invert it once. Raises :class:`ValueError` when the degrees of freedom are unusable or
        any submatrix is singular.
        """
        x = np.asarray(x, dtype=np.intp)
        y = np.asarray(y, dtype=np.intp)
        lo, hi = np.minimum(x, y), np.maximum(x, y)
        conds = np.sort(np.asarray(conditions, dtype=np.intp), axis=1)
        return self._canonical_statistic_pvalues(lo, hi, conds)

    def _canonical_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> np.ndarray:
        """P-values of many canonical queries sharing one conditioning-set size.

        ``x`` and ``y`` are ``(N,)`` int arrays with ``x[i] < y[i]``, and ``conditions`` the
        ``(N, m)`` int array whose row ``i`` is sorted and distinct and excludes ``x[i]`` and
        ``y[i]``, as the skeleton searches build their queries. Nothing is reordered or checked;
        entry ``i`` matches ``self(x[i], y[i], conditions[i])`` exactly.
        """
        return self._canonical_statistic_pvalues(x, y, conditions)[1]

    def _canonical_statistic_pvalues(
        self, lo: np.ndarray, hi: np.ndarray, conds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`statistic_pvalues` of canonical queries, ``lo[i] < hi[i]`` and each row of
        ``conds`` sorted."""
        df = self._degrees_of_freedom(conds.shape[1])
        corr = self._corr
        if conds.shape[1] == 0:
            r = corr[lo, hi]
        elif conds.shape[1] == 1:
            z_idx = conds[:, 0]
            rxz = corr[lo, z_idx]
            ryz = corr[hi, z_idx]
            r = (corr[lo, hi] - rxz * ryz) / np.sqrt((1.0 - rxz * rxz) * (1.0 - ryz * ryz))
        else:
            var = np.empty((len(lo), 2 + conds.shape[1]), dtype=np.intp)
            var[:, 0] = lo
            var[:, 1] = hi
            var[:, 2:] = conds
            inv = self._invert(corr[var[:, :, None], var[:, None, :]])
            r = -inv[:, 0, 1] / np.sqrt(np.abs(inv[:, 0, 0] * inv[:, 1, 1]))
        r = np.clip(r, -_ONE_MINUS_EPS, _ONE_MINUS_EPS)
        return r, np.asarray(self._pvalue(r, df))

    def residual_statistic_pvalues(
        self, data: np.ndarray, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Partial correlations and p-values of many queries, from least-squares residuals.

        ``data`` is the matrix the test was built from, and ``x``, ``y``, and ``conditions`` are
        as in :meth:`statistic_pvalues`. Each column is centered and scaled to unit variance; the
        partial correlation is the correlation of the residuals of ``x`` and of ``y`` after a
        least-squares fit on the conditioning columns, clamped just inside ``(-1, 1)``, and the
        p-value uses the degrees of freedom of :meth:`statistic_pvalues`. Without rounding it
        equals the partial correlation :meth:`statistic_pvalues` takes from the inverse of the
        correlation submatrix. It stays defined when the conditioning columns are collinear, where
        that inverse does not exist, because the residuals of a least-squares fit do not depend
        on which of the equivalent solutions it returns. Raises :class:`SingularCorrelationError`
        when the residual norm of ``x`` or ``y`` is at most ``eps * max(n, k)`` times the column's
        norm, for ``n`` rows, ``k`` conditioning columns, and the float64 machine epsilon
        ``eps``: the rank cutoff of the least-squares fit, below which the residual is rounding
        and the column lies in the span of the conditioning columns.
        """
        arr = np.asarray(data, dtype=np.float64)
        x = np.asarray(x, dtype=np.intp)
        y = np.asarray(y, dtype=np.intp)
        conds = np.asarray(conditions, dtype=np.intp).reshape(len(x), -1)
        df = self._degrees_of_freedom(conds.shape[1])
        r = np.empty(len(x), dtype=np.float64)
        # Columns are scaled to a squared norm of n, so a residual's squared norm at the cutoff is
        # n * (eps * max(n, k))**2.
        cutoff = len(arr) * (np.finfo(np.float64).eps * max(len(arr), conds.shape[1])) ** 2
        for k in range(len(x)):
            columns = arr[:, [int(x[k]), int(y[k]), *conds[k].tolist()]]
            columns = columns - columns.mean(axis=0)
            columns = columns / np.sqrt((columns * columns).mean(axis=0))
            pair, fit = columns[:, :2], columns[:, 2:]
            residual = pair - fit @ np.linalg.lstsq(fit, pair, rcond=None)[0]
            spread = (residual * residual).sum(axis=0)
            if not (spread > cutoff).all():
                raise SingularCorrelationError(
                    "a tested column lies in the span of its conditioning columns; cannot run a "
                    "partial-correlation test"
                )
            r[k] = (residual[:, 0] @ residual[:, 1]) / math.sqrt(spread[0] * spread[1])
        r = np.clip(r, -_ONE_MINUS_EPS, _ONE_MINUS_EPS)
        return r, np.asarray(self._pvalue(r, df))

    def marginal_pvalues(self) -> np.ndarray:
        """All-pairs marginal (``|S| == 0``) p-values as a ``(d, d)`` matrix.

        Vectorizes the marginal closed form over the whole correlation matrix in one shot;
        upper-triangle entry ``[x, y]`` matches ``self(x, y, ())`` exactly, using the same clip
        and p-value arithmetic. The lower triangle can inherit rounding asymmetry from the
        correlation matrix. The diagonal is self-correlation and is not meaningful.
        """
        df = self._degrees_of_freedom(0)
        r = np.clip(self._corr, -_ONE_MINUS_EPS, _ONE_MINUS_EPS)
        return np.asarray(self._pvalue(r, df))

    def first_order_pvalues(self, x: int, y: int, zs: Iterable[int]) -> np.ndarray:
        """First-order (``|S| == 1``) p-values for pair ``(x, y)`` over single conditioners ``zs``.

        One vectorized evaluation of the first-order partial-correlation closed form. For
        ``x < y``, the ``i``-th entry matches ``self(x, y, (zs[i],))`` exactly. Pairs are evaluated
        in their supplied order; ``zs`` must exclude ``x`` and ``y``.
        """
        conds = [[int(z)] for z in zs]
        if not conds:
            return np.empty(0, dtype=np.float64)
        return self._pvalues(int(x), int(y), conds)

    @classmethod
    def from_corr(cls, corr: np.ndarray, n_samples: int) -> Self:
        """Build a test from a precomputed correlation matrix and sample size, re-reading no data.

        The parallel skeleton search seeds each worker with the parent's exact ``_corr`` float64
        bytes, so every process feeds identical values into the closed forms and stays bit-identical
        to the serial oracle. ``corr`` must be a square ``(d, d)`` sample-correlation matrix and
        ``n_samples`` its observation count.
        """
        obj = cls.__new__(cls)
        obj._n = int(n_samples)
        obj._corr = np.asarray(corr, dtype=np.float64)
        return obj

    def to_seed(self) -> tuple[np.ndarray, int]:
        """The ``(correlation_matrix, n_samples)`` reconstructing this test via :meth:`from_corr`.

        The parallel skeleton search seeds workers with this rather than pickling the whole test:
        a worker rebuilds an equivalent test over the parent's exact correlation bytes.
        """
        return self._corr, self._n


class FisherZ(PartialCorrelationTest):
    """Fisher-Z partial-correlation conditional-independence test.

    Applies Fisher's z-transform to the partial correlation ``r`` and reports
    ``p = 2 * (1 - Phi(sqrt(n - |S| - 3) * |z|))`` with ``z = atanh(r)`` and ``Phi`` the
    standard-normal CDF. A query needs ``n >= |S| + 3``. See :class:`PartialCorrelationTest` for
    the partial correlation, the query paths, and their exact agreement.
    """

    def _degrees_of_freedom(self, size: int) -> int:
        df = self._n - size - 3
        if df < 0:
            raise ValueError(
                f"Fisher-Z degrees of freedom is negative (n={self._n}, |S|={size}); "
                "need n > |S| + 3"
            )
        return df

    @staticmethod
    def _pvalue(r: np.ndarray | float, df: int) -> np.ndarray | float:
        z = 0.5 * np.log((1.0 + r) / (1.0 - r))
        return 2.0 * (1.0 - ndtr(np.sqrt(df) * np.abs(z)))


class StudentT(PartialCorrelationTest):
    """Student-t partial-correlation conditional-independence test.

    Reports the same partial correlation ``r`` as :class:`FisherZ` with an exact Student-t p-value
    on ``df = n - |S| - 2`` degrees of freedom: ``t = r * sqrt(df / (1 - r**2))`` and
    ``p = 2 * P(T_df > |t|)``. Its p-values differ slightly from Fisher-Z's. ``r`` is clamped just
    inside ``(-1, 1)``, so a deterministic relation gives a tiny finite p-value. A query needs
    ``n > |S| + 2``. See :class:`PartialCorrelationTest` for the query paths.
    """

    def _degrees_of_freedom(self, size: int) -> int:
        df = self._n - size - 2
        if df <= 0:
            raise ValueError(
                f"Student-t degrees of freedom is not positive (n={self._n}, |S|={size}); "
                "need n > |S| + 2"
            )
        return df

    @staticmethod
    def _pvalue(r: np.ndarray | float, df: int) -> np.ndarray | float:
        t = r * np.sqrt(df / (1.0 - r * r))
        return 2.0 * stdtr(df, -np.abs(t))


# A joint table of at most this many cells is counted densely; a larger one counts only the cells
# that occur, so memory stays proportional to the rows whatever the cardinalities.
_DENSE_CELLS = 1 << 20
# The batched chi-square path takes a test whose conditioning set's configurations times its table
# cells are at most max(_TABLE_ROWS * n, _TABLE_MIN), and the scalar path above.
_TABLE_ROWS = 4
_TABLE_MIN = 1 << 16
# The most row indices, table cells, or Gram-matrix entries one batched count holds.
_BATCH_CELLS = 1 << 22


class ChiSquare:
    """Pearson chi-square or G-square test of conditional independence on category codes.

    ``__init__`` re-encodes each column to the codes ``0..k-1`` (see
    :func:`~andrey.core.categorical.encode`), so any finite values work, each distinct value of a
    column one category. Each call counts the joint table of ``x`` and ``y`` in every configuration
    of ``condition_set`` that occurs in the data, and compares it with the table expected under
    independence within that configuration, ``n_sx * n_sy / n_s``. The statistic is Pearson's
    ``sum (O - E)**2 / E`` or, with ``g_square``, ``2 * sum O * log(O / E)``, summed over the
    configurations [1]_ [2]_.

    The degrees of freedom are ``(r_x - 1) * (r_y - 1)`` summed over the configurations, where
    ``r_x`` and ``r_y`` count the values of ``x`` and ``y`` present in that configuration, so a
    value absent there adds no degree of freedom [2]_. With zero degrees of freedom the p-value is
    ``1``; otherwise it is the chi-square survival function of the statistic.

    Queries are canonicalized as :class:`FisherZ`'s are, and nothing is memoized.

    :meth:`batched_call`, :meth:`first_order_pvalues`, and :meth:`marginal_pvalues` answer many
    queries at once and return the p-values of :meth:`__call__` exactly. They count the tables of
    many queries together, with ``bincount`` over the configurations that occur, or, for the
    marginal tables of all pairs, one product of the category-indicator matrix with itself, whose
    integer counts are exact in float64. Every step after the counting is elementwise on the
    tables, except the sum of a query's statistic over its cells, which is ``np.sum`` over that
    query's cells in the order :meth:`__call__` sums them, so the sum, and a decision on the
    p-value, does not depend on how the queries are batched.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Finite values; each distinct value of a column is one category.
    g_square : bool, default=False
        Use the G-square (likelihood-ratio) statistic instead of Pearson's chi-square.

    References
    ----------
    .. [1] Spirtes, P., Glymour, C., and Scheines, R. (2000). Causation, Prediction, and Search,
       2nd ed. MIT Press.
    .. [2] Tsamardinos, I., Brown, L. E., and Aliferis, C. F. (2006). The max-min hill-climbing
       Bayesian network structure learning algorithm. Machine Learning 65(1), 31-78.
    """

    def __init__(self, data: np.ndarray, *, g_square: bool = False) -> None:
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError(
                f"data must be a 2-D (n_samples, n_features) array, got ndim={arr.ndim}"
            )
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise ValueError("data contains NaN or Inf; cannot run a chi-square test")
        self._n: int = arr.shape[0]
        self._codes, self._cardinalities = categorical.encode(arr)
        # Each column's codes as one contiguous row, for the batched counts; int32 halves the
        # memory those counts stream through.
        self._rows = self._codes.T.astype(np.int32 if self._n < 2**31 else np.int64, order="C")
        self._g_square = bool(g_square)

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the p-value for ``x`` independent of ``y`` given ``condition_set``.

        ``condition_set`` is canonicalized to sorted unique ints; ``x`` and ``y`` are ordered so the
        p-value is independent of argument order. A larger p-value means more independence.
        """
        return self._statistic_pvalue(x, y, condition_set)[1]

    def statistic_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The chi-square or G-square statistics and p-values of many queries, one per row.

        ``x`` and ``y`` are ``(N,)`` int arrays and ``conditions`` the ``(N, m)`` int array of
        conditioning sets; entry ``i`` holds query ``i``'s statistic and :meth:`__call__`'s p-value.
        """
        return per_query(self._statistic_pvalue, x, y, conditions)

    def _statistic_pvalue(
        self, x: int, y: int, condition_set: Iterable[int]
    ) -> tuple[float, float]:
        """The statistic and p-value of one query."""
        x_int, y_int, cond = _canonical(x, y, condition_set)
        codes = self._codes
        kx = int(self._cardinalities[x_int])
        ky = int(self._cardinalities[y_int])
        layer, m = categorical.configurations(codes, self._cardinalities, cond)
        xs = layer * kx + codes[:, x_int]  # (configuration, x) cell of each row
        ys = layer * ky + codes[:, y_int]
        n_sx = np.bincount(xs, minlength=m * kx).reshape(m, kx)
        n_sy = np.bincount(ys, minlength=m * ky).reshape(m, ky)
        n_s = n_sx.sum(axis=1)
        cells = xs * ky + codes[:, y_int]
        if m * kx * ky <= _DENSE_CELLS:
            table = np.bincount(cells, minlength=m * kx * ky)
            occupied = np.flatnonzero(table)
            observed = table[occupied]
        else:
            occupied, observed = np.unique(cells, return_counts=True)
        s, rest = np.divmod(occupied, kx * ky)
        sx, sy = np.divmod(rest, ky)
        expected = n_sx[s, sx] * n_sy[s, sy] / n_s[s]
        if self._g_square:
            statistic = 2.0 * float(np.sum(observed * np.log(observed / expected)))
        else:
            # Each configuration's expected counts sum to its observed ones, so Pearson's sum over
            # every cell, empty ones included, equals this sum over the occupied cells.
            statistic = float(np.sum(observed * (observed / expected))) - float(self._n)
        dof = int(np.sum(((n_sx > 0).sum(axis=1) - 1) * ((n_sy > 0).sum(axis=1) - 1)))
        if dof == 0:
            return statistic, 1.0
        return statistic, float(chdtrc(dof, max(statistic, 0.0)))

    def batched_call(self, x: int, y: int, condition_sets: Iterable[Iterable[int]]) -> list[float]:
        """Return p-values for one ``(x, y)`` pair over many conditioning sets.

        The result list matches the order of ``condition_sets``, and each p-value equals
        :meth:`__call__`'s. A conditioning set is canonicalized as in :meth:`__call__`. The tables
        of the sets are counted together (see :class:`ChiSquare`); a set with more than
        ``max(4 * n, 65536)`` possible cells, and a single-set batch, take the scalar path.
        """
        condition_sets = list(condition_sets)
        if len(condition_sets) <= 1:
            return [self(x, y, condition_set) for condition_set in condition_sets]
        x_int, y_int = (int(x), int(y)) if x < y else (int(y), int(x))
        conds = [_canonical(x_int, y_int, condition_set)[2] for condition_set in condition_sets]
        return self._pvalues(x_int, y_int, conds).tolist()

    def separating_prefix(
        self, x: int, y: int, condition_sets: Sequence[Iterable[int]], alpha: float
    ) -> list[float]:
        """Return the p-values of ``condition_sets``, in order, up to and including the first above
        ``alpha``: the answers a loop of scalar calls that stops at the first separating set gets.

        The sets are evaluated together (:func:`_separating_prefix`); each p-value equals
        :meth:`__call__`'s.
        """
        return _separating_prefix(self, x, y, condition_sets, alpha)

    def first_order_pvalues(self, x: int, y: int, zs: Iterable[int]) -> np.ndarray:
        """First-order (``|S| == 1``) p-values for pair ``(x, y)`` over single conditioners ``zs``.

        Entry ``i`` equals ``self(x, y, (zs[i],))``; ``zs`` must exclude ``x`` and ``y``.
        """
        x_int, y_int = (int(x), int(y)) if x < y else (int(y), int(x))
        return self._pvalues(x_int, y_int, [[int(z)] for z in zs])

    def marginal_pvalues(self) -> np.ndarray:
        """All-pairs marginal (``|S| == 0``) p-values as a symmetric ``(d, d)`` matrix.

        Entry ``[x, y]`` equals ``self(x, y, ())``; the diagonal is ``0``. With the categories of
        all columns stacked, column ``v``'s category ``c`` at ``offsets[v] + c``, the Gram matrix of
        the rows' category indicators holds every pair's table: entry ``[i, j]`` counts the rows in
        categories ``i`` and ``j``. Its rows are computed for a few columns ``x`` at a time,
        accumulated over blocks of rows. The counts are integers below ``2**53``, so the float64
        products and sums are exact in any order. A column whose rows of the Gram matrix alone
        exceed the block size takes the scalar path.
        """
        d = self._codes.shape[1]
        out = np.zeros((d, d))
        cards = self._cardinalities
        offsets = np.cumsum(cards) - cards
        total = int(cards.sum())
        x = 0
        while x < d - 1:
            if self._n == 0 or int(cards[x]) * total > _BATCH_CELLS:
                out[x, x + 1 :] = [self(x, y) for y in range(x + 1, d)]
                x += 1
                continue
            stop = x + 1
            while stop < d - 1 and int(cards[x : stop + 1].sum()) * total <= _BATCH_CELLS:
                stop += 1
            self._marginal_block(out, x, stop, offsets, total)
            x = stop
        return out + out.T

    def _marginal_block(
        self, out: np.ndarray, start: int, stop: int, offsets: np.ndarray, total: int
    ) -> None:
        """Fill ``out[x, y]`` for the columns ``start <= x < stop`` and every ``y > x``."""
        n, d = self._codes.shape
        cards = self._cardinalities
        first = int(offsets[start])
        gram = np.zeros((int(offsets[stop - 1] + cards[stop - 1]) - first, total))
        step = max(1, _BATCH_CELLS // total)
        for row in range(0, n, step):
            codes = self._codes[row : row + step]
            indicators = np.zeros((len(codes), total))
            indicators[np.arange(len(codes))[:, None], codes + offsets] = 1.0
            gram += indicators[:, first : first + len(gram)].T @ indicators
        xs, ys = np.triu_indices(d, 1)
        keep = (xs >= start) & (xs < stop)
        xs, ys = xs[keep], ys[keep]
        shapes = cards[xs] * (int(cards.max()) + 1) + cards[ys]
        for shape in np.unique(shapes):
            pairs = np.flatnonzero(shapes == shape)
            kx, ky = int(cards[xs[pairs[0]]]), int(cards[ys[pairs[0]]])
            for at in range(0, pairs.size, max(1, _BATCH_CELLS // (kx * ky))):
                part = pairs[at : at + max(1, _BATCH_CELLS // (kx * ky))]
                rows = offsets[xs[part], None, None] - first + np.arange(kx)[:, None]
                cols = offsets[ys[part], None, None] + np.arange(ky)
                tables = gram[rows, cols].astype(np.int64)
                out[xs[part], ys[part]] = self._table_pvalues(tables, np.arange(part.size + 1))

    def _pvalues(self, x: int, y: int, conds: list[list[int]]) -> np.ndarray:
        """P-values of the pair ``x < y`` given each canonical conditioning set in ``conds``.

        The sets whose tables have at most ``max(4 * n, 65536)`` possible cells are counted in
        blocks, grouped by size (:meth:`_block_pvalues`); the others take :meth:`__call__`.
        """
        out = np.empty(len(conds))
        n = self._n
        cards = self._cardinalities
        kx, ky = int(cards[x]), int(cards[y])
        cells = kx * ky
        limit = max(_TABLE_ROWS * n, _TABLE_MIN)
        by_size: dict[int, list[int]] = {}
        configurations = [math.prod(int(cards[c]) for c in cond) for cond in conds]
        for i, cond in enumerate(conds):
            if n == 0 or configurations[i] * cells > limit:
                out[i] = self(x, y, cond)
            else:
                by_size.setdefault(len(cond), []).append(i)
        if not by_size:
            return out
        target = self._rows[x] * ky + self._rows[y]  # each row's (x, y) cell
        rows_per_block = max(1, _BATCH_CELLS // n)
        for size, members in by_size.items():
            start = 0
            while start < len(members):
                stop, held = start + 1, configurations[members[start]] * cells
                while stop < len(members) and stop - start < rows_per_block:
                    held += configurations[members[stop]] * cells
                    if held > _BATCH_CELLS:
                        break
                    stop += 1
                block = members[start:stop]
                out[block] = self._block_pvalues(
                    target, cells, kx, ky, [conds[i] for i in block], size
                )
                start = stop
        return out

    def _block_pvalues(
        self, target: np.ndarray, cells: int, kx: int, ky: int, conds: list[list[int]], size: int
    ) -> np.ndarray:
        """P-values of one block of conditioning sets of equal ``size``, counted together.

        Row ``r`` of set ``i`` falls in configuration ``offset[i] + code``, with ``code`` the
        mixed-radix code of its conditioning categories, the first column most significant: the
        order :func:`~andrey.core.categorical.configurations` numbers them in, and in each
        configuration in cell ``target[r]``. Only the tables of the configurations that occur are
        kept, in that order: when every configuration's table together holds no more cells than
        the block has rows, all are counted and the empty ones dropped; otherwise the occurring
        configurations are numbered first and only their tables counted.
        """
        sets = np.array(conds, dtype=np.intp).reshape(len(conds), size)
        radix = self._cardinalities[sets]
        q = radix.prod(axis=1)
        offset = np.cumsum(q) - q
        total = int(q.sum())
        dtype = self._rows.dtype if total * cells < 2**31 else np.dtype(np.int64)
        if size:
            code = self._rows[sets[:, 0]].astype(dtype, copy=False)
            for j in range(1, size):
                code *= radix[:, j, None].astype(dtype)
                code += self._rows[sets[:, j]]
            code += offset[:, None].astype(dtype)
        else:
            code = np.repeat(offset[:, None].astype(dtype), self._n, axis=1)
        if total * cells <= code.size:
            code *= dtype.type(cells)
            code += target
            tables = np.bincount(code.ravel(), minlength=total * cells).reshape(total, cells)
            occurs = np.einsum("sc->s", tables) > 0
            # rank[c] counts the configurations up to c that occur.
            rank = np.cumsum(occurs)
            if rank[-1] < total:
                tables = tables[occurs]
        else:
            rank = np.cumsum(np.bincount(code.ravel(), minlength=total) > 0)
            index = rank.astype(dtype, copy=False)[code]
            index -= dtype.type(1)
            index *= dtype.type(cells)
            index += target
            tables = np.bincount(index.ravel(), minlength=int(rank[-1]) * cells)
        bounds = np.concatenate(([0], rank[offset + q - 1]))
        return self._table_pvalues(tables.reshape(-1, kx, ky), bounds)

    def _table_pvalues(self, tables: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """P-values of the tests whose ``(kx, ky)`` tables are stacked in ``tables``.

        Test ``i`` owns ``tables[bounds[i]:bounds[i + 1]]``, one table per configuration that
        occurs, in :meth:`__call__`'s order. Each step repeats :meth:`__call__`'s arithmetic on the
        same integer counts, and each test's statistic is ``np.sum`` over its own cells in the order
        :meth:`__call__` visits them, so every p-value equals :meth:`__call__`'s.
        """
        m, kx, ky = tables.shape
        n_sx = np.einsum("sxy->sx", tables)
        n_sy = np.einsum("sxy->sy", tables)
        n_s = np.einsum("sx->s", n_sx)
        flat = tables.reshape(-1)
        occupied = np.flatnonzero(flat)
        observed = flat[occupied]
        row = occupied // ky  # the cell's (configuration, x) row of n_sx
        s = row // kx
        expected = n_sx.reshape(-1)[row] * n_sy.reshape(-1)[s * ky + occupied - row * ky] / n_s[s]
        if self._g_square:
            values = observed * np.log(observed / expected)
        else:
            values = observed * (observed / expected)
        cuts = np.searchsorted(occupied, bounds * (kx * ky)).tolist()
        sums = np.array([np.sum(values[a:b]) for a, b in zip(cuts[:-1], cuts[1:])])
        statistic = 2.0 * sums if self._g_square else sums - float(self._n)
        present_x = np.einsum("sx->s", (n_sx > 0).view(np.int8), dtype=np.int64)
        present_y = np.einsum("sy->s", (n_sy > 0).view(np.int8), dtype=np.int64)
        per_table = (present_x - 1) * (present_y - 1)
        dof = np.add.reduceat(per_table, bounds[:-1])
        out = np.ones(len(dof))
        live = dof != 0
        out[live] = chdtrc(dof[live], np.maximum(statistic[live], 0.0))
        return out


# A conditional covariance's eigenvalue, on the scale of unit-variance columns, at or below this
# marks a direction the conditioning set determines.
_RANK_TOL = 1e-10
# The most bytes of correlation blocks one stacked evaluation of the degenerate Gaussian test holds.
_DG_STACK = 1 << 22
# The conditioning sets whose regressions one degenerate Gaussian test keeps.
_DG_SETS = 4096


class DGTest:
    """Degenerate Gaussian likelihood-ratio test of conditional independence on mixed data.

    The columns are embedded as :class:`~andrey.core.mixed.DegenerateGaussian` describes: a
    categorical variable with ``k`` categories becomes the indicators of ``k - 1`` of them, and
    every embedded column is treated as Gaussian [1]_. The test of ``x`` and ``y`` given ``S`` is
    the likelihood ratio of the linear model of ``x``'s columns on ``y``'s and ``S``'s against the
    one on ``S``'s alone [2]_: ``-n * sum log(1 - rho_i**2)``, with ``rho_i`` the canonical
    correlations of ``x``'s and ``y``'s columns after the linear regression on ``S``'s. Its degrees
    of freedom are the product of the two variables' numbers of columns, and its p-value is the
    chi-square survival function. A direction of ``x`` or ``y`` that ``S`` determines exactly
    carries no information and adds no degree of freedom; with none left, the p-value is ``1``. On
    continuous columns only, the statistic is ``-n log(1 - r**2)`` with ``r`` the partial
    correlation.

    The statistic does not depend on the columns' units, and queries are canonicalized as
    :class:`FisherZ`'s are, so the p-value does not depend on the argument order.

    :meth:`batched_call` and the skeleton's stacked scan answer many queries at once and return
    :meth:`__call__`'s p-values exactly. Each conditioning set's eigendecomposition is computed
    once for every query that conditions on it, and the last few thousand sets' are kept for the
    calls after, scalar ones included. The queries are then grouped by shape: the numbers
    of columns of ``x``, ``y``, and ``S``, and the directions kept. A group's products, whiteners,
    and singular values are stacked calls: a stacked NumPy product or decomposition runs, on each
    query's matrix, the BLAS or LAPACK call that query's own call runs, on the same values with the
    same memory layout, so every float operation and its order are those of :meth:`__call__`.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Finite values; categorical columns hold their categories as numbers, each distinct value of
        a column one category.
    categorical : array-like of bool, shape (n_features,), optional
        Marks the categorical columns; ``None`` marks none.

    References
    ----------
    .. [1] Andrews, B., Ramsey, J., and Cooper, G. F. (2019). Learning high-dimensional directed
       acyclic graphs with mixed data-types. Proceedings of Machine Learning Research 104, 4-21.
    .. [2] Andrews, B., Ramsey, J., and Cooper, G. F. (2018). Scoring Bayesian networks of mixed
       variables. International Journal of Data Science and Analytics 6(1), 3-18.
    """

    def __init__(self, data: np.ndarray, categorical: npt.ArrayLike | None = None) -> None:
        self._dg = DegenerateGaussian(data, categorical)
        sd = np.sqrt(np.diag(self._dg.cov))
        self._corr = self._dg.cov / np.outer(sd, sd)
        # Each variable's embedded columns are the consecutive ones from _start, _width of them.
        self._width = np.array([g.size for g in self._dg.groups], dtype=np.intp)
        self._start = np.array([g[0] if g.size else 0 for g in self._dg.groups], dtype=np.intp)
        self._regression = lru_cache(_DG_SETS)(self._fit_regression)

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the p-value for ``x`` independent of ``y`` given ``condition_set``.

        ``condition_set`` is canonicalized to sorted unique ints; ``x`` and ``y`` are ordered so the
        p-value is independent of argument order. A larger p-value means more independence.
        """
        return self._statistic_pvalue(x, y, condition_set)[1]

    def statistic_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The likelihood-ratio statistics and p-values of many queries, one per row.

        ``x`` and ``y`` are ``(N,)`` int arrays and ``conditions`` the ``(N, m)`` int array of
        conditioning sets; entry ``i`` holds query ``i``'s statistic and :meth:`__call__`'s p-value.
        """
        return per_query(self._statistic_pvalue, x, y, conditions)

    def _statistic_pvalue(
        self, x: int, y: int, condition_set: Iterable[int]
    ) -> tuple[float, float]:
        """The statistic and p-value of one query."""
        x_int, y_int, cond = _canonical(x, y, condition_set)
        groups = self._dg.groups
        both = np.concatenate([groups[x_int], groups[y_int]])
        block = self._corr[both[:, None], both]
        regression = self._regression(tuple(cond))
        if regression is not None:
            cs, basis, scale = regression
            cross = self._corr[both[:, None], cs] @ basis
            block = block - (cross / scale) @ cross.T
        return self._block_statistic_pvalue(block, groups[x_int].size)

    def batched_call(self, x: int, y: int, condition_sets: Iterable[Iterable[int]]) -> list[float]:
        """Return p-values for one ``(x, y)`` pair over many conditioning sets.

        The result list matches the order of ``condition_sets``, and each p-value equals
        :meth:`__call__`'s. A conditioning set is canonicalized as in :meth:`__call__`.
        """
        x_int, y_int = (int(x), int(y)) if x < y else (int(y), int(x))
        conds = [tuple(_canonical(x_int, y_int, s)[2]) for s in condition_sets]
        return self._pvalues([x_int] * len(conds), [y_int] * len(conds), conds).tolist()

    def separating_prefix(
        self, x: int, y: int, condition_sets: Sequence[Iterable[int]], alpha: float
    ) -> list[float]:
        """Return the p-values of ``condition_sets``, in order, up to and including the first above
        ``alpha``: the answers a loop of scalar calls that stops at the first separating set gets.

        The sets are evaluated together (:func:`_separating_prefix`); each p-value equals
        :meth:`__call__`'s.
        """
        return _separating_prefix(self, x, y, condition_sets, alpha)

    def _canonical_pvalues(
        self, xs: np.ndarray, ys: np.ndarray, conditions: np.ndarray
    ) -> np.ndarray:
        """P-values of the canonical queries ``(xs[i], ys[i], conditions[i])``, in order:
        ``xs[i] < ys[i]``, each row of ``conditions`` sorted and distinct."""
        conds = [tuple(c) for c in conditions.tolist()]
        return self._pvalues(xs.tolist(), ys.tolist(), conds)

    def _pvalues(self, xs: list[int], ys: list[int], conds: list[tuple[int, ...]]) -> np.ndarray:
        """P-values of canonical queries, stacked by shape (see :class:`DGTest`).

        When the stacked evaluation raises, each query takes :meth:`__call__` in order, so the error
        is the one the scalar calls raise.
        """
        try:
            return self._stacked_pvalues(xs, ys, conds)
        except Exception:
            return np.array([self(x, y, c) for x, y, c in zip(xs, ys, conds)], dtype=np.float64)

    def _stacked_pvalues(
        self, xs: list[int], ys: list[int], conds: list[tuple[int, ...]]
    ) -> np.ndarray:
        """:meth:`_pvalues`: each conditioning set regressed once, each shape stacked."""
        out = np.empty(len(xs))
        if not xs:
            return out
        sets: dict[tuple[int, ...], int] = {}
        zid = np.array([sets.setdefault(c, len(sets)) for c in conds], dtype=np.intp)
        regressions = [self._regression(c) for c in sets]
        columns = np.array([0 if r is None else r[0].size for r in regressions], dtype=np.intp)
        kept = np.array([0 if r is None else r[2].size for r in regressions], dtype=np.intp)
        x_arr, y_arr = np.array(xs, dtype=np.intp), np.array(ys, dtype=np.intp)
        px, py = self._width[x_arr], self._width[y_arr]
        keys = np.stack([px, py, columns[zid], kept[zid]], axis=1)
        shapes, group = np.unique(keys, axis=0, return_inverse=True)
        group = group.reshape(-1)
        for g, (gx, gy, gc, gk) in enumerate(shapes.tolist()):
            rows = np.flatnonzero(group == g)
            if not gx or not gy or (gc and not gk):
                # An empty variable, or a set with no direction kept: the scalar calls.
                out[rows] = [self(xs[i], ys[i], conds[i]) for i in rows.tolist()]
                continue
            step = max(1, _DG_STACK // (8 * (gx + gy) * (gx + gy + gc)))
            for at in range(0, rows.size, step):
                part = rows[at : at + step]
                out[part] = self._shape_pvalues(x_arr[part], y_arr[part], zid[part], regressions)
        return out

    def _shape_pvalues(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        zid: np.ndarray,
        regressions: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None],
    ) -> np.ndarray:
        """P-values of queries of one shape: the same numbers of columns of ``x``, ``y``, and the
        conditioning set, and of its directions kept. ``zid`` indexes each query's set in
        ``regressions``. Each stacked operand has the memory layout of its operand in
        :meth:`__call__` and :func:`_whitener`, so each query's BLAS and LAPACK calls are its
        own."""
        corr = self._corr
        px, py = int(self._width[xs[0]]), int(self._width[ys[0]])
        both = np.concatenate(
            (self._start[xs, None] + np.arange(px), self._start[ys, None] + np.arange(py)), axis=1
        )
        block = corr[both[:, :, None], both[:, None, :]]
        if regressions[zid[0]] is not None:
            sets, inverse = np.unique(zid, return_inverse=True)
            inverse = inverse.reshape(-1)
            parts = [cast("tuple[np.ndarray, ...]", regressions[z]) for z in sets.tolist()]
            cs = np.stack([p[0] for p in parts])[inverse]
            # Each basis column-major, as u[:, keep] is.
            basis = np.swapaxes(np.stack([p[1].T for p in parts])[inverse], 1, 2)
            scale = np.stack([p[2] for p in parts])[inverse]
            cross = corr[both[:, :, None], cs[:, None, :]] @ basis
            block = block - (cross / scale[:, None, :]) @ np.swapaxes(cross, 1, 2)
        values_x, vectors_x = np.linalg.eigh(block[:, :px, :px])
        values_y, vectors_y = np.linalg.eigh(block[:, px:, px:])
        out = np.empty(len(xs))
        full = (values_x > _RANK_TOL).all(axis=1) & (values_y > _RANK_TOL).all(axis=1)
        for i in np.flatnonzero(~full).tolist():
            out[i] = self._block_pvalue(block[i], px)
        if not full.any():
            return out
        # wx.T row-major and wy column-major, as _whitener's column-major whiteners give them.
        wx = vectors_x[full] / np.sqrt(values_x[full])[:, None, :]
        wy = vectors_y[full] / np.sqrt(values_y[full])[:, None, :]
        wx_t = np.ascontiguousarray(np.swapaxes(wx, 1, 2))
        wy = np.swapaxes(np.ascontiguousarray(np.swapaxes(wy, 1, 2)), 1, 2)
        rho = np.linalg.svd(wx_t @ block[full][:, :px, px:] @ wy, compute_uv=False)
        rho = np.minimum(rho, _ONE_MINUS_EPS)
        statistic = -self._dg.n * np.sum(np.log1p(-(rho * rho)), axis=1)
        out[full] = chdtrc(px * py, np.maximum(statistic, 0.0))
        return out

    def _fit_regression(
        self, cond: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """The embedded columns of the conditioning set ``cond``, the eigen-directions of their
        correlation matrix that are kept, and those directions' eigenvalues; ``None`` when the set
        has no columns. ``_regression`` keeps the results of the last ``_DG_SETS`` sets."""
        cs = self._dg.columns(cond)
        if not cs.size:
            return None
        # Regress on the conditioning columns through their nonzero eigen-directions, so collinear
        # conditioning columns need no inverse.
        w, u = np.linalg.eigh(self._corr[np.ix_(cs, cs)])
        keep = w > _RANK_TOL * w[-1]
        return cs, u[:, keep], w[keep]

    def _block_pvalue(self, block: np.ndarray, px: int) -> float:
        """The p-value from the correlation ``block`` of ``x``'s first ``px`` columns and ``y``'s,
        after the regression on the conditioning set."""
        return self._block_statistic_pvalue(block, px)[1]

    def _block_statistic_pvalue(self, block: np.ndarray, px: int) -> tuple[float, float]:
        """The statistic and p-value of :meth:`_block_pvalue`."""
        wx = _whitener(block[:px, :px])
        wy = _whitener(block[px:, px:])
        dof = wx.shape[1] * wy.shape[1]
        if dof == 0:
            return 0.0, 1.0
        rho = np.linalg.svd(wx.T @ block[:px, px:] @ wy, compute_uv=False)
        rho = np.minimum(rho, _ONE_MINUS_EPS)
        statistic = -self._dg.n * float(np.sum(np.log1p(-(rho * rho))))
        return statistic, float(chdtrc(dof, max(statistic, 0.0)))


def _whitener(cov: np.ndarray) -> np.ndarray:
    """Map ``cov``'s columns onto unit-variance, uncorrelated ones, dropping null directions."""
    w, u = np.linalg.eigh(cov)
    keep = w > _RANK_TOL
    return u[:, keep] / np.sqrt(w[keep])


# The most pairs one marginal pass of the conditional Gaussian test fits together.
_CG_PAIRS = 4096


class CGTest:
    """Conditional Gaussian likelihood-ratio test of conditional independence on mixed data.

    The variables are fit as :class:`~andrey.core.mixed.ConditionalGaussian` describes: the rows
    split into the cells of the categorical variables' joint categories, the continuous variables
    Gaussian within each cell, with a covariance shrunk toward that of all rows [1]_. With ``l(A)``
    the log-likelihood of the set ``A`` and ``k(A)`` its parameter count, the test of ``x`` and
    ``y`` given ``S`` compares the model of ``x`` given ``y`` and ``S`` with the model of ``x``
    given ``S`` [1]_:

    ``2 * (l(x, y, S) - l(y, S) - l(x, S) + l(S))``

    on ``k(x, y, S) - k(y, S) - k(x, S) + k(S)`` degrees of freedom, a chi-square survival function
    p-value. The expression is symmetric in ``x`` and ``y``. On categorical columns only, the
    statistic is the G-square statistic of :class:`ChiSquare`, with ``(r_x - 1) * (r_y - 1)``
    degrees of freedom for every joint category of ``S``, whether it occurs or not; on continuous
    columns only, it is ``-n log(1 - r**2)`` with ``r`` the partial correlation, on one degree of
    freedom.

    Log-likelihoods are memoized by set, and queries are canonicalized as :class:`FisherZ`'s are.
    :meth:`batched_call`, :meth:`first_order_pvalues`, and :meth:`marginal_pvalues` fit the sets of
    many queries in one :meth:`~andrey.core.mixed.ConditionalGaussian.loglik_many` call and return
    :meth:`__call__`'s p-values exactly.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Finite values; categorical columns hold their categories as numbers, each distinct value of
        a column one category.
    categorical : array-like of bool, shape (n_features,), optional
        Marks the categorical columns; ``None`` marks none.

    References
    ----------
    .. [1] Andrews, B., Ramsey, J., and Cooper, G. F. (2018). Scoring Bayesian networks of mixed
       variables. International Journal of Data Science and Analytics 6(1), 3-18.
    """

    def __init__(self, data: np.ndarray, categorical: npt.ArrayLike | None = None) -> None:
        self._cg = ConditionalGaussian(data, categorical)
        self._d: int = np.shape(data)[1]

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the p-value for ``x`` independent of ``y`` given ``condition_set``.

        ``condition_set`` is canonicalized to sorted unique ints; ``x`` and ``y`` are ordered so the
        p-value is independent of argument order. A larger p-value means more independence.
        """
        return self._statistic_pvalue(x, y, condition_set)[1]

    def statistic_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The likelihood-ratio statistics and p-values of many queries, one per row.

        ``x`` and ``y`` are ``(N,)`` int arrays and ``conditions`` the ``(N, m)`` int array of
        conditioning sets; entry ``i`` holds query ``i``'s statistic and :meth:`__call__`'s p-value.
        """
        return per_query(self._statistic_pvalue, x, y, conditions)

    def _statistic_pvalue(
        self, x: int, y: int, condition_set: Iterable[int]
    ) -> tuple[float, float]:
        """The statistic and p-value of one query."""
        x_int, y_int, cond = _canonical(x, y, condition_set)
        cg = self._cg
        sets = ([x_int, y_int, *cond], [y_int, *cond], [x_int, *cond], cond)
        ll = [cg.loglik(s) for s in sets]
        k = [cg.dof(s) for s in sets]
        statistic = 2.0 * (ll[0] - ll[1] - ll[2] + ll[3])
        dof = k[0] - k[1] - k[2] + k[3]
        return statistic, float(chdtrc(dof, max(statistic, 0.0)))

    def batched_call(self, x: int, y: int, condition_sets: Iterable[Iterable[int]]) -> list[float]:
        """Return p-values for one ``(x, y)`` pair over many conditioning sets.

        The result list matches the order of ``condition_sets``, and each p-value equals
        :meth:`__call__`'s: the four sets of every query are fit in one
        :meth:`~andrey.core.mixed.ConditionalGaussian.loglik_many` call, which gives each set the
        value it gets alone.
        """
        x_int, y_int = (int(x), int(y)) if x < y else (int(y), int(x))
        queries = [(x_int, y_int, _canonical(x_int, y_int, s)[2]) for s in condition_sets]
        return self._pvalues(queries).tolist()

    def separating_prefix(
        self, x: int, y: int, condition_sets: Sequence[Iterable[int]], alpha: float
    ) -> list[float]:
        """Return the p-values of ``condition_sets``, in order, up to and including the first above
        ``alpha``: the answers a loop of scalar calls that stops at the first separating set gets.

        The sets are evaluated together (:func:`_separating_prefix`); each p-value equals
        :meth:`__call__`'s.
        """
        return _separating_prefix(self, x, y, condition_sets, alpha)

    def first_order_pvalues(self, x: int, y: int, zs: Iterable[int]) -> np.ndarray:
        """First-order (``|S| == 1``) p-values for pair ``(x, y)`` over single conditioners ``zs``.

        Entry ``i`` equals ``self(x, y, (zs[i],))``; ``zs`` must exclude ``x`` and ``y``.
        """
        return np.asarray(self.batched_call(x, y, [(int(z),) for z in zs]), dtype=np.float64)

    def marginal_pvalues(self) -> np.ndarray:
        """All-pairs marginal (``|S| == 0``) p-values as a symmetric ``(d, d)`` matrix.

        Entry ``[x, y]`` equals ``self(x, y, ())``; the diagonal is ``0``. The pairs are fit
        together, a few thousand per :meth:`~andrey.core.mixed.ConditionalGaussian.loglik_many`
        call.
        """
        d = self._d
        out = np.zeros((d, d))
        xs, ys = np.triu_indices(d, 1)
        for at in range(0, xs.size, _CG_PAIRS):
            part = slice(at, at + _CG_PAIRS)
            queries = [(x, y, []) for x, y in zip(xs[part].tolist(), ys[part].tolist())]
            out[xs[part], ys[part]] = self._pvalues(queries)
        return out + out.T

    def _pvalues(self, queries: list[tuple[int, int, list[int]]]) -> np.ndarray:
        """P-values of canonical ``(x, y, conditioning set)`` queries, their sets fit together."""
        cg = self._cg
        sets = [s for x, y, c in queries for s in ([x, y, *c], [y, *c], [x, *c], c)]
        ll = cg.loglik_many(sets)
        k = [cg.dof(s) for s in sets]
        quads = range(0, len(sets), 4)
        statistic = [2.0 * (ll[i] - ll[i + 1] - ll[i + 2] + ll[i + 3]) for i in quads]
        dof = [k[i] - k[i + 1] - k[i + 2] + k[i + 3] for i in quads]
        return np.asarray(chdtrc(dof, np.maximum(statistic, 0.0)), dtype=np.float64).reshape(-1)


# The module and class of each nonlinear test. Building one imports only its own module: GPDC's
# loads scikit-learn, which takes a second or more.
_NONLINEAR_CLASSES: dict[str, tuple[str, str]] = {
    "kci": ("ci_kernel", "KCI"),
    "rcot": ("ci_kernel", "RCoT"),
    "cmiknn": ("ci_cmiknn", "CMIknn"),
    "gpdc": ("ci_gpdc", "GPDC"),
}
NONLINEAR_INDEP_TESTS: tuple[str, ...] = tuple(_NONLINEAR_CLASSES)
#: The tests that draw random numbers, which ``seed`` fixes: RCoT's random features, and the random
#: tie keys and permutations of CMIknn and GPDC. The other tests draw none.
STOCHASTIC_INDEP_TESTS: tuple[str, ...] = ("rcot", "cmiknn", "gpdc")


def _nonlinear(name: str, data: np.ndarray, **options: Any) -> CITest:
    """Build the nonlinear test ``name`` over ``data`` with ``options``."""
    module, cls = _NONLINEAR_CLASSES[name]
    test = getattr(importlib.import_module(f"{__package__}.{module}"), cls)
    return test(data, **options)


# The indep_test registry: a name maps to a factory building a :class:`CITest` over the data. The
# constraint facades resolve their ``indep_test`` argument through :func:`make_indep_test`, and
# :func:`register_indep_test` lets a downstream test or plugin add another under the same Protocol.
# The discrete tests read category codes and the mixed tests a mask of the categorical columns as
# well; the facades check both before a fit.
SHIPPED_INDEP_TESTS: tuple[str, ...] = (
    "fisherz",
    "student_t",
    "chisq",
    "gsq",
    "degen_gauss",
    "cond_gauss",
    *NONLINEAR_INDEP_TESTS,
)
DISCRETE_INDEP_TESTS: tuple[str, ...] = ("chisq", "gsq")
MIXED_INDEP_TESTS: tuple[str, ...] = ("degen_gauss", "cond_gauss")
_CI_REGISTRY: dict[str, Callable[..., CITest]] = {
    "fisherz": FisherZ,
    "student_t": StudentT,
    "chisq": ChiSquare,
    "gsq": partial(ChiSquare, g_square=True),
    "degen_gauss": DGTest,
    "cond_gauss": CGTest,
    **{name: partial(_nonlinear, name) for name in NONLINEAR_INDEP_TESTS},
}


def register_indep_test(name: str, factory: Callable[[np.ndarray], CITest]) -> None:
    """Register a :class:`CITest` factory under ``name`` for selection via ``indep_test``.

    ``factory`` maps a ``(n_samples, n_features)`` data matrix to a ``CITest`` (a callable
    ``test(x, y, condition_set) -> p-value``), and takes the keyword settings
    :func:`make_indep_test` passes, if any. Overwrites any existing entry for ``name``. A variant
    of a shipped test with other settings is ``partial(make_indep_test, "cmiknn", k_cmi=5)``.
    """
    _CI_REGISTRY[name] = factory


def available_indep_tests() -> tuple[str, ...]:
    """The ``indep_test`` names currently resolvable, sorted (shipped plus any registered)."""
    return tuple(sorted(_CI_REGISTRY))


def make_indep_test(
    name: str,
    data: np.ndarray,
    *,
    categorical: npt.ArrayLike | None = None,
    seed: int | None = None,
    **options: Any,
) -> CITest:
    """Construct the registered ``indep_test`` ``name`` over ``data``.

    ``options`` are keyword settings for the test's class, such as
    ``make_indep_test("cmiknn", data, k_cmi=5)``; a setting the test lacks raises ``TypeError``.
    ``seed`` reaches the tests that draw random numbers (:data:`STOCHASTIC_INDEP_TESTS`), which
    otherwise use ``ANDREY_SEED``; the other tests ignore it.
    ``categorical``, a boolean mask of the categorical columns, reaches the mixed tests
    (:data:`MIXED_INDEP_TESTS`) only; another test given one raises ``ValueError``. Raises
    :class:`NotImplementedError` when ``name`` is not registered, listing what is available. A
    :class:`PartialCorrelationTest` raises :class:`~andrey.core.score.CopiedColumnsError`, naming
    every group of :meth:`~PartialCorrelationTest.copies`, when ``data`` holds columns that repeat
    others up to scale and shift: its tests cannot tell such columns apart.

    ``data`` holding ``NaN`` gives a :class:`~andrey.core.ci_missing.TestwiseDeletion` of the test
    instead: each query reads only the rows observed in all of its columns, and raises where a
    complete-data test raises before any query, such as on copies among its columns.
    """
    try:
        factory = _CI_REGISTRY[name]
    except KeyError:
        raise NotImplementedError(
            f"unknown indep_test {name!r}; registered: {available_indep_tests()}"
        ) from None
    if seed is not None and name in STOCHASTIC_INDEP_TESTS:
        options["seed"] = seed
    mixed = name in MIXED_INDEP_TESTS
    if not mixed and categorical is not None:
        raise ValueError(f"indep_test {name!r} reads no column types; categorical must be None")
    if _has_nan(data):
        from .ci_missing import testwise_deletion

        return testwise_deletion(name, factory, data, categorical, options, mixed=mixed)
    if mixed:
        return factory(data, categorical, **options)
    test = factory(data, **options)
    if isinstance(test, PartialCorrelationTest) and test.copies():
        raise copies_error(test.copies())
    return test


def _has_nan(data: np.ndarray) -> bool:
    """Whether ``data``, a float array, holds a ``NaN``; any other array holds none."""
    arr = np.asarray(data)
    return arr.dtype.kind == "f" and bool(np.isnan(arr).any())


def _checked_data(data: np.ndarray, test_name: str, min_rows: int = 0) -> np.ndarray:
    """``data`` as a float64 ``(n_samples, n_features)`` matrix; raises on NaN or infinity, or on
    fewer than ``min_rows`` rows."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"data must be a 2-D (n_samples, n_features) array, got ndim={arr.ndim}")
    if arr.shape[0] < min_rows:
        raise ValueError(f"{test_name} needs at least {min_rows} rows, got {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"data contains NaN or Inf; cannot run {test_name} test")
    return arr


def pins_blas(test: CITest) -> bool:
    """Whether a skeleton search runs ``test`` with BLAS pinned to one thread.

    A test that sets ``pin_blas`` decides for itself. Otherwise a test whose p-values come from
    matrix algebra, the partial-correlation and the mixed-data tests, is pinned: it then gives the
    same bits whatever the host's thread count, and the partial-correlation tests match the
    one-thread parallel skeleton workers; the chi-square tests call no BLAS. The nonlinear tests set
    it: KCI is unpinned from 2,000 rows, where its kernel products run faster on every BLAS thread
    and its p-values are reproducible for a fixed thread count; RCoT, CMIknn, and GPDC are pinned.
    """
    pin = getattr(test, "pin_blas", None)
    if pin is not None:
        return bool(pin)
    return callable(getattr(test, "to_seed", None)) or isinstance(test, (DGTest, CGTest))


def _invert_nonsingular(stack: np.ndarray) -> np.ndarray:
    """Invert a ``(N, k, k)`` stack of correlation submatrices; raise if any is singular.

    A submatrix is singular when its smallest eigenvalue is at most ``_SINGULAR_RTOL`` times its
    largest; :class:`ValueError` reports it. ``np.linalg.inv`` alone does not decide this: it raises
    only on an exactly zero pivot, which OpenBLAS reaches on a duplicated column while Apple
    Accelerate can return a finite inverse instead. The eigenvalues decide on every backend.

    Most rows are settled from the inverse without eigenvalues. Each submatrix ``A`` is a principal
    submatrix of a sample correlation matrix: unit diagonal, positive semidefinite up to rounding.
    So ``lambda_max(A) <= trace(A) <= k``. The eigenvalue of smallest magnitude is
    ``1 / ||A^-1||_2 >= 1 / (k * max |A^-1_ij|)``, and when that is far above the rounding it is
    positive, so it is ``lambda_min(A)``. With ``c = _BOUND_MARGIN * _SINGULAR_RTOL``, a row whose
    inverse has every entry below ``1 / (c * k**2)`` thus has ``lambda_min > c * lambda_max`` and
    passes the eigenvalue test. The bound reads every entry of the computed inverse, not only its
    trace or diagonal, so a NaN, an infinity, or a single huge entry fails it. The LU inverse is
    backward stable on every LAPACK, so the computed inverse of a nearly singular matrix has
    entries of order ``1 / lambda_min`` and fails the bound. ``_BOUND_MARGIN`` absorbs the rounding
    of the inverse and the eigenvalues.

    The rows that fail the bound take the eigenvalue test. When ``np.linalg.inv`` raises, the
    whole stack takes it, and the inversion raises again if no row is singular.
    """
    try:
        inv = np.linalg.inv(stack)
    except np.linalg.LinAlgError:
        inv = None
    if inv is None:
        _raise_if_singular(stack)
        return np.linalg.inv(stack)
    size = stack.shape[-1]
    limit = 1.0 / (_BOUND_MARGIN * _SINGULAR_RTOL * size * size)
    unsettled = ~(np.abs(inv).max(axis=(1, 2)) < limit)  # NaN fails the comparison too
    if unsettled.any():
        _raise_if_singular(stack[unsettled])
    return inv


def _raise_if_singular(stack: np.ndarray) -> None:
    """Raise :class:`ValueError` if a matrix of the ``(N, k, k)`` stack is singular.

    A matrix is singular when its smallest eigenvalue is at most ``_SINGULAR_RTOL`` times its
    largest.
    """
    w = np.linalg.eigvalsh(stack)  # (N, k), ascending per row
    if (w[:, 0] <= w[:, -1] * _SINGULAR_RTOL).any():
        raise SingularCorrelationError(
            "correlation matrix is singular; cannot run a partial-correlation test"
        )


def _separating_prefix(
    test: Any, x: int, y: int, condition_sets: Sequence[Iterable[int]], alpha: float
) -> list[float]:
    """``test``'s p-values of ``condition_sets`` up to and including the first above ``alpha``.

    One ``batched_call`` evaluates every set; the sets after the first separating one are computed
    and dropped. When that call raises, the sets are tested one scalar call at a time, stopping at
    the first separating one, so an error surfaces only from a set the scalar loop reaches.
    """
    try:
        pvalues = list(test.batched_call(x, y, condition_sets))
    except Exception:
        pvalues = None
    if pvalues is not None:
        for i, p in enumerate(pvalues):
            if p > alpha:
                return pvalues[: i + 1]
        return pvalues
    out: list[float] = []
    for condition_set in condition_sets:
        out.append(test(x, y, condition_set))
        if out[-1] > alpha:
            break
    return out


def _canonical(x: int, y: int, condition_set: Iterable[int]) -> tuple[int, int, list[int]]:
    """Canonicalize a query to ordered ints and a sorted-unique conditioning set."""
    cond = sorted(set(int(c) for c in condition_set))
    x_int, y_int = (int(x), int(y)) if x < y else (int(y), int(x))
    assert x_int not in cond and y_int not in cond, "x, y cannot be in condition_set."
    return x_int, y_int, cond


def _canonical_ranked(
    x: int, y: int, condition_set: Iterable[int], rank: Sequence[int]
) -> tuple[int, int, list[int]]:
    """:func:`_canonical` of a query to a test that holds data column ``j`` at index ``rank[j]``."""
    return _canonical(rank[x], rank[y], [rank[c] for c in condition_set])


def per_query(
    answer: Callable[[int, int, list[int]], tuple[float, float]],
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    conditions: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """The statistics and p-values of many queries, ``answer(x, y, condition_set)`` one at a time.

    ``x`` and ``y`` are ``(N,)`` int arrays and ``conditions`` the ``(N, m)`` int array of
    conditioning sets.
    """
    xs = np.asarray(x, dtype=np.intp)
    ys = np.asarray(y, dtype=np.intp)
    conds = np.asarray(conditions, dtype=np.intp).reshape(len(xs), -1)
    statistics = np.empty(len(xs), dtype=np.float64)
    p_values = np.empty(len(xs), dtype=np.float64)
    for k in range(len(xs)):
        statistics[k], p_values[k] = answer(int(xs[k]), int(ys[k]), conds[k].tolist())
    return statistics, p_values
