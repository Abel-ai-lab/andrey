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
and first-order closed forms, and the many-query path
(:meth:`~PartialCorrelationTest.statistic_pvalues`) return the same p-values exactly.

The two partial-correlation tests differ only in how a partial correlation ``r`` and its degrees of
freedom become a p-value. :class:`FisherZ` applies the normal distribution to Fisher's z-transform
of ``r``; :class:`StudentT` applies the Student-t distribution to ``t = r * sqrt(df / (1 - r**2))``.

:class:`ChiSquare` implements the Pearson chi-square and G-square tests on category codes: one
contingency table of ``x`` and ``y`` per observed configuration of the conditioning set.

:class:`DGTest` and :class:`CGTest` test mixed data, continuous and categorical columns marked by a
mask, by the likelihood ratio of the degenerate Gaussian and the conditional Gaussian models of
:mod:`andrey.core.mixed`.

The registry at the end maps each ``indep_test`` name to a factory. The experimental nonlinear
tests live in their own modules (KCI and RCoT in :mod:`andrey.core.ci_kernel`, CMIknn in
:mod:`andrey.core.ci_cmiknn`, GPDC in :mod:`andrey.core.ci_gpdc`) and are imported when first
built.

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

import math
from collections.abc import Callable, Iterable
from functools import partial
from typing import Protocol, Self, cast, runtime_checkable

import numpy as np
import numpy.typing as npt
from scipy.special import chdtrc, ndtr, stdtr

from . import categorical, stats
from .mixed import ConditionalGaussian, DegenerateGaussian
from .score import copies_message, copy_groups
from .warning_policy import ExperimentalWarning, warn_once

__all__ = ["CITest", "FisherZ", "register_indep_test", "available_indep_tests", "make_indep_test"]

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

    The scalar path, :meth:`batched_call`, :meth:`statistic_pvalues`, :meth:`marginal_pvalues`, and
    :meth:`first_order_pvalues` use identical partial-correlation, clip, and p-value arithmetic.
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

    def _invert(self, stack: np.ndarray) -> np.ndarray:
        """:func:`_invert_nonsingular` of ``stack``; a singular error names the data's copies.

        The groups of copies come from the full correlation matrix, found once per test.
        """
        try:
            return _invert_nonsingular(stack)
        except SingularCorrelationError as error:
            if not hasattr(self, "_copies"):
                self._copies = copy_groups(self._corr, self._n)
            if not self._copies:
                raise
            named = SingularCorrelationError(
                f"correlation matrix is singular because {copies_message(self._copies)}"
            )
            named.groups = self._copies
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
        self._g_square = bool(g_square)

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the p-value for ``x`` independent of ``y`` given ``condition_set``.

        ``condition_set`` is canonicalized to sorted unique ints; ``x`` and ``y`` are ordered so the
        p-value is independent of argument order. A larger p-value means more independence.
        """
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
            return 1.0
        return float(chdtrc(dof, max(statistic, 0.0)))


# A conditional covariance's eigenvalue, on the scale of unit-variance columns, at or below this
# marks a direction the conditioning set determines.
_RANK_TOL = 1e-10


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

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the p-value for ``x`` independent of ``y`` given ``condition_set``.

        ``condition_set`` is canonicalized to sorted unique ints; ``x`` and ``y`` are ordered so the
        p-value is independent of argument order. A larger p-value means more independence.
        """
        x_int, y_int, cond = _canonical(x, y, condition_set)
        dg = self._dg
        cx, cy, cs = dg.columns([x_int]), dg.columns([y_int]), dg.columns(cond)
        both = np.concatenate([cx, cy])
        block = self._corr[np.ix_(both, both)]
        if cs.size:
            # Regress on the conditioning columns through their nonzero eigen-directions, so
            # collinear conditioning columns need no inverse.
            w, u = np.linalg.eigh(self._corr[np.ix_(cs, cs)])
            keep = w > _RANK_TOL * w[-1]
            cross = self._corr[np.ix_(both, cs)] @ u[:, keep]
            block = block - (cross / w[keep]) @ cross.T
        px = cx.size
        wx = _whitener(block[:px, :px])
        wy = _whitener(block[px:, px:])
        dof = wx.shape[1] * wy.shape[1]
        if dof == 0:
            return 1.0
        rho = np.linalg.svd(wx.T @ block[:px, px:] @ wy, compute_uv=False)
        rho = np.minimum(rho, _ONE_MINUS_EPS)
        statistic = -dg.n * float(np.sum(np.log1p(-(rho * rho))))
        return float(chdtrc(dof, max(statistic, 0.0)))


def _whitener(cov: np.ndarray) -> np.ndarray:
    """Map ``cov``'s columns onto unit-variance, uncorrelated ones, dropping null directions."""
    w, u = np.linalg.eigh(cov)
    keep = w > _RANK_TOL
    return u[:, keep] / np.sqrt(w[keep])


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

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the p-value for ``x`` independent of ``y`` given ``condition_set``.

        ``condition_set`` is canonicalized to sorted unique ints; ``x`` and ``y`` are ordered so the
        p-value is independent of argument order. A larger p-value means more independence.
        """
        x_int, y_int, cond = _canonical(x, y, condition_set)
        cg = self._cg
        sets = ([x_int, y_int, *cond], [y_int, *cond], [x_int, *cond], cond)
        ll = [cg.loglik(s) for s in sets]
        k = [cg.dof(s) for s in sets]
        statistic = 2.0 * (ll[0] - ll[1] - ll[2] + ll[3])
        dof = k[0] - k[1] - k[2] + k[3]
        return float(chdtrc(dof, max(statistic, 0.0)))


NONLINEAR_INDEP_TESTS: tuple[str, ...] = ("kci", "rcot", "cmiknn", "gpdc")


def _nonlinear(name: str, data: np.ndarray) -> CITest:
    """Build the experimental nonlinear test ``name``, warning once per process."""
    from .ci_cmiknn import CMIknn
    from .ci_gpdc import GPDC
    from .ci_kernel import KCI, RCoT

    test = {"kci": KCI, "rcot": RCoT, "cmiknn": CMIknn, "gpdc": GPDC}[name]
    warn_once(
        f"indep_test={name!r} is experimental: it has no published benchmark, and its settings "
        "may change without deprecation.",
        ExperimentalWarning,
    )
    return test(data)


# The indep_test registry: a name maps to a factory building a :class:`CITest` over the data. The
# constraint facades resolve their ``indep_test`` argument through :func:`make_indep_test`, and
# :func:`register_indep_test` lets a downstream test or plugin add another under the same Protocol.
# The discrete tests read category codes and the mixed tests a mask of the categorical columns as
# well; the facades check both before a fit. The nonlinear tests are experimental and warn when
# built.
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
    "gsq": lambda data: ChiSquare(data, g_square=True),
    "degen_gauss": DGTest,
    "cond_gauss": CGTest,
    **{name: partial(_nonlinear, name) for name in NONLINEAR_INDEP_TESTS},
}


def register_indep_test(name: str, factory: Callable[[np.ndarray], CITest]) -> None:
    """Register a :class:`CITest` factory under ``name`` for selection via ``indep_test``.

    ``factory`` maps a ``(n_samples, n_features)`` data matrix to a ``CITest`` (a callable
    ``test(x, y, condition_set) -> p-value``). Overwrites any existing entry for ``name``.
    """
    _CI_REGISTRY[name] = factory


def available_indep_tests() -> tuple[str, ...]:
    """The ``indep_test`` names currently resolvable, sorted (shipped plus any registered)."""
    return tuple(sorted(_CI_REGISTRY))


def make_indep_test(
    name: str, data: np.ndarray, *, categorical: npt.ArrayLike | None = None
) -> CITest:
    """Construct the registered ``indep_test`` ``name`` over ``data``.

    ``categorical``, a boolean mask of the categorical columns, reaches the mixed tests
    (:data:`MIXED_INDEP_TESTS`) only; another test given one raises ``ValueError``. Raises
    :class:`NotImplementedError` when ``name`` is not registered, listing what is available.
    """
    try:
        factory = _CI_REGISTRY[name]
    except KeyError:
        raise NotImplementedError(
            f"unknown indep_test {name!r}; registered: {available_indep_tests()}"
        ) from None
    if name in MIXED_INDEP_TESTS:
        return factory(data, categorical)
    if categorical is not None:
        raise ValueError(f"indep_test {name!r} reads no column types; categorical must be None")
    return factory(data)


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


def _canonical(x: int, y: int, condition_set: Iterable[int]) -> tuple[int, int, list[int]]:
    """Canonicalize a query to ordered ints and a sorted-unique conditioning set."""
    cond = sorted(set(int(c) for c in condition_set))
    x_int, y_int = (int(x), int(y)) if x < y else (int(y), int(x))
    assert x_int not in cond and y_int not in cond, "x, y cannot be in condition_set."
    return x_int, y_int, cond
