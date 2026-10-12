"""Conditional-independence test substrate: the partial-correlation Fisher-Z and Student-t tests.

The constraint-based discovery family (PC / FCI / CDNOD) consumes a conditional-independence test
through the :class:`CITest` contract: an object constructed once over the data and called as
``test(x, y, condition_set)`` returning a p-value, where a larger p-value means more independent.

:class:`PartialCorrelationTest` is the base of the two shipped tests. It precomputes the sample
correlation matrix once, then answers each query from the conditioned partial correlation of
columns ``x`` and ``y`` given ``condition_set``. Empty and single-element conditioning sets -- the
bulk of a constraint-based search -- use the closed-form (first-order) partial correlation and skip
the matrix inversion entirely; larger sets invert the correlation submatrix. numpy float64 is the
oracle. A query is canonicalized (``x``, ``y`` ordered, the conditioning set sorted unique) so its
p-value does not depend on argument order, and nothing is memoized: a test object holds the
correlation matrix and the sample size, whatever it is asked. The scalar path, the batched path
(:meth:`~PartialCorrelationTest.batched_call`), the all-pairs and first-order closed forms, and the
many-query path (:meth:`~PartialCorrelationTest.statistic_pvalues`) return the same p-values
exactly.

The two tests differ only in how a partial correlation ``r`` and its degrees of freedom become a
p-value. :class:`FisherZ` applies the normal distribution to Fisher's z-transform of ``r``;
:class:`StudentT` applies the Student-t distribution to ``t = r * sqrt(df / (1 - r**2))``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Protocol, Self, cast, runtime_checkable

import numpy as np
from scipy.special import ndtr, stdtr

from . import stats

# Partial correlations are clamped just inside (-1, 1) so Fisher's z-transform stays finite on
# deterministic or tiny-sample data.
_ONE_MINUS_EPS = 1.0 - np.finfo(np.float64).eps


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
            sub = corr[np.ix_(var, var)]
            # A singular submatrix must raise, but np.linalg.inv only raises under some BLAS
            # backends (OpenBLAS yes; Apple Accelerate no -- it returns a garbage inverse, and its
            # LU-solve residual is still small, so checking sub @ inv is not enough). The symmetric
            # spectrum exposes the rank deficiency (a zero eigenvalue) reliably on every backend.
            w = np.linalg.eigvalsh(sub)
            if w[0] <= w[-1] * 1e-12:
                raise ValueError(
                    "correlation matrix is singular; cannot run a partial-correlation test"
                )
            inv = np.linalg.inv(sub)
            r = -float(inv[0, 1]) / math.sqrt(abs(float(inv[0, 0]) * float(inv[1, 1])))
        return (
            -_ONE_MINUS_EPS if r < -_ONE_MINUS_EPS else _ONE_MINUS_EPS if r > _ONE_MINUS_EPS else r
        )

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
            stack = corr[var[:, :, None], var[:, None, :]]
            # Detect rank-deficiency via the symmetric spectrum (a zero eigenvalue), reliable on
            # every BLAS backend -- unlike np.linalg.inv, which raises on OpenBLAS but not Apple
            # Accelerate.
            w = np.linalg.eigvalsh(stack)  # (N, dim), ascending per row
            if (w[:, 0] <= w[:, -1] * 1e-12).any():
                raise ValueError(
                    "correlation matrix is singular; cannot run a partial-correlation test"
                )
            inv = np.linalg.inv(stack)
            r = -inv[:, 0, 1] / np.sqrt(np.abs(inv[:, 0, 0] * inv[:, 1, 1]))
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


# The indep_test registry: a name maps to a factory building a :class:`CITest` over the data.
# Fisher-Z and Student-t are the shipped tests; the constraint facades resolve their ``indep_test``
# argument through :func:`make_indep_test`, and :func:`register_indep_test` lets a downstream test
# or plugin add another under the same Protocol -- the generality this registry exposes.
SHIPPED_INDEP_TESTS: tuple[str, ...] = ("fisherz", "student_t")
_CI_REGISTRY: dict[str, Callable[[np.ndarray], CITest]] = {
    "fisherz": FisherZ,
    "student_t": StudentT,
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


def make_indep_test(name: str, data: np.ndarray) -> CITest:
    """Construct the registered ``indep_test`` ``name`` over ``data``.

    Raises :class:`NotImplementedError` when ``name`` is not registered, listing what is available.
    """
    try:
        factory = _CI_REGISTRY[name]
    except KeyError:
        raise NotImplementedError(
            f"unknown indep_test {name!r}; registered: {available_indep_tests()}"
        ) from None
    return factory(data)


def _canonical(x: int, y: int, condition_set: Iterable[int]) -> tuple[int, int, list[int]]:
    """Canonicalize a query to ordered ints and a sorted-unique conditioning set."""
    cond = sorted(set(int(c) for c in condition_set))
    x_int, y_int = (int(x), int(y)) if x < y else (int(y), int(x))
    assert x_int not in cond and y_int not in cond, "x, y cannot be in condition_set."
    return x_int, y_int, cond
