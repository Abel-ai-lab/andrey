"""Conditional-independence test substrate: the partial-correlation Fisher-Z test.

The constraint-based discovery family (PC / FCI / CDNOD) consumes a conditional-independence test
through the :class:`CITest` contract: an object constructed once over the data and called as
``test(x, y, condition_set)`` returning a p-value, where a larger p-value means more independent.

:class:`FisherZ` implements the Gaussian partial-correlation test (Fisher's z-transform). It
precomputes the sample correlation matrix once, then answers each query from the conditioned
partial correlation of columns ``x`` and ``y`` given ``condition_set``. Empty and single-element
conditioning sets -- the bulk of a constraint-based search -- use the closed-form (first-order)
partial correlation and skip the matrix inversion entirely; larger sets invert the correlation
submatrix. numpy float64 is the oracle. A query is canonicalized (``x``, ``y`` ordered, the
conditioning set sorted unique) so its p-value does not depend on argument order, and nothing is
memoized: a test object holds the correlation matrix and the sample size, whatever it is asked.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Protocol, cast, runtime_checkable

import numpy as np
from scipy.special import ndtr

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


class FisherZ:
    """Fisher-Z partial-correlation conditional-independence test.

    ``__init__`` records the sample size and precomputes the full column correlation matrix once.
    Each call forms the partial correlation ``r`` of ``x`` and ``y`` given the conditioning set --
    directly (``|S| == 0``), by the first-order closed form (``|S| == 1``), or by inverting the
    ``[x, y, *S]`` correlation submatrix (``|S| >= 2``) -- applies Fisher's z-transform, and reports
    ``p = 2 * (1 - Phi(sqrt(n - |S| - 3) * |z|))`` with ``Phi`` the standard-normal CDF. The
    ``|S| <= 1`` closed forms do not invert a submatrix, so on exactly-collinear columns they return
    a finite (near-1) partial correlation, clamped just inside ``(-1, 1)``, where an always-invert
    path would instead raise on a singular matrix.

    Queries are canonicalized (``x``, ``y`` ordered, ``condition_set`` sorted unique), so the
    p-value is independent of argument order. No query is memoized: the skeleton tests each
    conditioning set of a pair once, so retained results would grow with the number of subsets
    visited. Later queries are recomputed.

    The scalar path and :meth:`batched_call` use numpy's log with identical partial-correlation
    and clip arithmetic. Their p-values match exactly, so repeated queries take the same
    ``p > alpha`` decision across call paths, including at the threshold.
    """

    def __init__(self, data: np.ndarray) -> None:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f"data must be a 2-D (n_samples, n_features) array, got ndim={arr.ndim}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("data contains NaN or Inf; cannot run Fisher-Z test")
        self._n: int = arr.shape[0]
        self._corr: np.ndarray = np.asarray(stats.corrcoef(arr, rowvar=False), dtype=np.float64)

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the Fisher-Z p-value for ``x`` independent of ``y`` given ``condition_set``.

        ``condition_set`` is canonicalized to sorted unique ints; ``x`` and ``y`` are ordered so the
        p-value is independent of argument order. A larger p-value means more independence.
        """
        x_int, y_int, cond = _canonical(x, y, condition_set)
        m = len(cond)
        df = self._n - m - 3
        if df < 0:
            raise ValueError(
                f"Fisher-Z degrees of freedom is negative (n={self._n}, |S|={m}); need n > |S| + 3"
            )
        corr = self._corr
        if m == 0:
            r = float(corr[x_int, y_int])  # marginal correlation; no inversion
        elif m == 1:  # first-order partial correlation, closed form
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
                raise ValueError("correlation matrix is singular; cannot run Fisher-Z test")
            inv = np.linalg.inv(sub)
            r = -float(inv[0, 1]) / math.sqrt(abs(float(inv[0, 0]) * float(inv[1, 1])))
        r = -_ONE_MINUS_EPS if r < -_ONE_MINUS_EPS else _ONE_MINUS_EPS if r > _ONE_MINUS_EPS else r
        z = 0.5 * np.log((1.0 + r) / (1.0 - r))
        return 2.0 * (1.0 - float(ndtr(math.sqrt(df) * abs(z))))

    def batched_call(self, x: int, y: int, condition_sets: Iterable[Iterable[int]]) -> list[float]:
        """Return Fisher-Z p-values for one ``(x, y)`` pair over many conditioning sets.

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
        """Fisher-Z p-values for stacked queries sharing one conditioning-set size.

        ``conds`` are sorted-unique int lists of equal length. Empty and single conditioning sets
        use the closed-form partial correlation; larger sets build the ``(N, dim, dim)`` stack of
        correlation submatrices and invert it once.
        """
        cond_len = len(conds[0])
        df = self._n - cond_len - 3
        if df < 0:
            raise ValueError(
                f"Fisher-Z degrees of freedom is negative (n={self._n}, |S|={cond_len}); "
                "need n > |S| + 3"
            )
        corr = self._corr
        if cond_len == 0:
            r = np.full(len(conds), corr[x_int, y_int], dtype=np.float64)
        elif cond_len == 1:
            z_idx = np.asarray(conds, dtype=np.intp)[:, 0]
            rxz = corr[x_int, z_idx]
            ryz = corr[y_int, z_idx]
            r = (corr[x_int, y_int] - rxz * ryz) / np.sqrt((1.0 - rxz * rxz) * (1.0 - ryz * ryz))
        else:
            dim = 2 + cond_len
            var = np.empty((len(conds), dim), dtype=np.intp)
            var[:, 0] = x_int
            var[:, 1] = y_int
            var[:, 2:] = np.asarray(conds, dtype=np.intp)
            stack = corr[var[:, :, None], var[:, None, :]]
            # Detect rank-deficiency via the symmetric spectrum (a zero eigenvalue), reliable on
            # every BLAS backend -- unlike np.linalg.inv, which raises on OpenBLAS but not Apple
            # Accelerate.
            w = np.linalg.eigvalsh(stack)  # (n, dim), ascending per row
            if (w[:, 0] <= w[:, -1] * 1e-12).any():
                raise ValueError("correlation matrix is singular; cannot run Fisher-Z test")
            inv = np.linalg.inv(stack)
            r = -inv[:, 0, 1] / np.sqrt(np.abs(inv[:, 0, 0] * inv[:, 1, 1]))
        r = np.clip(r, -_ONE_MINUS_EPS, _ONE_MINUS_EPS)
        z = 0.5 * np.log((1.0 + r) / (1.0 - r))
        stat = np.sqrt(df) * np.abs(z)
        return 2.0 * (1.0 - ndtr(stat))

    def marginal_pvalues(self) -> np.ndarray:
        """All-pairs marginal (``|S| == 0``) p-values as a ``(d, d)`` matrix.

        Vectorizes the marginal Fisher-Z closed form over the whole correlation matrix in one shot;
        upper-triangle entry ``[x, y]`` matches ``self(x, y, ())`` exactly, using the same clip
        and z-transform. The lower triangle can inherit rounding asymmetry from the correlation
        matrix. The diagonal is self-correlation and is not meaningful.
        """
        df = self._n - 3
        if df < 0:
            raise ValueError(
                f"Fisher-Z degrees of freedom is negative (n={self._n}, |S|=0); need n > 3"
            )
        r = np.clip(self._corr, -_ONE_MINUS_EPS, _ONE_MINUS_EPS)
        z = 0.5 * np.log((1.0 + r) / (1.0 - r))
        return 2.0 * (1.0 - ndtr(np.sqrt(df) * np.abs(z)))

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
    def from_corr(cls, corr: np.ndarray, n_samples: int) -> FisherZ:
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


# The indep_test registry: a name maps to a factory building a :class:`CITest` over the data.
# Fisher-Z is the only shipped test; the constraint facades resolve their ``indep_test`` argument
# through :func:`make_indep_test`, and :func:`register_indep_test` lets a downstream test or plugin
# add another under the same Protocol -- the generality this registry exposes.
SHIPPED_INDEP_TESTS: tuple[str, ...] = ("fisherz",)
_CI_REGISTRY: dict[str, Callable[[np.ndarray], CITest]] = {"fisherz": FisherZ}


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
