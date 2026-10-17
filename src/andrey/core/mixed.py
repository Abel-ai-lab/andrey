"""Mixed data: the degenerate Gaussian and the conditional Gaussian likelihoods.

Mixed data hold continuous and categorical columns, told apart by a boolean mask over the columns.
Each class here fits one model to any set of variables and reports what the tests in
:mod:`andrey.core.ci` and the scores in :mod:`andrey.core.score` read.

:class:`DegenerateGaussian` replaces each categorical column of ``k`` categories by the indicators
of its first ``k - 1`` categories and treats every column as jointly Gaussian [1]_; one covariance
matrix of these embedded columns answers every query.

:class:`ConditionalGaussian` splits the rows into cells, one per joint category of a set's
categorical variables that occurs, and fits the set's continuous variables in each cell with a
Gaussian of the cell's own mean and a covariance shrunk toward that of all rows; the cell sizes
follow a multinomial [2]_ [3]_.

References
----------
.. [1] Andrews, B., Ramsey, J., and Cooper, G. F. (2019). Learning high-dimensional directed
   acyclic graphs with mixed data-types. Proceedings of Machine Learning Research 104, 4-21.
.. [2] Andrews, B., Ramsey, J., and Cooper, G. F. (2018). Scoring Bayesian networks of mixed
   variables. International Journal of Data Science and Analytics 6(1), 3-18.
   https://doi.org/10.1007/s41060-017-0085-7
.. [3] Friedman, J. H. (1989). Regularized discriminant analysis. Journal of the American
   Statistical Association 84(405), 165-175.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from functools import cache
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from . import categorical as categories
from . import stats

# A fitted covariance, on the scale of the standardized columns, counts as singular when its least
# eigenvalue is at most this.
_SINGULAR = 1e-12
# A batch of fitted covariances takes the Cholesky path only for sets whose every covariance has a
# condition number certified below this; the others take the eigenvalues.
_CONDITION = 1e8
_LOG_2PI = math.log(2.0 * math.pi)
# The memory, in bytes, the kept cell numberings of categorical-variable sets may take.
_CELL_BYTES = 64 * 2**20


def column_mask(categorical: npt.ArrayLike | None, d: int) -> np.ndarray:
    """Return ``categorical`` as a boolean mask over ``d`` columns; ``None`` marks none.

    Raises ``ValueError`` when the mask does not hold one entry per column.
    """
    if categorical is None:
        return np.zeros(d, dtype=bool)
    mask = np.asarray(categorical, dtype=bool).reshape(-1)
    if mask.shape != (d,):
        raise ValueError(f"categorical must mark each of the {d} columns, got {mask.size} entries")
    return mask


def _matrix(data: npt.ArrayLike) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"data must be a 2-D (n_samples, n_features) array, got ndim={arr.ndim}")
    if not np.isfinite(arr).all():
        raise ValueError("data contains NaN or Inf; cannot fit a mixed-data model")
    return arr


class DegenerateGaussian:
    """The covariance of the degenerate Gaussian embedding of mixed data.

    A continuous column is kept as it is. A categorical column with ``k`` categories, each distinct
    value one category, becomes ``k - 1`` columns: the indicators of its categories in ascending
    order of value, the last left out, since the ``k`` indicators sum to one [1]_. Which category is
    left out does not change a test or the comparison of two graphs' scores.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Finite values; categorical columns hold their categories as numbers.
    categorical : array-like of bool, shape (n_features,), optional
        Marks the categorical columns; ``None`` marks none.

    Attributes
    ----------
    n : int
        The number of rows.
    cov : ndarray, shape (p, p)
        Sample covariance (``ddof=1``) of the ``p`` embedded columns.
    groups : list of ndarray
        ``groups[j]`` lists the embedded columns of variable ``j``.

    References
    ----------
    .. [1] Andrews, B., Ramsey, J., and Cooper, G. F. (2019). Learning high-dimensional directed
       acyclic graphs with mixed data-types. Proceedings of Machine Learning Research 104, 4-21.
    """

    def __init__(self, data: npt.ArrayLike, categorical: npt.ArrayLike | None = None) -> None:
        arr = _matrix(data)
        mask = column_mask(categorical, arr.shape[1])
        blocks: list[np.ndarray] = []
        self.groups: list[np.ndarray] = []
        width = 0
        for j in range(arr.shape[1]):
            if mask[j]:
                codes = np.unique(arr[:, j], return_inverse=True)[1].reshape(-1)
                block = (codes[:, None] == np.arange(int(codes.max(initial=0)))).astype(np.float64)
            else:
                block = arr[:, j : j + 1]
            blocks.append(block)
            self.groups.append(np.arange(width, width + block.shape[1]))
            width += block.shape[1]
        self.n: int = arr.shape[0]
        embedded = np.hstack(blocks) if blocks else np.empty((arr.shape[0], 0))
        self.cov: np.ndarray = np.atleast_2d(stats.cov(embedded, rowvar=False)).reshape(
            width, width
        )

    def columns(self, variables: Sequence[int]) -> np.ndarray:
        """The embedded columns of ``variables``, in the order given."""
        if not len(variables):
            return np.zeros(0, dtype=np.intp)
        return np.concatenate([self.groups[int(v)] for v in variables])


class ConditionalGaussian:
    """Conditional Gaussian log-likelihoods and parameter counts of variable sets.

    For a set of variables, the rows split into cells, one per joint category of the set's
    categorical variables that occurs in the data [1]_. With ``n_c`` rows in cell ``c`` and ``d``
    continuous variables in the set, the cell's continuous variables are fit with a Gaussian of
    the cell's mean and the covariance

    ``C_c = (n_c S_c + R) / (n_c + 1)``

    where ``S_c`` is the cell's maximum-likelihood covariance, about its mean, and ``R`` the
    covariance of all the rows: the cell's estimate shrunk toward ``R`` by one pseudo-row, as in
    regularized discriminant analysis [2]_. A large cell keeps nearly its own estimate, and a cell
    with too few rows for its own, such as one with ``d`` rows or fewer, takes mostly ``R``'s, so
    every set has a finite log-likelihood, and nested sets are fit alike. The log-likelihood is

    ``sum_c [n_c log(n_c / n) - n_c / 2 (log det C_c + d log(2 pi) + tr(C_c^-1 S_c))]``

    and the parameter count is ``K (d (d + 3) / 2 + 1) - 1``, with ``K`` the product of the
    categorical variables' numbers of categories: a mean, a covariance, and a weight for every joint
    category, whether it occurs or not, less one weight, since the weights sum to one.

    Log-likelihoods are memoized by the sorted set of variables. Within the memory bound, each
    categorical set keeps its cells and their sums of the continuous columns and of their products,
    which every set with those categorical variables reads. :meth:`loglik_many` fits all of a
    request's sets together.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Finite values; categorical columns hold their categories as numbers, each distinct value of
        a column one category.
    categorical : array-like of bool, shape (n_features,), optional
        Marks the categorical columns; ``None`` marks none.
    cache_size : int or None, default=1_000_000
        Memoization cache bound; ``None`` is unbounded.

    Raises
    ------
    ValueError
        If a continuous column is constant; :meth:`loglik` also raises one for a set whose
        continuous columns are collinear.

    References
    ----------
    .. [1] Andrews, B., Ramsey, J., and Cooper, G. F. (2018). Scoring Bayesian networks of mixed
       variables. International Journal of Data Science and Analytics 6(1), 3-18.
    .. [2] Friedman, J. H. (1989). Regularized discriminant analysis. Journal of the American
       Statistical Association 84(405), 165-175.
    """

    def __init__(
        self,
        data: npt.ArrayLike,
        categorical: npt.ArrayLike | None = None,
        *,
        cache_size: int | None = 1_000_000,
    ) -> None:
        arr = _matrix(data)
        n, d = arr.shape
        self._mask = column_mask(categorical, d)
        self.n: int = n
        self._codes = np.zeros((n, d), dtype=np.int64)
        self._cardinalities = np.ones(d, dtype=np.int64)
        if self._mask.any():
            codes, cards = categories.encode(arr[:, self._mask])
            self._codes[:, self._mask] = codes
            self._cardinalities[self._mask] = cards
        # Continuous columns are standardized; ``_log_scale`` restores their units.
        scale = arr.std(axis=0)
        constant = np.flatnonzero(~self._mask & (scale == 0))
        if constant.size:
            raise ValueError(f"continuous column {int(constant[0])} is constant")
        self._codes = np.asfortranarray(self._codes)
        # ``_slot[v]`` is continuous variable v's row of ``_columns``.
        continuous = np.flatnonzero(~self._mask)
        z = (arr[:, continuous] - arr[:, continuous].mean(axis=0)) / scale[continuous]
        self._slot = np.full(d, -1, dtype=np.intp)
        self._slot[continuous] = np.arange(continuous.size)
        self._log_scale = np.log(scale[continuous])
        self._gram = z.T @ z / n  # the covariance of all rows
        # The standardized continuous columns as rows and, last, a row of ones, whose products with
        # a column are its sums.
        self._columns = np.vstack((z.T, np.ones((1, n))))
        self._one = continuous.size
        self._memo = _LRU(cache_size)
        self._stats = _LRU(_CELL_BYTES, weigh=_nbytes)

    def loglik(self, variables: Sequence[int]) -> float:
        """Return the log-likelihood of the set ``variables``; ``0`` for the empty set."""
        return self.loglik_many([variables])[0]

    def loglik_many(self, sets: Sequence[Sequence[int]]) -> list[float]:
        """Return the log-likelihood of each set in ``sets``, in order.

        The sets are fit together, and each gets the value it gets alone.
        """
        keys = [tuple(sorted({int(v) for v in variables})) for variables in sets]
        found = {key: self._memo.get(key) for key in keys}
        todo = [key for key, value in found.items() if value is None]
        if todo:
            for key, value in zip(todo, self._fit(todo), strict=True):
                found[key] = value
                self._memo.put(key, value)
        return [found[key] for key in keys]

    def dof(self, variables: Sequence[int]) -> int:
        """Return the parameter count of the model of the set ``variables``; ``0`` for none."""
        held = sorted({int(v) for v in variables})
        joint = math.prod(int(self._cardinalities[v]) for v in held if self._mask[v])
        d = sum(1 for v in held if not self._mask[v])
        return joint * (d * (d + 3) // 2 + 1) - 1

    def _cells(self, discrete: tuple[int, ...]) -> _Cells:
        """The cells of the categorical set ``discrete``, kept while the memory bound allows."""
        cells = self._stats.get(discrete)
        if cells is None:
            cell, m = categories.configurations(self._codes, self._cardinalities, discrete)
            counts = np.bincount(cell, minlength=m).astype(np.float64)
            loglik = float(np.sum(counts * np.log(counts / self.n)))
            cells = _Cells(cell, counts, loglik)
            self._stats.put(discrete, cells)
        return cells

    def _moments(self, pairs: list[tuple[tuple[int, ...], int, int]]) -> list[np.ndarray]:
        """Per-cell sums of products for each ``(categorical set, row a, row b)`` in ``pairs``.

        The row of ones makes ``(a, ones)`` the sums of ``a`` and ``(ones, ones)`` the cell sizes.
        Each result is kept under its triple, but the cell sizes, which the cells keep.
        """
        get, put, one = self._stats.get, self._stats.put, self._one
        found = []
        for pair in pairs:
            discrete, a, b = pair
            if a == one:
                found.append(self._cells(discrete).counts)
                continue
            value = get(pair)
            if value is None:
                cells = self._cells(discrete)
                weights = self._columns[a] if b == one else self._columns[a] * self._columns[b]
                value = np.bincount(cells.cell, weights, minlength=cells.counts.size)
                put(pair, value)
            found.append(value)
        return found

    def _fit(self, keys: list[tuple[int, ...]]) -> list[float]:
        """Log-likelihoods of the sets ``keys``.

        The cells of all the sets with ``d`` continuous variables are stacked, one row per cell of
        each set, and fit in one pass; each set's terms are then summed over its own rows.
        """
        mask, slot, one = self._mask, self._slot, self._one
        out = np.empty(len(keys))
        cells_of: list[_Cells] = []
        pairs_of: list[list[int]] = []
        by_size: dict[int, list[int]] = {}
        at: dict[tuple[tuple[int, ...], int, int], int] = {}
        for g, key in enumerate(keys):
            discrete = tuple(v for v in key if mask[v])
            cells = self._cells(discrete)
            cells_of.append(cells)
            row = [int(slot[v]) for v in key if not mask[v]]
            pairs_of.append([])
            if not row:
                out[g] = cells.loglik
                continue
            row.append(one)
            by_size.setdefault(len(row) - 1, []).append(g)
            pairs_of[g] = [
                at.setdefault((discrete, a, b), len(at)) for i, a in enumerate(row) for b in row[i:]
            ]
        if not at:
            return out.tolist()
        found = self._moments(list(at))
        offsets = np.cumsum([0] + [value.size for value in found])
        flat = np.concatenate(found)
        for d, members in by_size.items():
            sizes = np.array([cells_of[g].counts.size for g in members])
            starts = np.cumsum(sizes) - sizes
            owner = np.repeat(np.arange(len(members)), sizes)  # each row's set
            local = np.arange(owner.size) - starts[owner]  # each row's cell within its set
            where = offsets[np.array([pairs_of[g] for g in members])]
            # [r, t]: row r's sums of products of its set's t-th column pair, ones column last.
            moments = flat[where[owner] + local[:, None]]
            slots = np.array([[slot[v] for v in keys[g] if not mask[v]] for g in members])
            sets = [keys[g] for g in members]
            if d == 1:  # the pairs: the column with itself, with the ones column, ones with ones
                counts = moments[:, 2]
                sums = moments[:, 1]
                scatter = moments[:, 0] - sums * sums / counts
                whole = self._gram[slots[:, 0], slots[:, 0]][owner]
                fitted = (scatter + whole) / (counts + 1.0)
                _check_singular(fitted <= _SINGULAR, owner, sets, mask)
                log_det, quad = np.log(fitted), scatter / fitted
            else:
                full = moments[:, _triangle(d)]
                counts = full[:, d, d]
                sums = full[:, :d, d]
                scatter = (
                    full[:, :d, :d] - sums[:, :, None] * sums[:, None, :] / counts[:, None, None]
                )
                # Each cell's covariance, shrunk toward that of all rows.
                whole = self._gram[slots[:, :, None], slots[:, None, :]][owner]
                fitted = (scatter + whole) / (counts + 1.0)[:, None, None]
                log_det, quad = _gaussian_terms(fitted, scatter, starts, owner, sets, mask)
            terms = counts * (log_det + d * _LOG_2PI) + quad
            loglik = np.array([cells_of[g].loglik for g in members])
            values = loglik - 0.5 * np.add.reduceat(terms, starts)
            out[members] = values - self.n * np.sum(self._log_scale[slots], axis=1)
        return out.tolist()


class _Cells(NamedTuple):
    """One categorical set's cells."""

    cell: np.ndarray  # each row's cell
    counts: np.ndarray  # rows per cell
    loglik: float  # the multinomial log-likelihood of the counts


@cache
def _triangle(d: int) -> np.ndarray:
    """``[a, b]``: the position of the pair ``(min, max)`` in the upper triangle of ``d + 1``."""
    e = d + 1
    position = {}
    for a in range(e):
        for b in range(a, e):
            position[a, b] = len(position)
    return np.array([[position[min(a, b), max(a, b)] for b in range(e)] for a in range(e)])


def _gaussian_terms(
    fitted: np.ndarray,
    scatter: np.ndarray,
    starts: np.ndarray,
    owner: np.ndarray,
    sets: list[tuple[int, ...]],
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``log det C`` and ``tr(C^-1 S)`` for each fitted covariance ``C`` and scatter ``S``.

    Row ``r`` of the ``(rows, d, d)`` stacks is a cell of set ``owner[r]``, whose rows begin at
    ``starts``. A set's Cholesky factors give its log-determinants and its inverses the traces.
    A set takes its eigenvalues instead when a cell's condition number is not certified below
    ``_CONDITION`` by ``||C||_F ||C^-1||_F``, and every set does when a factor fails. A set with a
    cell's least eigenvalue at or below ``_SINGULAR`` raises ``ValueError``.
    """
    try:
        factor, inverse = np.linalg.cholesky(fitted), np.linalg.inv(fitted)
        norms = np.sum(inverse**2, axis=(-2, -1)) * np.sum(fitted**2, axis=(-2, -1))
        certified = norms < _CONDITION**2
    except np.linalg.LinAlgError:
        factor, inverse = np.ones_like(fitted), np.zeros_like(fitted)
        certified = np.zeros(len(fitted), dtype=bool)
    exact = ~np.logical_and.reduceat(certified, starts)[owner]
    log_det = 2.0 * np.sum(np.log(np.diagonal(factor, axis1=-2, axis2=-1)), axis=-1)
    quad = np.sum(inverse * scatter, axis=(-2, -1))
    if exact.any():
        eig = np.linalg.eigvalsh(fitted[exact])
        _check_singular(eig[:, 0] <= _SINGULAR, owner[exact], sets, mask)
        log_det[exact] = np.sum(np.log(eig), axis=-1)
        quad[exact] = np.trace(np.linalg.solve(fitted[exact], scatter[exact]), axis1=-2, axis2=-1)
    return log_det, quad


def _check_singular(
    singular: np.ndarray, owner: np.ndarray, sets: list[tuple[int, ...]], mask: np.ndarray
) -> None:
    """Raise ``ValueError`` for the first set owning a ``singular`` row."""
    if singular.any():
        continuous = [v for v in sets[int(owner[np.argmax(singular)])] if not mask[v]]
        raise ValueError(f"continuous columns {continuous} are collinear; cannot fit them")


def _nbytes(value: np.ndarray | tuple[Any, ...]) -> int:
    if isinstance(value, np.ndarray):
        return value.nbytes
    return sum(a.nbytes for a in value if isinstance(a, np.ndarray))


class _LRU:
    """A map that drops its least recently used entries once their total weight passes ``limit``.

    ``None`` sets no limit; ``weigh`` gives an entry's weight, one by default.
    """

    def __init__(self, limit: int | None, weigh: Callable[[Any], int] | None = None) -> None:
        self._data: OrderedDict[Hashable, Any] = OrderedDict()
        self._limit = math.inf if limit is None else limit
        self._weigh = weigh
        self._weight = 0

    def get(self, key: Hashable) -> Any:
        """The value under ``key``, now the most recently used, or ``None``."""
        value = self._data.get(key)
        if value is not None:
            self._data.move_to_end(key)
        return value

    def put(self, key: Hashable, value: Any) -> None:
        """Keep ``value`` under ``key``, dropping the oldest entries past the limit."""
        if key in self._data:
            old = self._data.pop(key)
            self._weight -= 1 if self._weigh is None else self._weigh(old)
        self._data[key] = value
        self._weight += 1 if self._weigh is None else self._weigh(value)
        while self._weight > self._limit and len(self._data) > 1:
            _, old = self._data.popitem(last=False)
            self._weight -= 1 if self._weigh is None else self._weigh(old)
