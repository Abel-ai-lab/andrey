"""Mixed data: the degenerate Gaussian and the conditional Gaussian likelihoods.

Mixed data hold continuous and categorical columns, told apart by a boolean mask over the columns.
Each class here fits one model to any set of variables and reports what the tests in
:mod:`andrey.core.ci` and the scores in :mod:`andrey.core.score` read.

:class:`DegenerateGaussian` replaces each categorical column of ``k`` categories by the indicators
of its first ``k - 1`` categories and treats every column as jointly Gaussian [1]_; one covariance
matrix of these embedded columns answers every query.

:class:`ConditionalGaussian` splits the rows into cells, one per joint category of a set's
categorical variables that occurs, and fits the set's continuous variables in each cell with a
Gaussian of the cell's own mean and covariance; the cell sizes follow a multinomial [2]_.

References
----------
.. [1] Andrews, B., Ramsey, J., and Cooper, G. F. (2019). Learning high-dimensional directed
   acyclic graphs with mixed data-types. Proceedings of Machine Learning Research 104, 4-21.
.. [2] Andrews, B., Ramsey, J., and Cooper, G. F. (2018). Scoring Bayesian networks of mixed
   variables. International Journal of Data Science and Analytics 6(1), 3-18.
   https://doi.org/10.1007/s41060-017-0085-7
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from . import categorical as categories
from . import stats

# A cell's covariance, on the scale of the standardized columns, counts as singular when its least
# eigenvalue is at most this.
_SINGULAR = 1e-12
_LOG_2PI = math.log(2.0 * math.pi)


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
    """Maximized conditional Gaussian log-likelihoods and parameter counts of variable sets.

    For a set of variables, the rows split into cells, one per joint category of the set's
    categorical variables that occurs in the data. With ``n_c`` rows in cell ``c`` and ``d``
    continuous variables in the set, the log-likelihood is [1]_

    ``sum_c [n_c log(n_c / n) - n_c / 2 (log det S_c + d log(2 pi) + d)]``

    where ``S_c`` is the cell's maximum-likelihood covariance of the continuous variables, about
    the cell's mean. A cell whose covariance is singular, such as one with ``d`` rows or fewer,
    instead takes the mean and covariance of all the rows. The parameter count is
    ``K (d (d + 3) / 2 + 1) - 1``, with ``K`` the product of the categorical variables' numbers of
    categories: a mean, a covariance, and a weight for every joint category, whether it occurs or
    not, less one weight, since the weights sum to one.

    Log-likelihoods are memoized by the sorted set of variables.

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
        If a continuous column is constant.

    References
    ----------
    .. [1] Andrews, B., Ramsey, J., and Cooper, G. F. (2018). Scoring Bayesian networks of mixed
       variables. International Journal of Data Science and Analytics 6(1), 3-18.
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
        scale[self._mask] = 1.0
        self._z = np.where(self._mask, 0.0, (arr - arr.mean(axis=0)) / scale)
        self._log_scale = np.log(scale)
        self._loglik = lru_cache(maxsize=cache_size)(self._loglik_impl)

    def loglik(self, variables: Sequence[int]) -> float:
        """Return the maximized log-likelihood of the set ``variables``; ``0`` for the empty set."""
        return self._loglik(tuple(sorted({int(v) for v in variables})))

    def dof(self, variables: Sequence[int]) -> int:
        """Return the parameter count of the model of the set ``variables``; ``0`` for none."""
        held = sorted({int(v) for v in variables})
        joint = math.prod(int(self._cardinalities[v]) for v in held if self._mask[v])
        d = sum(1 for v in held if not self._mask[v])
        return joint * (d * (d + 3) // 2 + 1) - 1

    def _loglik_impl(self, variables: tuple[int, ...]) -> float:
        discrete = [v for v in variables if self._mask[v]]
        continuous = [v for v in variables if not self._mask[v]]
        cell, m = categories.configurations(self._codes, self._cardinalities, discrete)
        counts = np.bincount(cell, minlength=m).astype(np.float64)
        n = self.n
        loglik = float(np.sum(counts * np.log(counts / n)))
        d = len(continuous)
        if d == 0:
            return loglik
        z = self._z[:, continuous]
        sums = np.stack([np.bincount(cell, z[:, a], minlength=m) for a in range(d)], axis=1)
        means = sums / counts[:, None]
        resid = z - means[cell]
        scatter = np.empty((m, d, d))
        for a in range(d):
            for b in range(a, d):
                scatter[:, a, b] = scatter[:, b, a] = np.bincount(
                    cell, resid[:, a] * resid[:, b], minlength=m
                )
        eig = np.linalg.eigvalsh(scatter / counts[:, None, None])
        own = (counts > d) & (eig[:, 0] > _SINGULAR)
        log_det = np.sum(np.log(eig[own]), axis=1)
        loglik -= 0.5 * float(np.sum(counts[own] * (log_det + d * (_LOG_2PI + 1.0))))
        if not own.all():
            # The rest take the mean (zero, the columns are centered) and covariance of all rows.
            w, u = np.linalg.eigh(z.T @ z / n)
            if w[0] <= _SINGULAR:
                raise ValueError(
                    f"continuous columns {continuous} are collinear; cannot fit a cell with too "
                    "few rows to estimate its own covariance"
                )
            rest = ~own
            second = scatter[rest] + counts[rest, None, None] * (
                means[rest, :, None] * means[rest, None, :]
            )
            quad = np.einsum("ij,cji->c", (u / w) @ u.T, second)
            pooled = float(np.sum(np.log(w))) + d * _LOG_2PI
            loglik -= 0.5 * float(np.sum(counts[rest] * pooled + quad))
        return loglik - n * float(np.sum(self._log_scale[continuous]))
