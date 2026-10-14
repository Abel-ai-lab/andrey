"""BIC scoring substrate for linear-Gaussian structure search.

The negative local BIC score (lower is better; proportional to ``-2 * BIC``) for a target node
given a parent set, evaluated from the sample covariance. Uses the linear-Gaussian BIC closed
form, branching on parent-set size for the scalar fast paths.

:func:`local_score_bic` is the pure per-node primitive; :class:`BICScore` precomputes the covariance
once and memoizes scores behind the :class:`Score` protocol that structure-search engines consume.

References
----------
Schwarz (1978). Estimating the Dimension of a Model. Annals of Statistics 6(2):461-464.
https://doi.org/10.1214/aos/1176344136
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol, runtime_checkable

import numpy as np

from . import stats

# A residual variance at most this fraction of the node's variance is zero up to rounding.
ZERO_RESIDUAL = 1e-12


def local_score_bic(
    cov: np.ndarray,
    n: int,
    node: int,
    parents: Sequence[int],
    *,
    lambda_value: float = 1.0,
    log_n: float | None = None,
    cov_diag: np.ndarray | None = None,
) -> float:
    """Return the negative local BIC score for ``node`` given ``parents``.

    Evaluates ``n * log R(node | parents) + log_n * len(parents) * lambda_value``, where ``R`` is
    the linear-regression residual variance
    ``cov[node, node] - cov[node, parents] @ inv(cov[parents, parents]) @ cov[parents, node]``.
    Lower is better; the value is proportional to ``-2 * BIC``.

    The family is unavailable, and the score ``inf``, when ``parents`` has more than ``n - 2``
    members, which fit ``node`` exactly, or when ``R`` is at most :data:`ZERO_RESIDUAL` times
    ``cov[node, node]``, zero up to rounding. A search that minimizes the score never selects an
    unavailable family.

    Branches on ``len(parents)`` for scalar fast paths (0/1/2 parents); the general path uses
    ``np.linalg.solve`` with a ``pinv`` fallback on singular ``cov[parents, parents]``.

    Parameters
    ----------
    cov : ndarray, shape (d, d)
        Sample covariance matrix (``np.cov(data, rowvar=False)``, ddof=1).
    n : int
        Sample count.
    node : int
        Target column index.
    parents : sequence of int
        Parent column indices.
    lambda_value : float
        Weight on the BIC complexity term.
    log_n : float, optional
        Precomputed ``log(n)``; computed from ``n`` when omitted.
    cov_diag : ndarray, optional
        Precomputed diagonal of ``cov``; computed from ``cov`` when omitted.
    """
    if log_n is None:
        log_n = float(np.log(n))
    if cov_diag is None:
        cov_diag = np.diag(cov)
    node = int(node)
    n_parents = len(parents)
    if n_parents > n - 2:
        return math.inf
    variance = float(cov_diag[node])
    penalty = log_n * n_parents * lambda_value

    if n_parents == 0:
        return _from_residual(variance, variance, n, penalty)

    if n_parents == 1:
        parent = int(parents[0])
        parent_var = float(cov_diag[parent])
        if parent_var != 0.0:
            cov_ip = float(cov[node, parent])
            residual = variance - cov_ip * (cov_ip / parent_var)
            return _from_residual(residual, variance, n, penalty)
        yX = cov[node : node + 1, parent : parent + 1]
        XX = cov[parent : parent + 1, parent : parent + 1]
        beta = _solve(XX, yX.T)
        return _from_residual(variance - (yX @ beta).item(), variance, n, penalty)

    if n_parents == 2:
        parent0 = int(parents[0])
        parent1 = int(parents[1])
        yX = np.array([[cov[node, parent0], cov[node, parent1]]])
        XX = np.array(
            [
                [cov[parent0, parent0], cov[parent0, parent1]],
                [cov[parent1, parent0], cov[parent1, parent1]],
            ]
        )
        beta = _solve(XX, yX.T)
        return _from_residual(variance - (yX @ beta).item(), variance, n, penalty)

    pa = [int(p) for p in parents]
    yX = cov[node : node + 1, pa]
    XX = cov[pa][:, pa]
    beta = _solve(XX, yX.T)
    return _from_residual(variance - (yX @ beta).item(), variance, n, penalty)


def _from_residual(residual: float, variance: float, n: int, penalty: float) -> float:
    """``n * log(residual) + penalty``, or ``inf`` when ``residual`` is zero up to rounding."""
    if not residual > ZERO_RESIDUAL * variance:
        return math.inf
    return float(n * np.log(residual) + penalty)


def check_columns(cov: np.ndarray, n: int) -> None:
    """Raise ``ValueError`` when a column of the data behind ``cov`` makes BIC scoring meaningless.

    A column is rejected when it is constant; with more than two rows ``n``, when its residual
    variance given one other column is at most ``ZERO_RESIDUAL`` times its variance (an affine copy
    of that column); and with more rows than columns, when the correlation matrix's smallest
    eigenvalue is at most ``ZERO_RESIDUAL`` times its largest (some columns are linear functions of
    others). Every column is a linear function of any ``n - 1`` others, so the last two checks
    apply only where the row count does not force such a fit.

    Parameters
    ----------
    cov : ndarray, shape (d, d)
        Sample covariance matrix.
    n : int
        Sample count.

    Raises
    ------
    ValueError
        If a column is constant, an affine copy of another column when ``n`` exceeds 2, or a linear
        function of other columns when ``n`` exceeds the column count.
    """
    variance = np.diag(cov)
    constant = np.flatnonzero(~(variance > 0.0))
    if constant.size:
        raise ValueError(f"data column {int(constant[0])} is constant; drop it before the fit")
    d = cov.shape[0]
    if d < 2 or n <= 2:
        return
    sd = np.sqrt(variance)
    corr = cov / np.outer(sd, sd)
    alike = np.abs(corr)
    np.fill_diagonal(alike, 0.0)
    i, j = sorted(int(k) for k in np.unravel_index(np.argmax(alike), alike.shape))
    if 1.0 - alike[i, j] ** 2 <= ZERO_RESIDUAL:
        raise ValueError(
            f"data column {j} is a linear function of column {i}; drop one of them before the fit"
        )
    if n <= d:
        return
    values = np.linalg.eigvalsh(corr)
    if values[0] <= ZERO_RESIDUAL * values[-1]:
        weights = np.abs(np.linalg.eigh(corr)[1][:, 0])
        columns = np.flatnonzero(weights > 1e-6 * weights.max()).tolist()
        raise ValueError(
            f"data columns {columns} are linearly dependent: one is a linear function of the "
            "others; drop one of them before the fit"
        )


def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``a @ x = b``, falling back to the pseudo-inverse when ``a`` is singular."""
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a) @ b


@runtime_checkable
class Score(Protocol):
    """Contract for a scoring object: the negative local score of a node given its parents."""

    def score(self, node: int, parents: Sequence[int]) -> float:
        """Return the negative local score for ``node`` given ``parents`` (lower is better)."""
        ...


class BICScore:
    """Linear-Gaussian BIC score over a fixed dataset.

    Precomputes the covariance, sample count, and ``log(n)``. Memoizes negative local scores.

    Parameters
    ----------
    data : ndarray, shape (n, d)
        Data matrix; rows are observations, columns are variables.
    lambda_value : float, default=1.0
        Weight on the BIC complexity term of the deviance the search minimizes.
    cache_size : int or None, default=1_000_000
        Memoization cache bound; ``None`` is unbounded.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, holds ``NaN`` or ``inf``, or has a column :func:`check_columns`
        rejects.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        lambda_value: float = 1.0,
        cache_size: int | None = 1_000_000,
    ) -> None:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(
                f"data must be a 2-D (n_samples, n_features) array, got ndim={data.ndim}"
            )
        if not np.isfinite(data).all():
            raise ValueError("data contains NaN or Inf; cannot compute BIC scores")
        # np.cov of a single column is a 0-d variance; keep the (1, 1) matrix the score reads.
        cov = np.atleast_2d(stats.cov(data, rowvar=False))
        check_columns(cov, data.shape[0])
        self._init(cov, data.shape[0], lambda_value, cache_size)

    @classmethod
    def from_cov(
        cls,
        cov: np.ndarray,
        n: int,
        *,
        lambda_value: float = 1.0,
        cache_size: int | None = 1_000_000,
    ) -> BICScore:
        """Build a scorer from a precomputed covariance.

        Pooled workers use the parent's ``cov`` bytes, giving every process identical inputs to
        :func:`local_score_bic` and bit-identical scores on the same host BLAS. The columns are not
        checked with :func:`check_columns`.

        Parameters
        ----------
        cov : ndarray, shape (d, d)
            Sample covariance matrix; must be square and finite.
        n : int
            Sample count the BIC complexity penalty uses.
        lambda_value : float, default=1.0
            Weight on the BIC complexity term of the deviance the search minimizes.
        cache_size : int or None, default=1_000_000
            Memoization cache bound; ``None`` is unbounded.

        Raises
        ------
        ValueError
            If ``cov`` is not a square 2-D matrix, or holds ``NaN`` / ``inf``.
        """
        cov = np.asarray(cov, dtype=np.float64)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(f"cov must be a square 2-D matrix, got shape {cov.shape}")
        if not np.isfinite(cov).all():
            raise ValueError("cov contains NaN or Inf; cannot compute BIC scores")
        obj = cls.__new__(cls)
        obj._init(cov, int(n), lambda_value, cache_size)
        return obj

    def _init(self, cov: np.ndarray, n: int, lambda_value: float, cache_size: int | None) -> None:
        lambda_value = float(lambda_value)
        if not np.isfinite(lambda_value) or lambda_value < 0.0:
            raise ValueError(f"lambda_value must be finite and non-negative, got {lambda_value}")
        self.cov = cov
        self.n = n
        self.lambda_value = lambda_value
        self.log_n = float(np.log(self.n))
        self._cov_diag = np.diag(self.cov)
        self._score_key = lru_cache(maxsize=cache_size)(self._score_key_impl)

    def _score_key_impl(self, node: int, parents: tuple[int, ...]) -> float:
        return local_score_bic(
            self.cov,
            self.n,
            node,
            parents,
            lambda_value=self.lambda_value,
            log_n=self.log_n,
            cov_diag=self._cov_diag,
        )

    def score(self, node: int, parents: Sequence[int]) -> float:
        """Return the memoized negative local BIC score for ``node`` given ``parents``.

        The cache is keyed by ``(node, sorted-parents-tuple)``, so parent-set order does not affect
        the lookup.
        """
        key = tuple(sorted(int(p) for p in parents))
        return self._score_key(int(node), key)

    def score_many(self, node: int, parent_sets: Sequence[Sequence[int]]) -> list[float]:
        """Return the negative local BIC scores for many parent sets, in request order."""
        return [self.score(node, parents) for parents in parent_sets]
