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

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol, runtime_checkable

import numpy as np

from . import stats


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

    if n_parents == 0:
        return float(n * np.log(cov_diag[node]))

    if n_parents == 1:
        parent = int(parents[0])
        parent_var = float(cov_diag[parent])
        if parent_var != 0.0:
            cov_ip = float(cov[node, parent])
            residual = float(cov_diag[node]) - cov_ip * (cov_ip / parent_var)
            return float(n * np.log(residual) + log_n * lambda_value)
        yX = cov[node : node + 1, parent : parent + 1]
        XX = cov[parent : parent + 1, parent : parent + 1]
        beta = _solve(XX, yX.T)
        H = float(np.log(cov_diag[node] - (yX @ beta).item()))
        return float(n * H + log_n * lambda_value)

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
        H = float(np.log(cov_diag[node] - (yX @ beta).item()))
        return float(n * H + log_n * 2 * lambda_value)

    pa = [int(p) for p in parents]
    yX = cov[node : node + 1, pa]
    XX = cov[pa][:, pa]
    beta = _solve(XX, yX.T)
    H = np.log(cov_diag[node] - (yX @ beta).item())
    return float(n * H + log_n * n_parents * lambda_value)


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
        :func:`local_score_bic` and bit-identical scores on the same host BLAS.

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
