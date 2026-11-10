"""Stein estimators of a density's score and of the diagonal of the score's Jacobian.

The score of a density ``p`` on ``R^d`` is ``grad log p``. Both estimators read ``n`` rows drawn
from ``p`` and return their estimate at those rows. NoGAM (:mod:`andrey.fcm.nogam`) reads the
score, and SCORE (:mod:`andrey.fcm.score_matching`) the diagonal of its Jacobian.

**The score.** Stein's identity, ``E[h(x) grad log p(x)^T + grad h(x)] = 0`` for a test function
``h`` that vanishes at infinity, becomes a ridge regression once its expectation is the mean over
the rows, and the kernel trick gives the Stein gradient estimator of Li and Turner (2018)::

    G = -(K + eta I)^-1 <grad, K>,    <grad, K>[i, j] = sum_k d kappa(x_i, x_k) / d x_k[j],

with ``K[i, k] = kappa(x_i, x_k)`` the kernel matrix of the rows and ``eta`` the ridge term.

**The Jacobian's diagonal.** Rolland et al. (2022, Eq. 12) apply the second-order Stein identity
the same way::

    J = -G * G + (K + eta I)^-1 <grad^2, K>,
    <grad^2, K>[i, j] = sum_k d^2 kappa(x_i, x_k) / d x_k[j]^2,

where ``G * G`` squares each entry and row ``i`` of ``J`` estimates the diagonal of the Hessian of
``log p`` at ``x_i``. The same ridge term serves both regressions (their Section 3.2).

**The kernel.** ``kappa(x, y) = exp(-||x - y||^2 / (2 s^2))``, with the bandwidth ``s`` the median
distance between the rows (their Algorithm 2). For this kernel

- ``<grad, K>[i, j] = sum_k K[i, k] (x_i[j] - x_k[j]) / s^2``, and
- ``<grad^2, K>[i, j] = sum_k K[i, k] ((x_i[j] - x_k[j])^2 / s^4 - 1 / s^2)``,

each computed from ``K X``, ``K X^2``, and the row sums of ``K``, with no array of shape
``(n, n, d)``.

The choices the papers leave open:

- The ridge term is :data:`ETA`, ``0.001``. Rolland et al. (2022) state no value; ``0.001`` is the
  value of the code published with the paper. It is added to the kernel matrix as Eq. 5 and
  Eq. 12 write it, a matrix with ones on its diagonal.
- The median is taken over the pairs of distinct rows.
- One kernel reads every column, so the estimates depend on the columns' units. A caller whose
  result should not depend on them passes standardized columns.

Each call factors one ``(n, n)`` matrix by Cholesky, so its time grows as ``n^3`` and its memory
as ``n^2``. Neither estimator draws random numbers.

References
----------
Li, Turner. "Gradient estimators for implicit models." ICLR, 2018.

Rolland, Cevher, Kleindessner, Russell, Janzing, Scholkopf, Locatello. "Score matching enables
causal discovery of nonlinear additive noise models." ICML, PMLR 162:18741-18753, 2022.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

#: Ridge term of both regressions.
ETA = 0.001


@dataclass(frozen=True)
class _Kernel:
    """The kernel matrix of the rows, its bandwidth, and the factor of the ridge system."""

    matrix: np.ndarray
    bandwidth: float
    factor: tuple[np.ndarray, bool]

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Return ``(K + eta I)^-1 rhs``."""
        return cho_solve(self.factor, rhs)


def squared_distances(X: np.ndarray) -> np.ndarray:
    """The ``(n, n)`` squared distances between the rows of ``X``, with a zero diagonal."""
    sq = np.einsum("ij,ij->i", X, X)
    dist = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(dist, 0.0, out=dist)
    np.fill_diagonal(dist, 0.0)
    return dist


def _kernel(X: np.ndarray, eta: float, method: str) -> _Kernel:
    """Build the Gaussian kernel of the rows of ``X`` at the median bandwidth and factor it."""
    n = X.shape[0]
    dist = squared_distances(X)
    bandwidth = float(np.sqrt(np.median(dist[np.triu_indices(n, 1)])))
    if not bandwidth > 0:
        raise ValueError(
            f"{method}: the median distance between the rows of data is 0, so the Gaussian kernel "
            "of the score estimate has no bandwidth; more than half of the row pairs are equal"
        )
    matrix = np.exp(dist / (-2.0 * bandwidth**2))
    ridged = matrix.copy()
    ridged[np.diag_indices(n)] += eta
    return _Kernel(matrix, bandwidth, cho_factor(ridged, lower=True, overwrite_a=True))


def _score(X: np.ndarray, kernel: _Kernel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the score estimate with ``K X`` and the row sums of ``K``."""
    row = kernel.matrix.sum(axis=1)[:, None]
    KX = kernel.matrix @ X
    gradient = (row * X - KX) / kernel.bandwidth**2
    return -kernel.solve(gradient), KX, row


def score(X: np.ndarray, *, eta: float = ETA, method: str = "stein") -> np.ndarray:
    """Estimate the score of the density of ``X``'s rows at those rows.

    Parameters
    ----------
    X : np.ndarray
        ``(n, d)`` rows drawn from the density.
    eta : float, default :data:`ETA`
        Ridge term, positive.
    method : str, default "stein"
        The method a ``ValueError`` names.

    Returns
    -------
    np.ndarray
        ``(n, d)``; row ``i`` estimates ``grad log p(x_i)``.

    Raises
    ------
    ValueError
        If the median distance between the rows is 0.
    """
    return _score(X, _kernel(X, eta, method))[0]


def jacobian_diagonal(X: np.ndarray, *, eta: float = ETA, method: str = "stein") -> np.ndarray:
    """Estimate the diagonal of the score's Jacobian at the rows of ``X``.

    Parameters
    ----------
    X : np.ndarray
        ``(n, d)`` rows drawn from the density.
    eta : float, default :data:`ETA`
        Ridge term of both regressions, positive.
    method : str, default "stein"
        The method a ``ValueError`` names.

    Returns
    -------
    np.ndarray
        ``(n, d)``; entry ``[i, j]`` estimates the second derivative of ``log p`` in ``x[j]`` at
        ``x_i``.

    Raises
    ------
    ValueError
        If the median distance between the rows is 0.
    """
    kernel = _kernel(X, eta, method)
    G, KX, row = _score(X, kernel)
    s2 = kernel.bandwidth**2
    second = (row * X**2 - 2.0 * X * KX + kernel.matrix @ X**2) / s2**2 - row / s2
    return -(G**2) + kernel.solve(second)
