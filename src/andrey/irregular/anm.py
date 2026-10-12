"""Additive noise model (ANM) direction test for a cause-effect pair.

Under ``effect = f(cause) + noise`` with the noise independent of the cause, a regression's
residual is independent of its input only in the causal direction. ANM fits a Gaussian process (a
scaled RBF kernel plus white noise, tuned by marginal likelihood) in each direction and tests the
residual against the input with :func:`~andrey.core.independence.hsic_statistic_test`. The
direction with the larger p-value is the inferred cause; on equal p-values, the one with the smaller
HSIC statistic relative to its mean under independence.

The regression is scikit-learn's ``GaussianProcessRegressor`` with its L-BFGS-B hyperparameter
search; only the gradient that search follows is computed here, from the inverse kernel matrix of
the Cholesky factor.

References
----------
Hoyer, Janzing, Mooij, Peters, Schölkopf. "Nonlinear causal discovery with additive noise
models." NIPS 21, 2008.

Rasmussen and Williams. "Gaussian Processes for Machine Learning." MIT Press, 2006, eq. 5.9.
"""

from __future__ import annotations

import warnings
from functools import partial

import numpy as np
from scipy.linalg import cho_solve, cholesky, get_lapack_funcs
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Kernel, WhiteKernel

from andrey.core.independence import hsic_statistic_test

#: ``GaussianProcessRegressor``'s default ``alpha``, added to the kernel matrix's diagonal.
_JITTER = 1e-10


def _kernel() -> Kernel:
    """``ConstantKernel * RBF + WhiteKernel``, with its starting values and bounds."""
    return ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.1, (1e-10, 1e1))


def _negative_log_marginal_likelihood(
    theta: np.ndarray, cause: np.ndarray, effect: np.ndarray
) -> tuple[float, np.ndarray]:
    """The objective of the hyperparameter search and its gradient, at ``theta = log(c, l, w)``.

    The kernel matrix is ``K = c * R + w * I`` with ``R = exp(-D / 2)`` and ``D`` the squared
    distances of ``cause / l``. The value is ``GaussianProcessRegressor.log_marginal_likelihood``,
    negated, operation for operation. Its gradient is ``0.5 * tr((a a^T - K^-1) dK)`` for each
    ``dK`` in ``(c * R, c * R * D, w * I)``, with ``a = K^-1 effect``: ``K^-1`` comes from the
    Cholesky factor (LAPACK ``potri``), where scikit-learn solves against the identity, and each
    trace is a sum over the elementwise product. The two agree up to rounding.

    The gradient's ``O(n^2)`` sums run in ``np.einsum``, outside BLAS: a multithreaded BLAS call
    this small spends most of its time waiting for the pool's threads when the cores are busy, and
    the search makes several per evaluation.
    """
    c, length, w = np.exp(theta[0]), np.exp(theta[1]), np.exp(theta[2])
    distances = pdist(cause / length, metric="sqeuclidean")
    rbf = squareform(np.exp(-0.5 * distances))
    np.fill_diagonal(rbf, 1)
    K = c * rbf
    K[np.diag_indices_from(K)] += w
    K[np.diag_indices_from(K)] += _JITTER
    try:
        L = cholesky(K, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return np.inf, np.zeros_like(theta)
    alpha = cho_solve((L, True), effect, check_finite=False)
    log_likelihood = -0.5 * np.einsum("ik,ik->k", effect, alpha)
    log_likelihood -= np.log(np.diag(L)).sum()
    log_likelihood -= K.shape[0] / 2 * np.log(2 * np.pi)

    (potri,) = get_lapack_funcs(("potri",), (L,))
    inverse, _ = potri(L, lower=1)  # K^-1 on and below the diagonal, zeros above
    a = alpha[:, 0]
    trace = np.trace(inverse)
    rbf_distances = rbf * squareform(distances)
    # For a symmetric M, sum(K^-1 * M) over the full matrix is 2 * sum(triangle * M) - the
    # diagonal's part; diag(R) is 1 and diag(R * D) is 0.
    triangle = inverse.T  # K^-1 on and above the diagonal, C-ordered like R
    triangle_rbf = np.einsum("ij,ij", triangle, rbf)
    triangle_rbf_distances = np.einsum("ij,ij", triangle, rbf_distances)
    gradient = 0.5 * np.array(
        [
            c * (np.einsum("i,ij,j", a, rbf, a) - 2 * triangle_rbf + trace),
            c * (np.einsum("i,ij,j", a, rbf_distances, a) - 2 * triangle_rbf_distances),
            w * (np.einsum("i,i", a, a) - trace),
        ]
    )
    return -log_likelihood.sum(axis=-1), -gradient


def _search(
    cause: np.ndarray, effect: np.ndarray, obj_func, initial_theta: np.ndarray, bounds: np.ndarray
) -> tuple[np.ndarray, float]:
    """``GaussianProcessRegressor``'s L-BFGS-B search on the objective computed here.

    Takes the arguments scikit-learn passes to a callable ``optimizer``; its own ``obj_func`` is
    left unused.
    """
    objective = partial(_negative_log_marginal_likelihood, cause=cause, effect=effect)
    result = minimize(objective, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds)
    return result.x, result.fun


def _regress_residual(cause: np.ndarray, effect: np.ndarray) -> np.ndarray:
    """Fit a Gaussian process ``effect ~ cause`` and return the ``(n, 1)`` residual column.

    The kernel is ``ConstantKernel * RBF + WhiteKernel``; the regressor optimizes it by maximizing
    the marginal likelihood, then the residual is ``effect`` minus the fitted prediction.
    """
    search = partial(_search, cause, effect)
    regressor = GaussianProcessRegressor(kernel=_kernel(), optimizer=search)
    with warnings.catch_warnings():
        # A hyperparameter reaches its bound when ``effect`` does not depend on ``cause``: the
        # signal variance at its floor is then the right fit, not a failed one.
        warnings.simplefilter("ignore", ConvergenceWarning)
        regressor.fit(cause, effect)
    prediction = regressor.predict(cause).reshape(-1, 1)
    return effect - prediction


def anm(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(pval_forward, pval_backward, stat_forward, stat_backward)`` for the pair ``x, y``.

    ``pval_forward`` is the HSIC independence p-value between ``x`` and the residual of the
    Gaussian process regressing ``y`` on ``x`` (the ``x -> y`` hypothesis); ``pval_backward`` is
    the same quantity with the roles swapped (the ``y -> x`` hypothesis). A larger p-value means
    the input and residual are more independent, so it favors that direction. ``stat_forward`` and
    ``stat_backward`` are the HSIC statistics of the same two tests, each divided by its mean under
    independence; a smaller value means weaker dependence. ``x`` and ``y`` are 1-D arrays of equal
    length.
    """
    # Contiguous, as the regressor stores its training data, so the objective's sums run alike.
    cause = np.ascontiguousarray(x, dtype=np.float64).reshape(-1, 1)
    effect = np.ascontiguousarray(y, dtype=np.float64).reshape(-1, 1)

    residual_effect = _regress_residual(cause, effect)
    stat_forward, pval_forward = hsic_statistic_test(cause, residual_effect)

    residual_cause = _regress_residual(effect, cause)
    stat_backward, pval_backward = hsic_statistic_test(effect, residual_cause)

    return pval_forward, pval_backward, stat_forward, stat_backward
