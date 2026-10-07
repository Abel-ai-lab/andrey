"""Additive noise model (ANM) direction test for a cause-effect pair.

Under ``effect = f(cause) + noise`` with the noise independent of the cause, a regression's
residual is independent of its input only in the causal direction. ANM fits a Gaussian process (a
scaled RBF kernel plus white noise, tuned by marginal likelihood) in each direction and tests the
residual against the input with :func:`~andrey.core.independence.hsic_test`. The direction with
the larger p-value is the inferred cause.

References
----------
Hoyer, Janzing, Mooij, Peters, Schölkopf. "Nonlinear causal discovery with additive noise
models." NIPS 21, 2008.
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from andrey.core.independence import hsic_test


def _regress_residual(cause: np.ndarray, effect: np.ndarray) -> np.ndarray:
    """Fit a Gaussian process ``effect ~ cause`` and return the ``(n, 1)`` residual column.

    The kernel is ``ConstantKernel * RBF + WhiteKernel``; the regressor optimizes it by maximizing
    the marginal likelihood, then the residual is ``effect`` minus the fitted prediction.
    """
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(
        0.1, (1e-10, 1e1)
    )
    regressor = GaussianProcessRegressor(kernel=kernel)
    regressor.fit(cause, effect)
    prediction = regressor.predict(cause).reshape(-1, 1)
    return effect - prediction


def anm(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return ``(pval_forward, pval_backward)`` for the ANM direction test on the pair ``x, y``.

    ``pval_forward`` is the HSIC independence p-value between ``x`` and the residual of the
    Gaussian process regressing ``y`` on ``x`` (the ``x -> y`` hypothesis); ``pval_backward`` is
    the same quantity with the roles swapped (the ``y -> x`` hypothesis). A larger p-value means
    the input and residual are more independent, so it favors that direction. ``x`` and ``y`` are
    1-D arrays of equal length.
    """
    cause = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    effect = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    residual_effect = _regress_residual(cause, effect)
    pval_forward = hsic_test(cause, residual_effect)

    residual_cause = _regress_residual(effect, cause)
    pval_backward = hsic_test(effect, residual_cause)

    return pval_forward, pval_backward
