"""Post-nonlinear (PNL) direction test, the engine behind :func:`andrey.pnl`.

The algorithm, its settings, and its references are described on :func:`andrey.pnl`.

In each direction the fit finds an inner transform ``G1`` of the assumed cause and an invertible
outer transform ``G2`` of the assumed effect that make ``e = G2(effect) - G1(cause)`` independent
of the cause, by maximizing ``-0.5 * sum(e**2) + sum(log|G2'(effect)|)``. Both transforms are sums
of fixed smooth basis functions, so ``G1`` is profiled out by least squares, and a damped Newton
iteration with a log barrier, which keeps ``G2`` increasing, reaches the global optimum of a
convex program. Each variable is first mapped to normal scores by rank.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata

from andrey.core.independence import hsic_test

_N_FEATURES = 10
_RIDGE = 1e-8
_BARRIER_FLOOR = 1e-6
_MAX_NEWTON = 60
_TOL = 1e-8


def _gaussianize(values: np.ndarray) -> np.ndarray:
    """Return the rank-based normal-score transform of ``values`` (marginal mapped to Gaussian).

    Each value is replaced by the standard-normal quantile of its empirical rank. This monotone
    per-variable warp partly undoes the outer nonlinearity, so the fitted outer transform only has
    to correct the residual distortion rather than represent a steep inverse from scratch; it
    matches the "make each variable close to Gaussian" preprocessing the post-nonlinear model
    prescribes. Applied identically to both variables, it does not bias the inferred direction.
    """
    n = values.shape[0]
    if n == 0:
        return values.astype(np.float64)
    ranks = rankdata(values, method="average")
    return norm.ppf((ranks - 0.5) / n)


def _rbf_centers(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Return evenly spread Gaussian-feature centers and a shared bandwidth for ``values``.

    Centers span the 2nd-to-98th percentile range so extreme points do not stretch the grid; the
    bandwidth is the center spacing, giving smoothly overlapping features.
    """
    lo, hi = np.percentile(values, [2.0, 98.0])
    if hi - lo < 1e-9:
        lo, hi = values.min() - 1.0, values.max() + 1.0
    centers = np.linspace(lo, hi, _N_FEATURES)
    bandwidth = (hi - lo) / max(_N_FEATURES - 1, 1)
    return centers, max(bandwidth, 1e-6)


def _design(values: np.ndarray, *, intercept: bool) -> np.ndarray:
    """Return the basis design matrix for ``values``: a linear column plus Gaussian radial features.

    With ``intercept`` an all-ones column leads, absorbing an additive offset (used for the inner
    transform so the outer transform's constant is unidentified and can be dropped).
    """
    centers, bandwidth = _rbf_centers(values)
    gaussians = np.exp(-0.5 * ((values[:, None] - centers[None, :]) / bandwidth) ** 2)
    columns = [values[:, None], gaussians]
    if intercept:
        columns.insert(0, np.ones((values.shape[0], 1)))
    return np.hstack(columns)


def _design_derivative(values: np.ndarray) -> np.ndarray:
    """Return the derivative of the no-intercept design of ``values`` column by column.

    The linear column differentiates to ones; each Gaussian feature to
    ``-(v - c) / bandwidth**2 * feature``. Used for the outer transform's Jacobian term.
    """
    centers, bandwidth = _rbf_centers(values)
    offset = (values[:, None] - centers[None, :]) / bandwidth
    gaussians = np.exp(-0.5 * offset**2)
    gauss_deriv = -(offset / bandwidth) * gaussians
    return np.hstack([np.ones((values.shape[0], 1)), gauss_deriv])


def _fit_disturbance(cause: np.ndarray, effect: np.ndarray) -> np.ndarray:
    """Recover the disturbance ``e = G2(effect) - G1(cause)`` under the ``cause -> effect`` model.

    ``G1`` is profiled out by regressing the outer transform on the cause's basis, so the objective
    ``0.5 * ||(I - P) G2(effect)||**2 - sum(log G2'(effect))`` is convex in ``G2``'s coefficients;
    a damped Newton iteration started at the identity transform drives it to the global optimum
    while a backtracking line search keeps ``G2'`` positive on every sample.
    """
    phi = _design(cause, intercept=True)
    outer = _design(effect, intercept=False)
    outer_deriv = _design_derivative(effect)

    # Residualize the outer basis against the inner basis (profiles out G1 by least squares).
    projection, *_ = np.linalg.lstsq(phi, outer, rcond=None)
    residual_basis = outer - phi @ projection
    gram = residual_basis.T @ residual_basis
    gram += _RIDGE * np.eye(gram.shape[0])

    n_params = outer.shape[1]
    weights = np.zeros(n_params)
    weights[0] = 1.0  # identity outer transform: derivative is one everywhere, strictly feasible.

    def objective(w: np.ndarray) -> float:
        slopes = outer_deriv @ w
        if np.any(slopes <= _BARRIER_FLOOR):
            return np.inf
        return 0.5 * float(w @ gram @ w) - float(np.sum(np.log(slopes)))

    value = objective(weights)
    for _ in range(_MAX_NEWTON):
        slopes = outer_deriv @ weights
        inv_slopes = 1.0 / slopes
        gradient = gram @ weights - outer_deriv.T @ inv_slopes
        hessian = gram + outer_deriv.T @ (outer_deriv * (inv_slopes**2)[:, None])
        step = np.linalg.solve(hessian + _RIDGE * np.eye(n_params), -gradient)
        if np.max(np.abs(step)) < _TOL:
            break
        alpha = 1.0
        while alpha > 1e-10:
            candidate = weights + alpha * step
            candidate_value = objective(candidate)
            if candidate_value < value - 1e-4 * alpha * float(gradient @ step):
                break
            alpha *= 0.5
        if alpha <= 1e-10:
            break
        weights, value = candidate, candidate_value

    return residual_basis @ weights


def pnl(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return ``(pval_forward, pval_backward)`` for the PNL direction test on the pair ``x, y``.

    ``pval_forward`` is the HSIC independence p-value between ``x`` and the disturbance recovered
    under the ``x -> y`` post-nonlinear model; ``pval_backward`` is the same quantity under the
    ``y -> x`` model. A larger p-value means the assumed cause and the recovered disturbance are
    more independent, so it favors that direction. ``x`` and ``y`` are 1-D arrays of equal length.
    """
    cause = _gaussianize(np.asarray(x, dtype=np.float64).ravel())
    effect = _gaussianize(np.asarray(y, dtype=np.float64).ravel())

    disturbance_forward = _fit_disturbance(cause, effect)
    pval_forward = hsic_test(cause, disturbance_forward)

    disturbance_backward = _fit_disturbance(effect, cause)
    pval_backward = hsic_test(effect, disturbance_backward)

    return pval_forward, pval_backward
