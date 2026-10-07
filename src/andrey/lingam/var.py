"""VAR-LiNGAM: structural vector-autoregression by non-Gaussianity (Hyvarinen et al. 2010).

Fits a structural VAR that separates the contemporaneous (instantaneous) causal structure from the
time-lagged influences of a linear, non-Gaussian process. The reduced-form VAR(``lags``) is fit by
ordinary least squares (no trend); DirectLiNGAM (:func:`andrey.lingam.direct.direct_lingam`)
recovers the instantaneous causal order and weights ``B0`` from the VAR residuals; the lagged
structural matrices are then ``M_tau = (I - B0) @ A_tau`` for each reduced-form lag matrix.

Deterministic, with a fixed lag order and no pruning. Reduced-form ordinary least squares
fits the lag coefficients without an intercept or trend.

References
----------
Hyvarinen, Zhang, Shimizu and Hoyer (2010). Estimation of a Structural Vector Autoregression
Model Using Non-Gaussianity. https://www.jmlr.org/papers/v11/hyvarinen10a.html
"""

from __future__ import annotations

import numpy as np

from andrey.lingam.direct import direct_lingam


def var_lingam(X: np.ndarray, *, lags: int = 1) -> tuple[list[int], np.ndarray]:
    """Fit VAR-LiNGAM: return ``(causal_order, adjacency_matrices)``.

    ``causal_order`` is the instantaneous DirectLiNGAM ordering. ``adjacency_matrices`` is the
    ``(lags + 1, d, d)`` structural stack: index 0 is the instantaneous ``B0`` (``x_i(t) =
    sum_j B0[i, j] x_j(t)``) and index ``tau`` in ``1..lags`` is ``M_tau = (I - B0) @ A_tau``,
    the structural effect of ``x(t - tau)`` on ``x(t)``.
    """
    if lags < 1:
        raise ValueError(f"lags must be >= 1, got {lags}")
    X = np.asarray(X, dtype=np.float64)
    ar_coefs, residuals = _estimate_var(X, lags)
    causal_order, b0 = direct_lingam(residuals)
    transform = np.eye(X.shape[1], dtype=np.float64) - b0
    adjacency_matrices = np.stack([b0, *(transform @ a_tau for a_tau in ar_coefs)])
    return causal_order, adjacency_matrices


def _estimate_var(X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit a reduced-form VAR(``lags``) by OLS with no trend; return ``(ar_coefs, residuals)``.

    ``ar_coefs`` has shape ``(lags, d, d)`` with ``ar_coefs[tau - 1]`` the lag-``tau`` matrix
    ``A_tau`` (``x(t) ~ sum_tau A_tau x(t - tau)``); ``residuals`` has shape ``(n_samples - lags,
    d)``, one row per fitted time step. Matches statsmodels ``VAR(X).fit(maxlags=lags,
    trend="n")``.
    """
    n_samples, d = X.shape
    if n_samples <= lags:
        raise ValueError(f"need more than {lags} samples to fit VAR({lags}), got {n_samples}")
    response = X[lags:]
    predictors = np.concatenate([X[lags - tau : n_samples - tau] for tau in range(1, lags + 1)], 1)
    weights, *_ = np.linalg.lstsq(predictors, response, rcond=None)
    residuals = response - predictors @ weights
    ar_coefs = np.stack([weights[tau * d : (tau + 1) * d].T for tau in range(lags)])
    return ar_coefs, residuals
