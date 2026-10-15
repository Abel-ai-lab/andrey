"""Longitudinal LiNGAM: linear non-Gaussian causal discovery across repeated time points.

Recovers, from panel data observed at ``T`` time points (each an ``(n_samples, n_features)`` array
of the *same* units), both the instantaneous causal structure within a time point and the lagged
effects from earlier time points. At each time point ``t`` the observed variables are first
regressed on their lagged values ``X(t-1) .. X(t-n_lags)`` to remove the predictable component; the
non-Gaussian residuals ``N(t)`` are handed to DirectLiNGAM to estimate the instantaneous weighted
adjacency ``B(t,t)`` (``x_i = sum_j B[i, j] x_j``) and its causal order, and the lagged adjacencies
``B(t,t-tau) = (I - B(t,t)) M(t,t-tau)`` are recovered from the regression coefficients. The full
procedure is deterministic (DirectLiNGAM's ``pwling`` measure).

Kadowaki, Shimizu & Washio, "Estimation of causal structures in longitudinal data using
non-Gaussianity" (MLSP 2013).
"""

from __future__ import annotations

import numpy as np

from andrey.core.independence import hsic_gamma_test
from andrey.lingam.direct import direct_lingam


def _check_panel(X_list: list[np.ndarray]) -> tuple[list[np.ndarray], int, int, int]:
    """Validate the panel and return ``(transposed, T, n_samples, n_features)``.

    Each returned array is a single time point transposed to ``(n_features, n_samples)``.
    """
    first = np.asarray(X_list[0], dtype=np.float64)
    n_samples, n_features = first.shape
    transposed = []
    for X in X_list:
        X = np.asarray(X, dtype=np.float64)
        if X.shape != (n_samples, n_features):
            raise ValueError("all time points must share the same shape")
        transposed.append(X.T)
    return transposed, len(X_list), n_samples, n_features


def _regress_out_lags(X_predictors: np.ndarray, X_target: np.ndarray) -> np.ndarray:
    """Ordinary least squares of each target column on ``X_predictors`` (intercept via centering).

    Returns coefficients of shape ``(n_targets, n_predictors)``; the intercept is absorbed by
    mean-centering both sides, matching a fit-intercept linear regression.
    """
    n_targets = X_target.shape[1]
    n_predictors = X_predictors.shape[1]
    coefs = np.empty((n_targets, n_predictors), dtype=np.float64)
    Xc = X_predictors - X_predictors.mean(axis=0)
    for i in range(n_targets):
        y = X_target[:, i] - X_target[:, i].mean()
        coef, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        coefs[i] = coef
    return coefs


def _compute_residuals(
    X_t: list[np.ndarray], T: int, n: int, p: int, n_lags: int
) -> tuple[np.ndarray, np.ndarray]:
    """Regress each time point on its available lags; return lag coefficients ``M`` and residuals.

    ``M`` has shape ``(T, n_lags, p, p)`` with ``M[t, tau]`` the effect of ``X(t-tau-1)`` on
    ``X(t)``. ``N`` has shape ``(T, p, n)``; ``N[t]`` is the residual after removing all lags.
    Time point 0 has no predecessors, so ``N[0]`` is left as ``NaN``. Occasion ``t`` has only ``t``
    predecessors, so a lag ``tau + 1 > t`` reaching before the panel start is excluded from the
    regression -- never wrapped (negative indexing) into a future occasion, which would leak future
    data into the retained coefficients of the valid lags. Its ``M[t, tau]`` stays 0.
    """
    M_tau = np.zeros((T, n_lags, p, p))
    N_t = np.full((T, p, n), np.nan)
    for t in range(1, T):
        n_avail = min(n_lags, t)  # only lags with real history; never index before the panel start
        X_predictors = np.empty((n, p * n_avail))
        for tau in range(n_avail):
            X_predictors[:, p * tau : p * tau + p] = X_t[t - (tau + 1)].T
        coefs = _regress_out_lags(X_predictors, X_t[t].T)
        residual = X_t[t].copy()
        for tau in range(n_avail):
            M_tau[t, tau] = coefs[:, p * tau : p * tau + p]
            residual = residual - M_tau[t, tau] @ X_t[t - (tau + 1)]
        N_t[t] = residual
    return M_tau, N_t


def _estimate_instantaneous(
    N_t: np.ndarray, T: int, p: int, measure: str
) -> tuple[np.ndarray, list[list[int]]]:
    """Apply DirectLiNGAM to each residual set; return ``B(t,t)`` and per-time causal orders."""
    B_t = np.zeros((T, p, p))
    causal_orders: list[list[int]] = [[]]
    for t in range(1, T):
        order, B = direct_lingam(N_t[t].T, measure=measure)
        causal_orders.append(order)
        B_t[t] = B
    return B_t, causal_orders


def _estimate_lagged(B_t: np.ndarray, M_tau: np.ndarray, T: int, p: int, n_lags: int) -> np.ndarray:
    """Convert lag regression coefficients to lagged adjacencies ``(I - B(t,t)) M(t,t-tau)``."""
    B_tau = np.zeros((T, n_lags, p, p))
    eye = np.eye(p)
    for t in range(T):
        for tau in range(n_lags):
            B_tau[t, tau] = (eye - B_t[t]) @ M_tau[t, tau]
    return B_tau


class LongitudinalLiNGAM:
    """Longitudinal LiNGAM estimator for panel / repeated-measures data.

    Fit with :meth:`fit` on a list of ``(n_samples, n_features)`` arrays, one per time point (the
    same units measured repeatedly). After fitting, :attr:`adjacency_matrices_` holds the
    instantaneous and lagged weighted adjacencies and :attr:`causal_orders_` the per-time causal
    orders.
    """

    def __init__(self, n_lags: int = 1, measure: str = "pwling") -> None:
        """Configure the number of lags and the DirectLiNGAM independence ``measure``."""
        self.n_lags = n_lags
        self.measure = measure
        self._T = 0
        self._n = 0
        self._p = 0
        self._adjacency_matrices: np.ndarray | None = None
        self._causal_orders: list[list[int]] | None = None
        self._residuals: np.ndarray | None = None

    def fit(self, X_list: list[np.ndarray]) -> LongitudinalLiNGAM:
        """Estimate instantaneous and lagged structure from the panel; return ``self``."""
        X_t, self._T, self._n, self._p = _check_panel(X_list)
        M_tau, N_t = _compute_residuals(X_t, self._T, self._n, self._p, self.n_lags)
        B_t, causal_orders = _estimate_instantaneous(N_t, self._T, self._p, self.measure)
        B_tau = _estimate_lagged(B_t, M_tau, self._T, self._p, self.n_lags)

        adjacency = np.full((self._T, 1 + self.n_lags, self._p, self._p), np.nan)
        for t in range(1, self._T):
            adjacency[t, 0] = B_t[t]
            for lag in range(self.n_lags):
                if lag >= t:  # the lag-(lag+1) block reaches before the panel start -> uncomputable
                    continue
                adjacency[t, lag + 1] = B_tau[t, lag]
        self._adjacency_matrices = adjacency
        self._causal_orders = causal_orders

        residuals = np.zeros((self._T, self._n, self._p))
        for t in range(self._T):
            residuals[t] = N_t[t].T
        self._residuals = residuals
        return self

    def get_error_independence_p_values(self) -> np.ndarray:
        """Pairwise HSIC p-values between instantaneous error variables, per time point.

        Returns an array of shape ``(T, p, p)``; entry ``[t, i, j]`` is the gamma-HSIC p-value of
        independence between errors ``i`` and ``j`` at time ``t``. Time point 0 is ``NaN``. A larger
        p-value means more independent.
        """
        if self._adjacency_matrices is None or self._residuals is None:
            raise ValueError("call fit before get_error_independence_p_values")
        p = self._p
        p_values = np.full((self._T, p, p), np.nan)
        eye = np.eye(p)
        for t in range(1, self._T):
            B_t = self._adjacency_matrices[t, 0]
            errors = ((eye - B_t) @ self._residuals[t].T).T
            table = np.zeros((p, p))
            for i in range(p):
                for j in range(i + 1, p):
                    _, pval = hsic_gamma_test(errors[:, i], errors[:, j])
                    table[i, j] = pval
                    table[j, i] = pval
            p_values[t] = table
        return p_values

    @property
    def adjacency_matrices_(self) -> np.ndarray:
        """Adjacencies of shape ``(T, 1 + n_lags, p, p)``.

        ``[t, 0]`` is the instantaneous ``B(t,t)`` and ``[t, lag + 1]`` the lagged ``B(t,t-lag-1)``,
        with ``x_i = sum_j B[i, j] x_j``. Uncomputable entries (such as time point 0) are ``NaN``.
        """
        if self._adjacency_matrices is None:
            raise ValueError("call fit before accessing adjacency_matrices_")
        return self._adjacency_matrices

    @property
    def causal_orders_(self) -> list[list[int]]:
        """Per-time-point instantaneous causal orders (time point 0 is an empty placeholder)."""
        if self._causal_orders is None:
            raise ValueError("call fit before accessing causal_orders_")
        return self._causal_orders

    @property
    def residuals_(self) -> np.ndarray:
        """Lag-removed residuals of shape ``(T, n_samples, n_features)``."""
        if self._residuals is None:
            raise ValueError("call fit before accessing residuals_")
        return self._residuals


def longitudinal_lingam(
    X_list: list[np.ndarray], *, n_lags: int = 1, measure: str = "pwling"
) -> tuple[np.ndarray, list[list[int]]]:
    """Fit Longitudinal LiNGAM and return ``(adjacency_matrices, causal_orders)``.

    ``adjacency_matrices`` has shape ``(T, 1 + n_lags, p, p)`` (see
    :attr:`LongitudinalLiNGAM.adjacency_matrices_`); ``causal_orders`` lists the instantaneous
    causal order per time point.
    """
    model = LongitudinalLiNGAM(n_lags=n_lags, measure=measure).fit(X_list)
    return model.adjacency_matrices_, model.causal_orders_
