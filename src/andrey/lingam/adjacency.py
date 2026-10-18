"""Weighted-adjacency estimation for LiNGAM: adaptive-Lasso regression on a fixed causal order.

Given a causal order, estimate ``B`` (``x_i = sum_j B[i, j] x_j``) by regressing each variable on
its predecessors with adaptive Lasso. Shared by the LiNGAM methods (DirectLiNGAM, each group of
MultiGroupDirectLiNGAM, the temporal LiNGAMs' pruning step).
"""

from __future__ import annotations

import numpy as np

from andrey.lingam.pwling import _DEGENERATE_TOL


def predict_adaptive_lasso(
    X: np.ndarray, predictors: list[int], target: int, gamma: float = 1.0, *, refit: bool = False
) -> np.ndarray:
    """Adaptive-Lasso effect of ``predictors`` on ``target`` (OLS weights -> a BIC LassoLarsIC).

    Returns the shrunk Lasso weights, or with ``refit=True`` an ordinary least-squares fit (with an
    intercept) of ``target`` on the selected predictors, zero for the rest.
    """
    from sklearn.linear_model import LinearRegression

    ols = LinearRegression().fit(X[:, predictors], X[:, target])
    return _adaptive_lasso(X, predictors, target, ols.coef_, gamma, refit=refit)


def _adaptive_lasso(
    X: np.ndarray,
    predictors: list[int],
    target: int,
    ols_coef: np.ndarray,
    gamma: float = 1.0,
    *,
    refit: bool = False,
) -> np.ndarray:
    """:func:`predict_adaptive_lasso` from the least-squares weights ``ols_coef``."""
    from sklearn.linear_model import LassoLarsIC, LinearRegression

    weight = np.power(np.abs(ols_coef), gamma)
    reg = LassoLarsIC(criterion="bic")
    reg.fit(X[:, predictors] * weight, X[:, target])
    if not refit:
        return reg.coef_ * weight
    selected = np.abs(reg.coef_ * weight) > 0.0
    coef = np.zeros(len(predictors), dtype=np.float64)
    if selected.any():
        columns = np.asarray(predictors)[selected]
        coef[selected] = LinearRegression().fit(X[:, columns], X[:, target]).coef_
    return coef


def estimate(X: np.ndarray, order: list[int], *, refit: bool = False) -> np.ndarray:
    """Weighted adjacency ``B`` from a fixed order: each vertex regressed on its predecessors.

    A vertex that its kept predecessors determine, so that a least-squares fit leaves at most
    ``_DEGENERATE_TOL`` of its variance (a copy of another column, up to scale), takes the
    least-squares weights, without the ones whose contribution is rounding-sized, and is left out
    of later regressions, where it would split its effect with the column it copies by rounding.
    Every other vertex is kept and takes :func:`predict_adaptive_lasso` weights, with ``refit``.
    """
    from sklearn.linear_model import LinearRegression

    d = X.shape[1]
    B = np.zeros((d, d), dtype=np.float64)
    kept = list(order[:1])
    for target in order[1:]:
        y, P = X[:, target], X[:, kept]
        ols = LinearRegression().fit(P, y)
        residual = y - ols.intercept_ - P @ ols.coef_
        if np.var(residual) > _DEGENERATE_TOL * np.var(y):
            B[target, kept] = _adaptive_lasso(X, kept, target, ols.coef_, refit=refit)
            kept.append(target)
        else:
            coef = ols.coef_.copy()
            coef[np.abs(coef) * P.std(axis=0) <= _DEGENERATE_TOL**0.5 * y.std()] = 0.0
            B[target, kept] = coef
    return B
