"""Weighted-adjacency estimation for LiNGAM: adaptive-Lasso regression on a fixed causal order.

Given a causal order, estimate ``B`` (``x_i = sum_j B[i, j] x_j``) by regressing each variable on
its predecessors with adaptive Lasso. Shared by the LiNGAM methods (DirectLiNGAM, each group of
MultiGroupDirectLiNGAM, the temporal LiNGAMs' pruning step).
"""

from __future__ import annotations

import numpy as np


def predict_adaptive_lasso(
    X: np.ndarray, predictors: list[int], target: int, gamma: float = 1.0, *, refit: bool = False
) -> np.ndarray:
    """Adaptive-Lasso effect of ``predictors`` on ``target`` (OLS weights -> a BIC LassoLarsIC).

    Returns the shrunk Lasso weights, or with ``refit=True`` an ordinary least-squares fit (with an
    intercept) of ``target`` on the selected predictors, zero for the rest.
    """
    from sklearn.linear_model import LassoLarsIC, LinearRegression

    lr = LinearRegression()
    lr.fit(X[:, predictors], X[:, target])
    weight = np.power(np.abs(lr.coef_), gamma)
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

    ``refit`` is passed to :func:`predict_adaptive_lasso`.
    """
    d = X.shape[1]
    B = np.zeros((d, d), dtype=np.float64)
    for pos in range(1, len(order)):
        target = order[pos]
        predictors = order[:pos]
        B[target, predictors] = predict_adaptive_lasso(X, predictors, target, refit=refit)
    return B
