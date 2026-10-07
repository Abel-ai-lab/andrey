"""Weighted-adjacency estimation for LiNGAM: adaptive-Lasso regression on a fixed causal order.

Given a causal order, estimate ``B`` (``x_i = sum_j B[i, j] x_j``) by regressing each variable on
its predecessors with adaptive Lasso. Shared by the LiNGAM methods (DirectLiNGAM, each group of
MultiGroupDirectLiNGAM, the temporal LiNGAMs' pruning step).
"""

from __future__ import annotations

import numpy as np


def predict_adaptive_lasso(
    X: np.ndarray, predictors: list[int], target: int, gamma: float = 1.0
) -> np.ndarray:
    """Adaptive-Lasso effect of ``predictors`` on ``target`` (OLS weights -> a BIC LassoLarsIC)."""
    from sklearn.linear_model import LassoLarsIC, LinearRegression

    lr = LinearRegression()
    lr.fit(X[:, predictors], X[:, target])
    weight = np.power(np.abs(lr.coef_), gamma)
    reg = LassoLarsIC(criterion="bic")
    reg.fit(X[:, predictors] * weight, X[:, target])
    return reg.coef_ * weight


def estimate(X: np.ndarray, order: list[int]) -> np.ndarray:
    """Weighted adjacency ``B`` from a fixed order: each vertex regressed on its predecessors."""
    d = X.shape[1]
    B = np.zeros((d, d), dtype=np.float64)
    for pos in range(1, len(order)):
        target = order[pos]
        predictors = order[:pos]
        B[target, predictors] = predict_adaptive_lasso(X, predictors, target)
    return B
