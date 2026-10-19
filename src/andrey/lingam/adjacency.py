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


def predict_with_required(
    X: np.ndarray, required: list[int], predictors: list[int], target: int, gamma: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Effects on ``target`` of unpenalized ``required`` columns and Lasso-selected ``predictors``.

    The ``required`` columns are partialled out of ``target`` and of each of ``predictors`` by
    least squares (with an intercept); the adaptive Lasso of :func:`predict_adaptive_lasso` selects
    among ``predictors`` on those residuals, which by the Frisch-Waugh theorem is the selection
    with the required effects held fixed. One least-squares fit of ``target`` on ``required`` and
    the selected predictors then gives every coefficient. Returns the coefficients of ``required``
    and of ``predictors``, zero for an unselected one.
    """
    from sklearn.linear_model import LinearRegression

    coef = np.zeros(len(predictors), dtype=np.float64)
    if predictors:
        columns = [target, *predictors]
        fitted = LinearRegression().fit(X[:, required], X[:, columns])
        residuals = X[:, columns] - fitted.predict(X[:, required])
        lasso = predict_adaptive_lasso(residuals, list(range(1, len(columns))), 0, gamma)
        selected = np.abs(lasso) > 0.0
    else:
        selected = np.zeros(0, dtype=bool)
    chosen = [*required, *np.asarray(predictors, dtype=np.intp)[selected].tolist()]
    weights = LinearRegression().fit(X[:, chosen], X[:, target]).coef_
    coef[selected] = weights[len(required) :]
    return weights[: len(required)], coef


def estimate(
    X: np.ndarray,
    order: list[int],
    *,
    refit: bool = False,
    forbidden: np.ndarray | None = None,
    required: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted adjacency ``B`` from a fixed order: each vertex regressed on its predecessors.

    ``refit`` is passed to :func:`predict_adaptive_lasso`. ``forbidden[j, i]`` drops predecessor
    ``j`` of ``i``, and ``required[j, i]`` keeps it unpenalized (:func:`predict_with_required`),
    so its coefficient is nonzero; the order must put each required parent before its child.
    """
    d = X.shape[1]
    B = np.zeros((d, d), dtype=np.float64)
    for pos in range(1, len(order)):
        target = order[pos]
        predictors = order[:pos]
        if forbidden is None and required is None:
            B[target, predictors] = predict_adaptive_lasso(X, predictors, target, refit=refit)
            continue
        if forbidden is not None:
            predictors = [j for j in predictors if not forbidden[j, target]]
        forced = [] if required is None else [j for j in predictors if required[j, target]]
        if forced:
            free = [j for j in predictors if j not in forced]
            B[target, forced], B[target, free] = predict_with_required(X, forced, free, target)
        elif predictors:
            B[target, predictors] = predict_adaptive_lasso(X, predictors, target, refit=refit)
    return B
