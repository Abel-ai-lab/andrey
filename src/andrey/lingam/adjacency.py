"""Weighted-adjacency estimation for LiNGAM: adaptive-Lasso regression on a fixed causal order.

Given a causal order, estimate ``B`` (``x_i = sum_j B[i, j] x_j``) by regressing each variable on
its predecessors with adaptive Lasso. Shared by the LiNGAM methods (DirectLiNGAM, each group of
MultiGroupDirectLiNGAM, the temporal LiNGAMs' pruning step).
"""

from __future__ import annotations

import numpy as np

from andrey.lingam.pwling import _COPY_TOL


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

    A vertex that its kept predecessors determine, so that a least-squares fit leaves at most
    ``_COPY_TOL`` of its variance (a copy of another column, up to scale), takes the
    least-squares weights, without the ones whose contribution is rounding-sized, and is left out
    of later regressions, where it would split its effect with the column it copies by rounding.
    Every other vertex is kept and takes :func:`predict_adaptive_lasso` weights, with ``refit``.

    ``forbidden[j, i]`` drops predecessor ``j`` of ``i``. ``required[j, i]`` makes ``j`` a
    predictor of ``i`` even when ``j`` was left out as a copy, and keeps its weight: it is not
    penalized (:func:`predict_with_required`, least-squares weights for every predictor) and not
    dropped as rounding-sized. The order must put each required parent before its child.
    """
    from sklearn.linear_model import LinearRegression

    d = X.shape[1]
    B = np.zeros((d, d), dtype=np.float64)
    kept = list(order[:1])
    for pos in range(1, len(order)):
        target = order[pos]
        forced = [] if required is None else [j for j in order[:pos] if required[j, target]]
        free = kept
        if forbidden is not None or forced:
            free = [
                j
                for j in kept
                if j not in forced and (forbidden is None or not forbidden[j, target])
            ]
        columns = [*forced, *free]
        if not columns:
            kept.append(target)
            continue
        y, P = X[:, target], X[:, columns]
        ols = LinearRegression().fit(P, y)
        residual = y - ols.intercept_ - P @ ols.coef_
        if np.var(residual) > _COPY_TOL * np.var(y):
            if forced:
                B[target, forced], B[target, free] = predict_with_required(X, forced, free, target)
            else:
                B[target, free] = _adaptive_lasso(X, free, target, ols.coef_, refit=refit)
            kept.append(target)
        else:
            coef = ols.coef_.copy()
            small = np.abs(coef) * P.std(axis=0) <= _COPY_TOL**0.5 * y.std()
            small[: len(forced)] = False
            coef[small] = 0.0
            B[target, columns] = coef
    return B
