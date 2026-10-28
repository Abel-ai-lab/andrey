"""Weighted-adjacency estimation for LiNGAM: adaptive-Lasso regression on a fixed causal order.

Given a causal order, estimate ``B`` (``x_i = sum_j B[i, j] x_j``) by regressing each variable on
its predecessors with adaptive Lasso. Shared by the LiNGAM methods (DirectLiNGAM, each group of
MultiGroupDirectLiNGAM, the temporal LiNGAMs' pruning step). The Lasso's BIC is taken over the whole
LARS path, down to the least-squares fit, so the selected predictors do not depend on the data's
unit (:func:`_lasso_bic`).
"""

from __future__ import annotations

import numpy as np

from andrey.core.warning_policy import convergence_as_search_limit
from andrey.lingam.pwling import _COPY_TOL

#: Above this standard deviation of the target, LARS's float32 tiny number (``2**-126``), which
#: it adds to its step denominators, can reach the steps' size, so the first fit is skipped.
_LARS_MAX_STD = 2.0**64
#: A refit divides the data so the target's standard deviation lies in ``[2**31, 2**32)``.
_LARS_STD_EXPONENT = 32
#: How a scikit-learn ConvergenceWarning from the LARS path names its source.
_LARS = "the adaptive Lasso's LARS path (scikit-learn's LassoLarsIC)"


def _lasso_bic(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Coefficients of ``y`` on ``Z`` at the BIC-best point of the whole Lasso path.

    The path is LARS's [1]_ on the adaptive Lasso's reweighted predictors [2]_, and BIC counts the
    nonzero coefficients as the degrees of freedom [3]_. scikit-learn's ``LassoLarsIC`` ends the
    path once the penalty falls to ``float32`` eps (``2**-23``) in the target's squared units, so a
    predictor whose share of the target's variance is below that never enters, although BIC on the
    whole path keeps it when the target is almost noiseless.

    The fit runs first on the data as given, unless the target's standard deviation is above
    ``_LARS_MAX_STD``. When that path reaches the least-squares fit (a final penalty of exactly
    ``0``), it is the answer. Otherwise the fit runs again on ``Z`` and ``y`` divided by the power
    of two (an exact division) that puts the target's standard deviation near ``2**32``, where both
    of LARS's absolute constants lie far below ``float64`` rounding. The first fit is kept when the
    second selects the same predictors, since BIC's choice then lies on the part of the path both
    computed, and the second fit is used otherwise.

    References
    ----------
    .. [1] Efron, B., Hastie, T., Johnstone, I., and Tibshirani, R. (2004). Least angle regression.
       Annals of Statistics 32(2), 407-499.
    .. [2] Zou, H. (2006). The adaptive lasso and its oracle properties. Journal of the American
       Statistical Association 101(476), 1418-1429.
    .. [3] Zou, H., Hastie, T., and Tibshirani, R. (2007). On the "degrees of freedom" of the
       lasso. Annals of Statistics 35(5), 2173-2192.
    """
    from sklearn.linear_model import LassoLarsIC

    std = float(np.std(y))
    first = None
    if std <= _LARS_MAX_STD:
        first = convergence_as_search_limit(_LARS, lambda: LassoLarsIC(criterion="bic").fit(Z, y))
        if first.alphas_[-1] == 0.0:
            return first.coef_
    unit = float(np.ldexp(1.0, np.frexp(std)[1] - _LARS_STD_EXPONENT))
    coef = convergence_as_search_limit(
        _LARS, lambda: LassoLarsIC(criterion="bic").fit(Z / unit, y / unit)
    ).coef_
    if first is not None and np.array_equal(coef != 0, first.coef_ != 0):
        return first.coef_
    return coef


def predict_adaptive_lasso(
    X: np.ndarray, predictors: list[int], target: int, gamma: float = 1.0, *, refit: bool = False
) -> np.ndarray:
    """Adaptive-Lasso effect of ``predictors`` on ``target`` (OLS weights -> a BIC Lasso).

    The Lasso runs on the predictors scaled by their least-squares weights, and BIC picks its
    point on the whole path (:func:`_lasso_bic`), so the same predictors are selected for the data
    in any unit. Returns the shrunk Lasso weights, or with ``refit=True`` an ordinary least-squares
    fit (with an intercept) of ``target`` on the selected predictors, zero for the rest.
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
    from sklearn.linear_model import LinearRegression

    weight = np.power(np.abs(ols_coef), gamma)
    lasso = _lasso_bic(X[:, predictors] * weight, X[:, target])
    if not refit:
        return lasso * weight
    selected = np.abs(lasso * weight) > 0.0
    coef = np.zeros(len(predictors), dtype=np.float64)
    if selected.any():
        columns = np.asarray(predictors)[selected]
        coef[selected] = LinearRegression().fit(X[:, columns], X[:, target]).coef_
    return coef


def predict_with_required(
    X: np.ndarray, required: list[int], predictors: list[int], target: int, gamma: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Effects on ``target`` of unpenalized ``required`` columns and Lasso-selected ``predictors``.

    The ``required`` columns are partialed out of ``target`` and of each of ``predictors`` by least
    squares (with an intercept), and the adaptive Lasso of :func:`predict_adaptive_lasso` selects
    among ``predictors`` on those residuals, so the required columns are never penalized. One
    least-squares fit of ``target`` on ``required`` and the selected predictors then estimates
    every coefficient: a required coefficient is estimated, not fixed, and can be ``0``. Returns
    the coefficients of ``required`` and of ``predictors``, zero for an unselected one.
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
