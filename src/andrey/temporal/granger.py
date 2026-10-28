"""Linear Granger causality by a cross-validated multivariate lasso VAR fit.

Stacks a time series into a lag-regression design (each column at time ``t`` regressed on all
columns at times ``t-1 .. t-maxlag``) and fits one cross-validated lasso per target variable. The
resulting coefficient matrix is the Granger lag graph: ``coeff[i, j + d*k]`` is the influence of
variable ``j`` at lag ``k+1`` on variable ``i`` at time ``t``, and its nonzero support is the
directed lag adjacency (a zero coefficient means no Granger cause at that lag).

Each target's fit is scikit-learn's ``LassoCV(cv=cv)`` with its defaults, computed step for step.
Every target shares the design, so the folds' centered designs and Gram matrices are computed once
for all of them. The targets run on the configured worker threads (the coordinate descent releases
the GIL), each with one BLAS thread.

References
----------
Granger (1969). Investigating Causal Relations by Econometric Models and Cross-spectral
Methods. Econometrica 37(3):424-438. https://doi.org/10.2307/1912791
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache

import numpy as np
from sklearn.linear_model import Lasso, lasso_path
from sklearn.model_selection import KFold
from threadpoolctl import ThreadpoolController

from andrey.core import backend
from andrey.core.units import least_squares

#: ``LassoCV`` defaults: penalties per grid, and the smallest penalty as a fraction of the largest.
_N_ALPHAS = 100
_EPS = 1e-3


@cache
def _blas() -> ThreadpoolController:
    """The loaded BLAS libraries, found once: the search reads every loaded library's path."""
    return ThreadpoolController()


@dataclass(frozen=True)
class _Fold:
    """One cross-validation split of the design, shared by every target."""

    train: np.ndarray
    test: np.ndarray
    centered: np.ndarray  # the training rows minus their column means
    offset: np.ndarray  # those column means
    gram: np.ndarray  # centered.T @ centered
    held_out: np.ndarray  # the test rows, uncentered


def _fold(design: np.ndarray, train: np.ndarray, test: np.ndarray) -> _Fold:
    """Center the training rows and form their Gram matrix as ``LassoCV`` does for each fold."""
    centered = design[train]
    offset = np.average(centered, axis=0)
    centered -= offset
    gram = np.empty((design.shape[1], design.shape[1]), order="C")
    np.dot(centered.T, centered, out=gram)
    for array in (centered, offset, gram):
        array.setflags(write=False)  # shared across worker threads
    return _Fold(train, test, centered, offset, gram, design[test])


def _alpha_grid(design: np.ndarray, column_mean: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One target's ``LassoCV`` penalty grid, geometric down from the least all-zero penalty."""
    centered_y = y - np.average(y, axis=0)
    xy = design.T @ centered_y - column_mean * np.sum(centered_y, axis=0)
    alpha_max = np.sqrt(np.max(xy**2)) / design.shape[0]
    resolution = np.finfo(np.float64).resolution
    if alpha_max <= resolution:  # y is constant: no penalty leaves a coefficient nonzero
        return np.full(_N_ALPHAS, resolution)
    return np.geomspace(alpha_max, alpha_max * _EPS, num=_N_ALPHAS)


def _fold_mse(fold: _Fold, y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Held-out mean squared error of the fold's lasso path at each penalty."""
    y_train = y[fold.train]
    y_offset = np.average(y_train, axis=0)
    y_train -= y_offset
    xy = np.empty(fold.gram.shape[0])
    np.dot(fold.centered.T, y_train, out=xy)
    _, coefs, _ = lasso_path(
        fold.centered,
        y_train,
        alphas=alphas,
        precompute=fold.gram,
        Xy=xy,
        copy_X=False,
        check_input=False,
    )
    coefs = coefs[np.newaxis, :, :]
    intercepts = np.atleast_1d(y_offset)[:, np.newaxis] - np.dot(fold.offset, coefs)
    residues = np.dot(fold.held_out, coefs) - y[fold.test][:, np.newaxis, np.newaxis]
    residues += intercepts
    return (residues**2).mean(axis=0).mean(axis=0)


def _fit_target(
    design: np.ndarray, column_mean: np.ndarray, folds: list[_Fold], y: np.ndarray
) -> np.ndarray:
    """``LassoCV`` for one target: the penalty of least mean held-out error, refit on all rows."""
    alphas = _alpha_grid(design, column_mean, y)
    paths = np.reshape([_fold_mse(fold, y, alphas) for fold in folds], (1, len(folds), -1))
    best = np.argmin(np.mean(paths, axis=1)[0])  # LassoCV's (1, folds, alphas) reduction
    return Lasso(alpha=alphas[best], precompute=False).fit(design, y).coef_


def _fit_known(
    design: np.ndarray, y: np.ndarray, cv: int, drop: np.ndarray, force: np.ndarray
) -> np.ndarray:
    """One target's coefficients when knowledge removes the ``drop`` columns and requires ``force``.

    The ``force`` columns are partialed out of ``y`` and of every free column by least squares
    (:func:`~andrey.core.units.least_squares`, with an intercept), so the lasso never penalizes
    them; the cross-validated lasso runs on the free columns' residuals with their own Gram
    matrices, and one least-squares fit of what the free columns leave of ``y`` on the ``force``
    columns gives their coefficients. A ``drop`` column has coefficient 0.
    """
    free = ~drop & ~force
    coefs = np.zeros(design.shape[1], dtype=np.float64)
    if not free.any() and not force.any():
        return coefs
    x = design[:, free]
    if force.any():
        forced = design[:, force]
        coef, intercept = least_squares(forced, np.column_stack([y, x]))
        residuals = np.column_stack([y, x]) - (forced @ coef.T + intercept)
        y_free, x = residuals[:, 0], residuals[:, 1:]
    else:
        y_free = y
    if free.any():
        folds = [_fold(x, train, test) for train, test in KFold(cv).split(x)]
        coefs[free] = _fit_target(x, np.average(x, axis=0), folds, np.ascontiguousarray(y_free))
    if force.any():
        left = y - design[:, free] @ coefs[free]
        coefs[force] = least_squares(design[:, force], left)[0]
    return coefs


def granger_lasso(
    data: np.ndarray,
    *,
    maxlag: int = 2,
    cv: int = 5,
    forbidden: np.ndarray | None = None,
    required: np.ndarray | None = None,
) -> np.ndarray:
    """Fit the lasso VAR and return the Granger lag-coefficient matrix.

    ``data`` is a ``(n_samples, d)`` time series (row = time step). Returns ``coeff`` of shape
    ``(d, d * maxlag)`` where ``coeff[i, j + d*k]`` is the influence of variable ``j`` at lag
    ``k + 1`` on variable ``i``; ``coeff != 0`` is the directed lag adjacency. Each target row is an
    independent ``cv``-fold cross-validated lasso regression on the stacked lagged design, equal to
    ``LassoCV(cv=cv)``. The rows are fitted on up to ``backend.worker_count()`` threads.

    With background knowledge, ``forbidden[k, j, i]`` leaves the lag-``k + 1`` regressor ``j`` out
    of target ``i``'s regression, and ``required[k, j, i]`` keeps it unpenalized
    (:func:`_fit_known`); both are boolean arrays of shape ``(maxlag, d, d)``. A target that
    knowledge leaves alone is fitted as without knowledge.
    """
    data = np.asarray(data, dtype=np.float64)
    n_samples, d = data.shape
    targets = data[maxlag:]
    design = np.hstack([data[maxlag - k : n_samples - k] for k in range(1, maxlag + 1)])
    design.setflags(write=False)
    column_mean = np.average(design, axis=0)
    folds = [_fold(design, train, test) for train, test in KFold(cv).split(design)]
    # Column (k - 1) * d + j of the design is variable j at lag k, so [:, :, i] flattens to it.
    drop = np.zeros((d, d * maxlag), dtype=bool) if forbidden is None else _by_target(forbidden)
    force = np.zeros((d, d * maxlag), dtype=bool) if required is None else _by_target(required)

    def fit(i: int) -> np.ndarray:
        y = np.ascontiguousarray(targets[:, i])
        if drop[i].any() or force[i].any():
            return _fit_known(design, y, cv, drop[i], force[i])
        return _fit_target(design, column_mean, folds, y)

    workers = min(backend.worker_count(), d)
    if workers <= 1:
        rows = [fit(i) for i in range(d)]
    else:  # one BLAS thread per worker thread, so they do not oversubscribe the cores
        with _blas().limit(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(fit, range(d)))
    return np.asarray(rows, dtype=np.float64).reshape(d, d * maxlag)


def _by_target(links: np.ndarray) -> np.ndarray:
    """``links[k, j, i]`` (lag ``k + 1``, cause ``j``, effect ``i``) as one row per effect ``i`` of
    design columns ``k * d + j``."""
    lags, d, _ = links.shape
    return np.asarray(links, dtype=bool).transpose(2, 0, 1).reshape(d, lags * d)
