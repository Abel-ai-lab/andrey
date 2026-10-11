"""Linear Granger causality by a cross-validated multivariate lasso VAR fit.

Stacks a time series into a lag-regression design (each column at time ``t`` regressed on all
columns at times ``t-1 .. t-maxlag``) and fits one cross-validated lasso per target variable. The
resulting coefficient matrix is the Granger lag graph: ``coeff[i, j + d*k]`` is the influence of
variable ``j`` at lag ``k+1`` on variable ``i`` at time ``t``, and its nonzero support is the
directed lag adjacency (a zero coefficient means no Granger cause at that lag).

Each target's fit is scikit-learn's ``LassoCV(cv=cv)`` with its defaults, computed step for step.
Every target shares the design, so the folds' centered designs and Gram matrices are computed once
for all of them; the targets run on the configured worker threads (the coordinate descent releases
the GIL), and BLAS is pinned to one thread so the result does not depend on the thread count.

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


def granger_lasso(data: np.ndarray, *, maxlag: int = 2, cv: int = 5) -> np.ndarray:
    """Fit the lasso VAR and return the Granger lag-coefficient matrix.

    ``data`` is a ``(n_samples, d)`` time series (row = time step). Returns ``coeff`` of shape
    ``(d, d * maxlag)`` where ``coeff[i, j + d*k]`` is the influence of variable ``j`` at lag
    ``k + 1`` on variable ``i``; ``coeff != 0`` is the directed lag adjacency. Each target row is an
    independent ``cv``-fold cross-validated lasso regression on the stacked lagged design, equal to
    ``LassoCV(cv=cv)``. The rows are fitted on up to ``backend.worker_count()`` threads.
    """
    data = np.asarray(data, dtype=np.float64)
    n_samples, d = data.shape
    targets = data[maxlag:]
    design = np.hstack([data[maxlag - k : n_samples - k] for k in range(1, maxlag + 1)])
    design.setflags(write=False)
    workers = min(backend.worker_count(), d)
    with _blas().limit(limits=1):
        column_mean = np.average(design, axis=0)
        folds = [_fold(design, train, test) for train, test in KFold(cv).split(design)]

        def fit(i: int) -> np.ndarray:
            return _fit_target(design, column_mean, folds, np.ascontiguousarray(targets[:, i]))

        if workers <= 1:
            rows = [fit(i) for i in range(d)]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(fit, range(d)))
    return np.asarray(rows, dtype=np.float64).reshape(d, d * maxlag)
