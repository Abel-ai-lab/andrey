"""Linear Granger causality by a t-test (the F-test of one coefficient) on a vector autoregression.

Stacks a time series into a lag-regression design (each column at time ``t`` regressed on all
columns at times ``t-1 .. t-maxlag``) and fits one ordinary least-squares regression per target
variable. The test of each lagged coefficient is the F-test of the target's regression with and
without that regressor, which is the square of the coefficient's t-ratio: ``p[i, j + d*k]`` is the
p-value of variable ``j`` at lag ``k+1`` given every other lagged variable, the coefficient p-values
of a vector autoregression. A link is significant when ``p <= alpha``; with ``fdr_method="fdr_bh"``
the p-values of all the links are first replaced by their Benjamini-Hochberg adjusted values
(:mod:`andrey.core.fdr`).

References
----------
Granger (1969). Investigating Causal Relations by Econometric Models and Cross-spectral
Methods. Econometrica 37(3):424-438. https://doi.org/10.2307/1912791

Lutkepohl (2005). New Introduction to Multiple Time Series Analysis. Springer.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.linalg import solve_triangular

from andrey.core.fdr import check_fdr_method, corrected_pvalues


def _ols(design: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares coefficients and two-sided p-values of ``targets`` on ``design``.

    ``design`` carries the intercept in column 0 and has shape ``(m, p)``; ``targets`` has shape
    ``(m, k)``. Returns two ``(p - 1, k)`` arrays without the intercept row: the coefficients and
    their p-values from the Student ``t`` distribution with ``m - p`` degrees of freedom. Raises
    ``ValueError`` when the design is singular.
    """
    m, p = design.shape
    q, r = np.linalg.qr(design)
    singular_values = np.linalg.svd(r, compute_uv=False)
    if singular_values[-1] <= singular_values[0] * p * np.finfo(np.float64).eps:
        raise ValueError(
            "granger: the lagged regression is singular (some lagged variables are linear "
            "combinations of others); drop or merge the redundant columns"
        )
    coef = solve_triangular(r, q.T @ targets)
    r_inverse = solve_triangular(r, np.eye(p))
    df = m - p
    variance = np.sum((targets - design @ coef) ** 2, axis=0) / df
    scale = np.sqrt(np.sum(r_inverse**2, axis=1)[:, np.newaxis] * variance[np.newaxis, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ratio = coef / scale
    p_values = 2 * stats.t.sf(np.abs(t_ratio), df)
    # A perfect fit leaves no noise: a nonzero coefficient is then certain (p = 0, from t = inf),
    # and a zero one says nothing (0 / 0).
    p_values = np.where(np.isnan(p_values), 1.0, p_values)
    return coef[1:], p_values[1:]


def granger_f_test(
    data: np.ndarray,
    *,
    maxlag: int,
    alpha: float = 0.05,
    fdr_method: str = "none",
    forbidden: np.ndarray | None = None,
    required: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the vector autoregression and return the Granger lag coefficients and p-values.

    ``data`` is a ``(n_samples, d)`` time series (row = time step). Returns ``(coeff, p_values)``,
    both of shape ``(d, d * maxlag)``, where entry ``[i, j + d*k]`` belongs to variable ``j`` at lag
    ``k + 1`` and target ``i``. ``p_values`` holds the p-value of the test that the coefficient is
    zero in the target's regression on every other lagged variable; ``coeff`` holds the
    least-squares coefficient where ``p_values <= alpha`` and ``0`` elsewhere, so ``coeff != 0`` is
    the directed lag adjacency. Each target row is an independent regression, so the rows are fitted
    together for the targets that share a design.

    ``fdr_method`` corrects the p-values of all the links together before ``alpha`` decides which
    are significant (:func:`~andrey.core.fdr.corrected_pvalues`): ``"none"`` leaves them as tested,
    and ``"fdr_bh"`` returns their Benjamini-Hochberg adjusted values, and keeps the links whose
    adjusted value is at most ``alpha``.

    With background knowledge, ``forbidden[k, j, i]`` leaves the lag-``k + 1`` regressor ``j`` out
    of target ``i``'s regression (coefficient ``0``, p-value ``1``), and ``required[k, j, i]`` keeps
    it in the graph whatever its test says (p-value ``0``); both are boolean arrays of shape
    ``(maxlag, d, d)``. The other regressors are tested given the ones that stay in the regression.
    A forbidden link is no test: the correction does not count it, and it keeps p-value ``1``; a
    required link is counted with p-value ``0``.
    """
    check_fdr_method(fdr_method)
    data = np.asarray(data, dtype=np.float64)
    n_samples, d = data.shape
    targets = data[maxlag:]
    # Column (k - 1) * d + j + 1 of the design is variable j at lag k; column 0 is the intercept.
    design = np.hstack(
        [np.ones((n_samples - maxlag, 1))]
        + [data[maxlag - k : n_samples - k] for k in range(1, maxlag + 1)]
    )
    drop = np.zeros((d, d * maxlag), dtype=bool) if forbidden is None else _by_target(forbidden)
    force = np.zeros((d, d * maxlag), dtype=bool) if required is None else _by_target(required)
    coeff = np.zeros((d, d * maxlag), dtype=np.float64)
    p_values = np.ones((d, d * maxlag), dtype=np.float64)
    for pattern in np.unique(drop, axis=0):  # targets with the same dropped regressors share a fit
        rows = np.flatnonzero((drop == pattern).all(axis=1))
        kept = np.flatnonzero(~pattern)
        fit, p = _ols(design[:, [0, *(1 + kept)]], targets[:, rows])
        coeff[np.ix_(rows, kept)] = fit.T
        p_values[np.ix_(rows, kept)] = p.T
    p_values = corrected_pvalues(np.where(force, 0.0, p_values), fdr_method, drop)
    return np.where((p_values <= alpha) | force, coeff, 0.0), p_values


def _by_target(links: np.ndarray) -> np.ndarray:
    """``links[k, j, i]`` (lag ``k + 1``, cause ``j``, effect ``i``) as one row per effect ``i`` of
    design columns ``k * d + j``."""
    lags, d, _ = links.shape
    return np.asarray(links, dtype=bool).transpose(2, 0, 1).reshape(d, lags * d)
