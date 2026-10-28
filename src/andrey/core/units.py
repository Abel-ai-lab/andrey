"""Unit-scale columns for the numerical steps that would otherwise read the data's units.

A method's result depends on the data's units only where its model does. A few numerical steps
reach the units through a shortcut instead: a rank cutoff relative to the largest singular value
(least squares, a pseudo-inverse), an optimizer's start and tolerances, or an estimator defined only
up to the columns' scale. Each such step runs on unit-scale columns, the data's columns multiplied
by their :func:`unit_factors`, and its results are mapped back to the data's units: a coefficient
on a scaled predictor is multiplied by the predictor's factor, and a value in a scaled column's
units is divided by the column's factor.
"""

from __future__ import annotations

import numpy as np


def unit_factors(x: np.ndarray, *, standardize: bool = False) -> np.ndarray:
    """The factor of each column of ``x`` that puts the column on unit scale.

    By default each factor is the power of two that puts the column's standard deviation in
    ``[2**-0.5, 2**0.5)``. A product with a power of two is exact, so results map back exactly; a
    column already in that range, such as a standardized one, keeps the factor ``1`` and its bits;
    and a column rescaled by any power of two gives the same scaled column. With
    ``standardize=True`` each factor is one over the standard deviation, for a step that still
    reads the scale within a factor of two, such as an optimizer: the scaled column is then the
    same, up to rounding, for any positive unit. A constant column keeps the factor ``1``.
    """
    sd = np.std(x, axis=0)
    usable = np.isfinite(sd) & (sd > 0)
    if standardize:
        return np.where(usable, 1.0 / np.where(usable, sd, 1.0), 1.0)
    mantissa, exponent = np.frexp(sd)
    exponent = np.where(mantissa < 2.0**-0.5, exponent - 1, exponent)
    return np.where(usable, np.ldexp(1.0, -exponent), 1.0)


def least_squares(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary least squares of ``y`` on ``x`` with an intercept: ``(coef, intercept)``.

    scikit-learn's ``LinearRegression`` drops the directions whose singular value lies below its
    ``tol`` relative to the largest, so on raw columns it can drop a predictor whose scale lies far
    from the others'. The fit runs on the unit-scale predictors (:func:`unit_factors`) and the
    coefficients are mapped back. A ``y`` with several columns gives one row of ``coef`` and one
    ``intercept`` per column. The fitted values are ``x @ coef.T + intercept``.
    """
    from sklearn.linear_model import LinearRegression

    factors = unit_factors(x)
    fit = LinearRegression().fit(x * factors, y)
    return fit.coef_ * factors, np.asarray(fit.intercept_)
