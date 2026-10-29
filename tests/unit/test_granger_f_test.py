"""Granger's lagged-coefficient tests agree with statsmodels' vector autoregression."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

import andrey
from andrey.temporal.granger import granger_f_test


def var_series(n: int = 400, seed: int = 0) -> np.ndarray:
    """A stable VAR(2): x0 -> x1 at lag 1, x1 -> x2 at lag 2, each variable on its own lag 1."""
    rng = np.random.default_rng(seed)
    d = 3
    x = np.zeros((n + 50, d))
    for t in range(2, n + 50):
        e = rng.standard_normal(d)
        x[t, 0] = 0.5 * x[t - 1, 0] + e[0]
        x[t, 1] = 0.4 * x[t - 1, 1] + 0.5 * x[t - 1, 0] + e[1]
        x[t, 2] = 0.3 * x[t - 1, 2] + 0.5 * x[t - 2, 1] + e[2]
    return x[50:]


@pytest.mark.parametrize("lags", [1, 2, 3])
def test_coefficients_and_p_values_are_the_var_regression_ones(lags):
    """``[i, j + d * (k - 1)]`` is the coefficient of variable ``j`` at lag ``k`` in the regression
    of ``i``, and its p-value is the two-sided t-test of ``VARResults``' t-ratio with its residual
    degrees of freedom, the F-test of that regressor (row 0 of its tables is the constant, its
    columns the equations). ``VARResults.pvalues`` takes the same t-ratio against the normal
    distribution, which the t-test approaches as the series grows."""
    x = var_series()
    d = x.shape[1]
    result = VAR(x).fit(maxlags=lags, ic=None, trend="c")
    coeff, p_values = granger_f_test(x, maxlag=lags, alpha=1.0)
    assert coeff.shape == p_values.shape == (d, d * lags)
    np.testing.assert_allclose(coeff, result.params[1:].T, rtol=1e-8, atol=1e-12)
    reference = 2 * stats.t.sf(np.abs(result.tvalues[1:].T), result.df_resid)
    np.testing.assert_allclose(p_values, reference, rtol=1e-6, atol=1e-300)
    assert (p_values >= result.pvalues[1:].T).all()  # the t distribution has the heavier tails


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_two_variables_at_one_lag_give_the_pairwise_granger_f_test(seed):
    """With two variables and one lag, the coefficient test is statsmodels' ``ssr_ftest``."""
    x = var_series(seed=seed)[:, :2]
    _, p_values = granger_f_test(x, maxlag=1)
    for cause, effect in ((0, 1), (1, 0)):
        reference = grangercausalitytests(x[:, [effect, cause]], maxlag=1)[1][0]["ssr_ftest"][1]
        np.testing.assert_allclose(p_values[effect, cause], reference, rtol=1e-8, atol=1e-300)


def test_an_edge_is_a_p_value_at_most_alpha_and_carries_its_coefficient():
    x = var_series()
    full, p_values = granger_f_test(x, maxlag=2, alpha=1.0)
    for alpha in (0.001, 0.05, 0.5):
        coeff, same = granger_f_test(x, maxlag=2, alpha=alpha)
        np.testing.assert_array_equal(same, p_values)
        np.testing.assert_array_equal(coeff != 0, p_values <= alpha)
        np.testing.assert_array_equal(coeff[coeff != 0], full[coeff != 0])


def test_the_facade_returns_the_planted_edges_and_their_p_values():
    out = andrey.granger(var_series(n=1000), tau_max=2)
    edges = {(k, i, j) for k in (1, 2) for i, j, _ in out.structure.lag(k).oriented_edges()}
    assert {(1, 0, 1), (2, 1, 2)} <= edges
    p_values = np.asarray(out.metadata["p_values"])
    assert p_values.shape == (2, 3, 3)  # [lag - 1, cause, effect]
    for k, i, j in edges:
        assert p_values[k - 1, i, j] <= 0.05
        assert out.structure.lag_weights[k - 1][i, j] != 0
    assert p_values[0, 0, 1] < 1e-6 and p_values[1, 1, 2] < 1e-6


def test_a_lower_alpha_returns_a_subset_of_the_edges():
    x = var_series(n=300)
    sets = [
        {
            (k, i, j)
            for k in (1, 2)
            for i, j, _ in andrey.granger(x, tau_max=2, alpha=a).structure.lag(k).oriented_edges()
        }
        for a in (0.001, 0.05, 0.5)
    ]
    assert sets[0] <= sets[1] <= sets[2] and sets[0] < sets[2]


def test_the_edges_and_p_values_do_not_depend_on_the_units():
    x = var_series()
    scaled = x * np.array([1e-6, 1.0, 1e6])
    plain, rescaled = andrey.granger(x, tau_max=2), andrey.granger(scaled, tau_max=2)
    np.testing.assert_allclose(
        plain.metadata["p_values"], rescaled.metadata["p_values"], rtol=1e-6, atol=1e-300
    )
    for k in (1, 2):
        assert plain.structure.lag(k).oriented_edges() == rescaled.structure.lag(k).oriented_edges()


def test_cv_is_ignored_with_a_warning():
    x = var_series()
    with pytest.warns(andrey.DeprecatedArgumentWarning, match="cv"):
        out = andrey.granger(x, cv=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        same = andrey.granger(x)
    np.testing.assert_array_equal(out.metadata["p_values"], same.metadata["p_values"])


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5])
def test_alpha_must_lie_in_the_open_unit_interval(alpha):
    with pytest.raises(ValueError, match="alpha"):
        andrey.granger(var_series(), alpha=alpha)


def test_a_regression_without_a_degree_of_freedom_left_is_rejected():
    x = var_series(n=40)
    needed = (x.shape[1] + 1) * 2 + 2
    with pytest.raises(ValueError, match=f"at least {needed} rows"):
        andrey.granger(x[: needed - 1], tau_max=2)
    assert andrey.granger(x[:needed], tau_max=2).structure.lags == (1, 2)


def test_a_variable_that_is_a_combination_of_others_is_rejected():
    x = var_series()
    with pytest.raises(ValueError, match="singular"):
        andrey.granger(np.column_stack([x, x[:, 0] + 2 * x[:, 1]]))
