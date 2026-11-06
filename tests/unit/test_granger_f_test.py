"""Granger's lagged-coefficient tests are the least-squares t-tests of each equation of the vector
autoregression, computed here in closed form."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import andrey
from andrey.temporal.granger import granger_f_test


def least_squares(y: np.ndarray, regressors: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The least-squares fit of ``y`` on a constant and ``regressors``: each regressor's
    coefficient, the two-sided t-test p-value of each on the residual degrees of freedom, and the
    residual sum of squares."""
    design = np.column_stack([np.ones(len(y)), regressors])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ coef
    df = len(y) - design.shape[1]
    rss = float(residuals @ residuals)
    se = np.sqrt(rss / df * np.diag(np.linalg.inv(design.T @ design)))
    return coef[1:], 2 * stats.t.sf(np.abs(coef / se), df)[1:], rss


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
    of ``i`` on a constant and every variable's ``lags`` past values, and its p-value is that
    coefficient's two-sided t-test on the regression's residual degrees of freedom, the F-test of
    that one regressor."""
    x = var_series()
    n, d = x.shape
    design = np.hstack([x[lags - k : n - k] for k in range(1, lags + 1)])
    coeff, p_values = granger_f_test(x, maxlag=lags, alpha=1.0)
    assert coeff.shape == p_values.shape == (d, d * lags)
    for i in range(d):
        reference, reference_p, _ = least_squares(x[lags:, i], design)
        np.testing.assert_allclose(coeff[i], reference, rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(p_values[i], reference_p, rtol=1e-6, atol=1e-300)


def test_two_variables_at_one_lag_give_the_pairwise_granger_f_test():
    """With two variables and one lag, the coefficient test is the F-test of the cause's past
    against the effect's own: ``F = (RSS_own - RSS_both) / (RSS_both / df)`` on 1 and ``df``
    degrees of freedom, ``df`` the rows less the three coefficients of the larger regression."""
    x = var_series()[:, :2]
    _, p_values = granger_f_test(x, maxlag=1)
    for cause, effect in ((0, 1), (1, 0)):
        y = x[1:, effect]
        *_, own = least_squares(y, x[:-1, [effect]])
        *_, both = least_squares(y, x[:-1, [effect, cause]])
        df = len(y) - 3
        reference = stats.f.sf((own - both) / (both / df), 1, df)
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
