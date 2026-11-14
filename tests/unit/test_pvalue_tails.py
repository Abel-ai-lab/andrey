"""Exact upper-tail p-values: a strong dependence keeps a positive p-value far below ``1e-16``.

The tests whose p-value was ``1 - cdf`` (Fisher z, the Fisher-aggregated HSIC test of
BottomUpParceLiNGAM, and the Anderson-Darling test of one value) compute the upper tail directly.
Each agrees with a closed-form or high-precision reference in the tail, and with the subtraction
where that is accurate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import gammainc, ndtr

from andrey.core.ci import FisherZ, NormalScores
from andrey.core.independence import gamma_gram
from andrey.irregular.bottom_up_parce import _fisher_hsic_test
from andrey.markov import anderson_darling_uniform

N = 200


def _two_sided_normal_tail(x: float) -> float:
    """``2 * P(Z > x)``, from the complementary error function, not from ``ndtr``."""
    return math.erfc(x / math.sqrt(2.0))


def _corr(r: float) -> np.ndarray:
    """Three variables: ``r`` between the first two, the third uncorrelated with both."""
    return np.array([[1.0, r, 0.0], [r, 1.0, 0.0], [0.0, 0.0, 1.0]])


# ---- Fisher z ----------------------------------------------------------------------------------


@pytest.mark.parametrize("r", [0.8, 0.9, 0.95, -0.9])
@pytest.mark.parametrize("condition_set", [(), (2,)])
def test_fisher_z_strong_dependence_is_positive_and_matches_the_reference(r, condition_set):
    # sqrt(df) * |z| runs from about 15 to 26 here: the old p-value was 0 from about 8.3 on.
    p = FisherZ.from_corr(_corr(r), N)(0, 1, condition_set)
    df = N - len(condition_set) - 3
    expected = _two_sided_normal_tail(math.sqrt(df) * abs(math.atanh(r)))
    assert 0.0 < p < 1e-16
    assert p == pytest.approx(expected, rel=1e-9, abs=0.0)


def test_fisher_z_p_value_matches_mpmath():
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 50
    for r in (0.8, 0.9, 0.95):
        x = mpmath.sqrt(N - 3) * mpmath.atanh(mpmath.mpf(r))
        expected = float(mpmath.erfc(x / mpmath.sqrt(2)))
        assert FisherZ.from_corr(_corr(r), N)(0, 1) == pytest.approx(expected, rel=1e-9, abs=0.0)


def test_fisher_z_orders_strong_dependences_by_strength():
    # Every p-value below 2e-16 used to be 0, so these tied.
    test = FisherZ.from_corr(_corr(0.8), N)
    stronger = FisherZ.from_corr(_corr(0.9), N)
    assert stronger(0, 1) < test(0, 1) < 1e-16


def test_normal_scores_p_value_is_positive_on_a_strong_dependence():
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    data = np.column_stack([x, x + 0.5 * rng.normal(size=100)])
    r, p = NormalScores(data).statistic_pvalue(0, 1)
    expected = _two_sided_normal_tail(math.sqrt(100 - 3) * math.atanh(abs(r)))
    assert 0.0 < p < 1e-16
    assert p == pytest.approx(expected, rel=1e-9, abs=0.0)


def test_fisher_z_moderate_p_values_equal_the_subtraction():
    r = np.concatenate([np.linspace(-0.4, 0.4, 41), [0.0]])
    df = 30
    z = np.arctanh(r)
    old = 2.0 * (1.0 - ndtr(np.sqrt(df) * np.abs(z)))
    new = np.asarray(FisherZ._pvalue(r, df))
    assert old.min() > 1e-3
    np.testing.assert_allclose(new, old, rtol=1e-12, atol=0.0)
    assert FisherZ._pvalue(0.0, df) == 1.0


# ---- Fisher-aggregated HSIC test --------------------------------------------------------------


def _upper_chi_squared_tail(k: int, half_statistic: float) -> float:
    """``exp(-x) * sum_{j < k} x^j / j!``, the gamma upper tail of integer shape ``k`` at ``x``."""
    return math.exp(-half_statistic) * sum(half_statistic**j / math.factorial(j) for j in range(k))


def _fisher_hsic(residual_scale: float, seed: int) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    n = 200
    predictors = [rng.normal(size=n), rng.normal(size=n)]
    residual = residual_scale * predictors[0] + rng.normal(size=n)
    grams = [gamma_gram(p[:, None]) for p in predictors]
    p, statistic = _fisher_hsic_test(grams, residual[:, None], n, np.inf)
    return p, statistic, len(predictors)


def test_fisher_hsic_strong_dependence_is_positive_and_matches_the_reference():
    # The statistic is about 238, so the p-value is about 2e-50; `1 - gammainc` gave 0.
    p, statistic, k = _fisher_hsic(10.0, 0)
    assert 0.0 < p < 1e-16
    assert p == pytest.approx(_upper_chi_squared_tail(k, 0.5 * statistic), rel=1e-9, abs=0.0)


@pytest.mark.parametrize("scale", [0.0, 0.35])
def test_fisher_hsic_moderate_p_values_equal_the_subtraction(scale):
    p, statistic, k = _fisher_hsic(scale, 1)
    old = 1.0 - gammainc(k, 0.5 * statistic)
    assert 1e-3 < p < 1.0
    assert p == pytest.approx(old, rel=1e-12, abs=0.0)


# ---- Anderson-Darling test of one value --------------------------------------------------------


@pytest.mark.parametrize("u", [1e-5, 1e-12, 1e-20, 1e-150, 1.0 - 1e-12])
def test_anderson_darling_one_value_is_exact_in_the_tail(u):
    # A single value has A^2 = -1 - ln(u (1 - u)), and its p-value is 2 min(u, 1 - u).
    _, p = anderson_darling_uniform([u])
    expected = 2.0 * min(u, 1.0 - u)
    assert p > 0.0
    assert p == pytest.approx(expected, rel=1e-9, abs=0.0)


@pytest.mark.parametrize("u", [0.5, 0.3, 0.05, 0.99])
def test_anderson_darling_one_value_moderate_p_values_equal_the_subtraction(u):
    a2, p = anderson_darling_uniform([u])
    old = 1.0 - math.sqrt(max(0.0, 1.0 - 4.0 * math.exp(-1.0 - a2)))
    assert p == pytest.approx(old, rel=1e-12, abs=0.0)
