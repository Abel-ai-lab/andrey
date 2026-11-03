"""The invariance tests of ``andrey.core.invariance``: level, power, and the regressions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from andrey.core.invariance import (
    INVARIANCE_TESTS,
    TWO_GROUP_INVARIANCE_TESTS,
    default_invariance_test,
    make_invariance_test,
)


def two_groups(rng, n0=300, n1=300, *, shift=0.0, weight=0.5, sd=1.0):
    """Columns x0, x1, y, w: y = weight * x0 + noise in group 1, 0.5 * x0 + noise in group 0.

    Group 1's x0 has another mean and spread, so the marginal of y changes even under the null.
    """
    x0 = np.r_[rng.standard_normal(n0), 1.0 + 2.0 * rng.standard_normal(n1)]
    x1 = rng.standard_normal(n0 + n1)
    noise = rng.standard_normal(n0 + n1)
    y = np.r_[0.5 * x0[:n0] + noise[:n0], weight * x0[n0:] + shift + sd * noise[n0:]]
    w = y + rng.standard_normal(n0 + n1)
    groups = np.r_[np.zeros(n0, dtype=int), np.ones(n1, dtype=int)]
    return np.column_stack([x0, x1, y, w]), groups


def direct(X, groups, y, S):
    """The Gaussian test's p-value from least squares on the rows of each group."""

    def fit(rows):
        A = np.column_stack([np.ones(rows.sum()), X[rows][:, S]])
        coef, _, rank, _ = np.linalg.lstsq(A, X[rows, y], rcond=None)
        r = X[rows, y] - A @ coef
        return r @ r, rank, rows.sum()

    (r0, q0, n0), (r1, q1, n1) = fit(groups == 0), fit(groups == 1)
    rp, qp, _ = fit(groups >= 0)
    df1, df2 = q0 + q1 - qp, n0 + n1 - q0 - q1
    ratio = (r1 / (n1 - q1)) / (r0 / (n0 - q0))
    p_var = 2 * min(stats.f.cdf(ratio, n1 - q1, n0 - q0), stats.f.sf(ratio, n1 - q1, n0 - q0))
    if df1 == 0:
        return min(1.0, 2 * p_var)
    p_coef = stats.f.sf(((rp - r0 - r1) / df1) / ((r0 + r1) / df2), df1, df2)
    return min(1.0, 2 * min(p_coef, p_var))


def test_the_registry_holds_the_two_group_tests():
    assert set(TWO_GROUP_INVARIANCE_TESTS) <= set(INVARIANCE_TESTS)
    assert default_invariance_test("fisherz") == "gaussian"
    assert default_invariance_test("normal_scores") == "gaussian"
    assert default_invariance_test("kci") == "indicator"
    assert default_invariance_test("gsq") == "indicator"


def test_an_unknown_test_or_more_than_two_groups_raise():
    X, groups = two_groups(np.random.default_rng(0))
    with pytest.raises(ValueError, match="unknown invariance_test 'chow'"):
        make_invariance_test("chow", X, groups, indep_test="fisherz")
    with pytest.raises(ValueError, match="compares two groups"):
        make_invariance_test("gaussian", X, np.where(groups == 1, 2, 0), indep_test="fisherz")


@pytest.mark.parametrize(("y", "S"), [(2, [0]), (2, [0, 1]), (2, []), (3, [0, 1, 2]), (1, [3])])
def test_the_gaussian_p_value_is_the_least_squares_one(y, S):
    X, groups = two_groups(np.random.default_rng(1), shift=0.3)
    test = make_invariance_test("gaussian", X, groups, indep_test="fisherz")
    assert test(y, S) == pytest.approx(direct(X, groups, y, S), rel=1e-9)


def test_the_gaussian_level_holds_under_the_null():
    """Equal conditionals, unequal marginals of the conditioning variable: the rejection rate at
    0.05 is within the 99% binomial interval of 2,000 draws."""
    rng = np.random.default_rng(2)
    draws = 2000
    rejected = 0
    for _ in range(draws):
        X, groups = two_groups(rng, 60, 40)
        rejected += (
            make_invariance_test("gaussian", X, groups, indep_test="fisherz")(2, [0]) <= 0.05
        )
    low, high = stats.binom.interval(0.99, draws, 0.05)
    assert low <= rejected <= high


@pytest.mark.parametrize(
    "change",
    [{"shift": 1.0}, {"weight": 0.25}, {"sd": 2**0.5}],
    ids=["shift", "weight", "variance"],
)
def test_the_gaussian_test_rejects_a_changed_conditional(change):
    X, groups = two_groups(np.random.default_rng(3), 2000, 2000, **change)
    assert make_invariance_test("gaussian", X, groups, indep_test="fisherz")(2, [0]) < 1e-5


def test_the_gaussian_test_reads_no_units_and_no_group_order():
    X, groups = two_groups(np.random.default_rng(4), shift=0.2)
    p = make_invariance_test("gaussian", X, groups, indep_test="fisherz")(2, [0, 1])
    scaled = X * np.array([3.0, -0.5, 2.0**-7, 11.0]) + np.array([5.0, -1.0, 0.25, 100.0])
    assert make_invariance_test("gaussian", scaled, groups, indep_test="fisherz")(
        2, [0, 1]
    ) == pytest.approx(p, rel=1e-9)
    swapped = make_invariance_test("gaussian", X, 1 - groups, indep_test="fisherz")
    assert swapped(2, [0, 1]) == pytest.approx(p, rel=1e-9)


def test_a_conditioning_column_constant_in_a_group_costs_a_degree_of_freedom():
    rng = np.random.default_rng(5)
    X, groups = two_groups(rng)
    X[groups == 1, 0] = 1.5
    test = make_invariance_test("gaussian", X, groups, indep_test="fisherz")
    assert test(2, [0]) == pytest.approx(direct(X, groups, 2, [0]), rel=1e-9)
    X[groups == 0, 0] = -1.0  # constant at different values: no coefficient to compare
    test = make_invariance_test("gaussian", X, groups, indep_test="fisherz")
    assert test(2, [0]) == pytest.approx(direct(X, groups, 2, [0]), rel=1e-9)


def test_normal_scores_read_each_column_through_its_ranks_over_all_rows():
    X, groups = two_groups(np.random.default_rng(6), shift=0.3)
    test = make_invariance_test("gaussian", X, groups, indep_test="normal_scores")
    monotone = X.copy()
    monotone[:, 0] = np.exp(X[:, 0])
    monotone[:, 2] = X[:, 2] ** 3
    other = make_invariance_test("gaussian", monotone, groups, indep_test="normal_scores")
    assert other(2, [0]) == pytest.approx(test(2, [0]), rel=1e-12)


@pytest.mark.parametrize("indep_test", ["fisherz", "kci"])
def test_the_indicator_test_rejects_a_shift_and_keeps_an_unchanged_conditional(indep_test):
    rng = np.random.default_rng(7)
    X, groups = two_groups(rng, 200, 200, shift=1.0)
    test = make_invariance_test("indicator", X, groups, indep_test=indep_test, seed=0)
    assert test(2, [0]) < 1e-3
    X, groups = two_groups(rng, 200, 200)
    test = make_invariance_test("indicator", X, groups, indep_test=indep_test, seed=0)
    assert test(2, [0]) > 0.01


def test_the_indicator_test_reads_categories_with_a_categorical_test():
    rng = np.random.default_rng(8)
    x = rng.integers(0, 3, 1200)
    y = np.where(rng.random(1200) < 0.7, x, rng.integers(0, 3, 1200))
    groups = np.repeat([0, 1], 600)
    changed = y.copy()
    changed[600:] = rng.integers(0, 3, 600)
    for column, invariant in ((y, True), (changed, False)):
        X = np.column_stack([x, column]).astype(float)
        p = make_invariance_test("indicator", X, groups, indep_test="gsq")(1, [0])
        assert (p > 0.01) is invariant


def test_the_indicator_test_marks_the_indicator_categorical_for_a_mixed_test():
    X, groups = two_groups(np.random.default_rng(9), 300, 300, shift=1.0)
    test = make_invariance_test(
        "indicator", X, groups, indep_test="cond_gauss", categorical=np.zeros(4, dtype=bool)
    )
    assert test(2, [0]) < 1e-3


def test_rows_outside_the_two_groups_are_not_read():
    X, groups = two_groups(np.random.default_rng(10), shift=0.4)
    extra = np.vstack([X, 50.0 + X[:100]])
    labeled = np.r_[groups, np.full(100, -1)]
    for name in TWO_GROUP_INVARIANCE_TESTS:
        p = make_invariance_test(name, X, groups, indep_test="fisherz")(2, [0])
        assert make_invariance_test(name, extra, labeled, indep_test="fisherz")(
            2, [0]
        ) == pytest.approx(p, rel=1e-9)
