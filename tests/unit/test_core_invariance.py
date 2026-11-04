"""The invariance tests of ``andrey.core.invariance``: level, power, and the regressions.

ICP's two K-group tests are checked against a reference that fits every regression on the rows
themselves: the approximate test by Welch's t-test and the F-test on the pooled residuals, the exact
test by Chow's statistic with its full covariance matrix (Peters, Buhlmann and Meinshausen 2016,
eq. 17). The tests compute the same numbers from Gram matrices.
"""

from __future__ import annotations

import itertools

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


ICP_TESTS = ("icp_approximate", "icp_exact")


def groups_data(sizes: tuple[int, ...], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``x0 -> x1 -> y -> x3`` and ``x0 -> y``, with group ``k`` scaling the noise of ``x0`` and
    shifting ``x1`` by ``k``."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(len(sizes)), sizes)
    n = groups.size
    x0 = rng.standard_normal(n) * (1.0 + groups)
    x1 = 0.6 * x0 + rng.standard_normal(n) + 0.8 * groups
    y = 0.5 * x0 - 0.7 * x1 + rng.standard_normal(n)
    x3 = 0.9 * y + rng.standard_normal(n)
    return np.column_stack([x0, x1, y, x3]), groups


def _residuals(X, y):
    design = np.column_stack([np.ones(len(y)), X])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ coef, coef, design


def direct_icp(X, groups, y, S, name):
    """The p-value of the test ``name`` of column ``y`` given ``S``, from the rows."""
    labels = np.unique(groups)
    target, rows = X[:, y], X[:, list(S)]
    p = []
    for g in labels:
        inside = groups == g
        if name == "icp_approximate":
            r, _, _ = _residuals(rows, target)
            a, b = r[inside], r[~inside]
            p_mean = stats.ttest_ind(a, b, equal_var=False).pvalue
            ratio = a.var(ddof=1) / b.var(ddof=1)
            f = stats.f(len(a) - 1, len(b) - 1)
            p.append(2 * min(p_mean, 2 * min(f.cdf(ratio), f.sf(ratio))))
            continue
        r, coef, design = _residuals(rows[~inside], target[~inside])
        in_design = np.column_stack([np.ones(inside.sum()), rows[inside]])
        D = target[inside] - in_design @ coef
        cov = np.eye(inside.sum()) + in_design @ np.linalg.solve(design.T @ design, in_design.T)
        df = (~inside).sum() - in_design.shape[1]
        statistic = D @ np.linalg.solve(cov, D) / (inside.sum() * (r @ r / df))
        p.append(stats.f.sf(statistic, inside.sum(), df))
    factor = 1 if (name == "icp_approximate" and len(labels) == 2) else len(labels)
    return min(1.0, factor * min(p))


def test_the_registry_holds_icps_tests():
    assert set(ICP_TESTS) <= set(INVARIANCE_TESTS)


def test_an_unknown_test_or_one_group_raises():
    X, groups = groups_data((60, 60))
    with pytest.raises(ValueError, match=r"unknown invariance_test 'nope'; registered: \["):
        make_invariance_test("nope", X, groups, indep_test="fisherz")
    with pytest.raises(ValueError, match="compares two or more groups"):
        make_invariance_test("icp_exact", X, np.zeros(len(X), dtype=int), indep_test="fisherz")


@pytest.mark.parametrize("name", ICP_TESTS)
@pytest.mark.parametrize("sizes", [(70, 50), (40, 60, 50)], ids=["two groups", "three groups"])
def test_the_p_value_is_the_one_from_the_rows(sizes, name):
    X, groups = groups_data(sizes)
    test = make_invariance_test(name, X, groups, indep_test="fisherz")
    for S in (S for s in range(4) for S in itertools.combinations((0, 1, 3), s)):
        want = direct_icp(X, groups, 2, S, name)
        assert test(2, S) == pytest.approx(want, rel=1e-9, abs=1e-14), S


def test_two_groups_compare_once_under_the_approximate_test():
    """With two groups, each one against the other is the same test, so the factor is 1."""
    X, groups = groups_data((70, 50))
    r, _, _ = _residuals(X[:, [0]], X[:, 2])
    first = stats.ttest_ind(r[groups == 0], r[groups == 1], equal_var=False).pvalue
    second = stats.ttest_ind(r[groups == 1], r[groups == 0], equal_var=False).pvalue
    assert first == pytest.approx(second, rel=1e-12)
    test = make_invariance_test("icp_approximate", X, groups, indep_test="fisherz")
    assert test(2, [0]) == pytest.approx(
        direct_icp(X, groups, 2, (0,), "icp_approximate"), rel=1e-9
    )


@pytest.mark.parametrize("name", ICP_TESTS)
def test_a_batch_gives_each_querys_p_value_and_pooled_regression(name):
    X, groups = groups_data((40, 60, 50))
    test = make_invariance_test(name, X, groups, indep_test="fisherz")
    sets = np.array([[0, 1], [0, 3], [1, 3]])
    p, _, beta, rss = test.fit(2, sets)
    assert p.tolist() == [test(2, S) for S in sets.tolist()]
    for S, coef, sum_of_squares in zip(sets.tolist(), beta, rss, strict=True):
        r, want, _ = _residuals(X[:, S], X[:, 2])
        np.testing.assert_allclose(coef[1:], want[1:], rtol=1e-9)
        assert sum_of_squares == pytest.approx(r @ r, rel=1e-9)


@pytest.mark.parametrize("name", ICP_TESTS)
def test_rows_outside_the_groups_are_not_read_and_the_set_order_does_not_matter(name):
    X, groups = groups_data((40, 60, 50))
    read = make_invariance_test(name, X[groups < 2], groups[groups < 2], indep_test="fisherz")
    marked = make_invariance_test(name, X, np.where(groups < 2, groups, -1), indep_test="fisherz")
    assert marked(2, [0, 3]) == read(2, [0, 3]) == read(2, [3, 0])


def test_a_singular_regression_names_the_group_it_leaves_out():
    from andrey.core.invariance import SingularRegression

    X, groups = groups_data((60, 60))
    # x0 is 0 on every row outside group 0, and so is its pooled mean: the column is all zero there.
    X[:, 0] = np.r_[np.repeat([1.5, -1.5], 30), np.zeros(60)]
    test = make_invariance_test("icp_exact", X, groups, indep_test="fisherz")
    with pytest.raises(SingularRegression) as error:
        test(2, [0])
    assert error.value.group == 0
