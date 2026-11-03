"""The invariance tests of ``andrey.core.invariance``: ICP's two K-group regression tests.

The reference fits every regression on the rows themselves: the approximate test by Welch's t-test
and the F-test on the pooled residuals, the exact test by Chow's statistic with its full
covariance matrix (Peters, Buhlmann and Meinshausen 2016, eq. 17). The tests compute the same
numbers from Gram matrices.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats

from andrey.core.invariance import INVARIANCE_TESTS, make_invariance_test

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


def direct(X, groups, y, S, name):
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
        want = direct(X, groups, 2, S, name)
        assert test(2, S) == pytest.approx(want, rel=1e-9, abs=1e-14), S


def test_two_groups_compare_once_under_the_approximate_test():
    """With two groups, each one against the other is the same test, so the factor is 1."""
    X, groups = groups_data((70, 50))
    r, _, _ = _residuals(X[:, [0]], X[:, 2])
    first = stats.ttest_ind(r[groups == 0], r[groups == 1], equal_var=False).pvalue
    second = stats.ttest_ind(r[groups == 1], r[groups == 0], equal_var=False).pvalue
    assert first == pytest.approx(second, rel=1e-12)
    test = make_invariance_test("icp_approximate", X, groups, indep_test="fisherz")
    assert test(2, [0]) == pytest.approx(direct(X, groups, 2, (0,), "icp_approximate"), rel=1e-9)


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
