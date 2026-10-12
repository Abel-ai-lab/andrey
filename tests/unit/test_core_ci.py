"""Fisher-Z and Student-t conditional-independence tests: equivalence, error, and known values.

Covers the contract both tests share through :class:`PartialCorrelationTest`: exact agreement of
the scalar, batched, many-query, all-pairs, and first-order paths, argument-order independence, no
per-query state retained, and fail-fast error paths. :class:`FisherZ` adds an independently derived
closed-form p-value for the empty conditioning set and a small inline baseline pinning current
outputs. The numpy path is pinned as the exact float64 oracle. :class:`StudentT` reports the same
partial correlation with a Student-t p-value, checked against a direct scipy computation.
"""

from __future__ import annotations

import sys
from itertools import combinations

import numpy as np
import pytest
from scipy import stats
from scipy.special import ndtr

from andrey.core import ci
from andrey.core.ci import FisherZ, StudentT


@pytest.fixture(autouse=True)
def force_numpy(monkeypatch):
    """numpy is the exact float64 oracle for the correlation matrix and every p-value."""
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


PARTIAL_CORRELATION_TESTS = [FisherZ, StudentT]


@pytest.fixture
def data():
    """A moderate Gaussian sample: enough rows to invert every submatrix up to |S| = 3."""
    rng = np.random.default_rng(3)
    return rng.standard_normal((200, 6))


# ---- scalar-vs-batched agreement --------------------------------------------------------


@pytest.mark.parametrize("test_class", PARTIAL_CORRELATION_TESTS)
@pytest.mark.parametrize("seed", range(15))
def test_batched_matches_scalar_exactly(test_class, seed):
    # Separate instances exercise each path's arithmetic without sharing any computed p-values.
    data = np.random.default_rng(seed).standard_normal((200, 10))
    rng = np.random.default_rng(11)
    n_features = data.shape[1]
    # Group random queries by (x, y) so each batched_call receives several conditioning sets and
    # takes the vectorized _pvalues path; a length-1 batch would route back through scalar __call__.
    queries: dict[tuple[int, int], list[tuple[int, ...]]] = {}
    for _ in range(400):
        x, y = (int(v) for v in rng.choice(n_features, size=2, replace=False))
        pool = [c for c in range(n_features) if c not in (x, y)]
        k = int(rng.integers(0, 4))  # |S| in {0, 1, 2, 3}
        cond = tuple(sorted(int(c) for c in rng.choice(pool, size=k, replace=False)))
        queries.setdefault((x, y), []).append(cond)

    scalar_test = test_class(data)
    batched_test = test_class(data)
    scalar_p: list[float] = []
    batched_p: list[float] = []
    for (x, y), conds in queries.items():
        scalar_p.extend(scalar_test(x, y, c) for c in conds)
        batched_p.extend(batched_test.batched_call(x, y, conds))

    np.testing.assert_array_equal(batched_p, scalar_p)


@pytest.mark.parametrize("test_class", PARTIAL_CORRELATION_TESTS)
@pytest.mark.parametrize("seed", range(15))
def test_prescreen_pvalues_match_canonical_scalar_exactly(test_class, seed):
    data = np.random.default_rng(seed).standard_normal((200, 10))
    test = test_class(data)
    marginal = test.marginal_pvalues()
    for x, y in combinations(range(data.shape[1]), 2):
        assert marginal[x, y] == test(x, y)
        zs = [z for z in range(data.shape[1]) if z not in (x, y)]
        np.testing.assert_array_equal(
            test.first_order_pvalues(x, y, zs), [test(x, y, (z,)) for z in zs]
        )


@pytest.mark.parametrize("test_class", PARTIAL_CORRELATION_TESTS)
@pytest.mark.parametrize("seed", range(5))
def test_many_query_path_matches_scalar_exactly(test_class, seed):
    # Each row has its own pair (in either order) and conditioning set (unsorted); the statistic
    # and the p-value of row i equal the scalar answer for that row, bit for bit.
    data = np.random.default_rng(seed).standard_normal((120, 9))
    test = test_class(data)
    rng = np.random.default_rng(50 + seed)
    for size in range(5):
        rows = []
        for _ in range(60):
            x, y, *cond = (int(v) for v in rng.permutation(9)[: 2 + size])
            rows.append((x, y, cond))
        x = np.array([row[0] for row in rows])
        y = np.array([row[1] for row in rows])
        conditions = np.array([row[2] for row in rows], dtype=np.intp).reshape(len(rows), size)
        statistics, pvalues = test.statistic_pvalues(x, y, conditions)
        scalar = [test.statistic_pvalue(*row) for row in rows]
        np.testing.assert_array_equal(statistics, [r for r, _ in scalar])
        np.testing.assert_array_equal(pvalues, [p for _, p in scalar])


@pytest.mark.parametrize("test_class", PARTIAL_CORRELATION_TESTS)
def test_many_query_path_error_paths(test_class):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    X[:, 4] = X[:, 0]
    one = np.array([1])
    with pytest.raises(ValueError, match="singular"):
        test_class(X).statistic_pvalues(one, one + 1, np.array([[0, 4]]))
    with pytest.raises(ValueError, match="degrees of freedom"):
        test_class(X[:3]).statistic_pvalues(one, one + 1, np.array([[0, 3]]))


@pytest.mark.parametrize("test_class", PARTIAL_CORRELATION_TESTS)
def test_from_corr_rebuilds_an_equal_test_of_the_same_class(test_class, data):
    test = test_class(data)
    rebuilt = test_class.from_corr(*test.to_seed())
    assert type(rebuilt) is test_class
    for query in [(0, 1, ()), (1, 4, (2,)), (4, 2, (0, 1, 3))]:
        assert rebuilt(*query) == test(*query)
    np.testing.assert_array_equal(rebuilt.marginal_pvalues(), test.marginal_pvalues())


@pytest.mark.parametrize("test_class", PARTIAL_CORRELATION_TESTS)
def test_no_per_query_state_is_retained(test_class, data):
    # A skeleton search issues one query per conditioning subset it visits -- combinatorially many
    # at large d -- so the test object must hold exactly the same state after answering them as
    # before (the correlation matrix and the sample size). Any per-query memo would grow with the
    # subsets visited and take the search's memory with it.
    test = test_class(data)

    def footprint() -> dict[str, tuple[type, int]]:
        return {
            name: (type(value), getattr(value, "nbytes", None) or sys.getsizeof(value))
            for name, value in vars(test).items()
        }

    before = footprint()
    n_features = data.shape[1]
    for x, y in combinations(range(n_features), 2):
        pool = [c for c in range(n_features) if c not in (x, y)]
        conds = [s for k in range(4) for s in combinations(pool, k)]
        test.batched_call(x, y, conds)
        for s in conds[:5]:
            test(x, y, s)
    assert footprint() == before


@pytest.mark.parametrize("test_class", PARTIAL_CORRELATION_TESTS)
def test_empty_batch_returns_empty_list(test_class, data):
    assert test_class(data).batched_call(0, 1, []) == []


# ---- argument-order independence ---------------------------------------------------------


def test_p_value_independent_of_argument_order(data):
    # (x, y) is sorted and the condition set is canonicalized to sorted-unique, so swapping the pair
    # and reordering the conditioning columns must give the identical p-value.
    test = FisherZ(data)
    assert test(0, 1, (2, 3)) == test(1, 0, (3, 2))
    assert test(4, 2, (0, 1, 5)) == test(2, 4, (5, 1, 0))


def test_duplicate_condition_columns_canonicalized(data):
    # A repeated conditioning column collapses to the sorted-unique set, matching the deduplicated
    # query exactly.
    test = FisherZ(data)
    assert test(0, 1, (2, 2, 3)) == test(0, 1, (2, 3))


# ---- fail-fast error paths ---------------------------------------------------------------


def test_singular_correlation_raises_value_error():
    # A duplicated column makes the [x, y, *S] correlation submatrix exactly singular. The duplicate
    # pair must sit inside that submatrix with |S| >= 2 so the inversion actually runs -- |S| <= 1
    # takes the closed-form partial correlation and never inverts -- and the LinAlgError surfaces as
    # a ValueError, not a raw LinAlgError.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    X[:, 4] = X[:, 0]
    test = FisherZ(X)
    with pytest.raises(ValueError, match="singular"):
        test(1, 2, (0, 4))


def test_too_few_samples_raises_value_error():
    # With n - |S| - 3 < 0 the Fisher-Z degrees of freedom go negative; the test refuses rather than
    # taking a square root of a negative number.
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        FisherZ(rng.standard_normal((2, 4)))(0, 1)
    with pytest.raises(ValueError):
        FisherZ(rng.standard_normal((4, 4)))(0, 3, (1, 2))


def test_nonfinite_data_rejected_at_construction():
    with pytest.raises(ValueError, match="NaN or Inf"):
        FisherZ(np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]))
    with pytest.raises(ValueError, match="NaN or Inf"):
        FisherZ(np.array([[1.0, np.inf], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]))


def test_wrong_ndim_rejected_at_construction():
    with pytest.raises(ValueError, match="2-D"):
        FisherZ(np.arange(10.0))


def test_variable_in_condition_set_is_assertion_error(data):
    # x or y appearing in the conditioning set is a caller contract violation, surfaced as an
    # AssertionError rather than a silent wrong answer -- on both the scalar and batched paths.
    test = FisherZ(data)
    with pytest.raises(AssertionError, match="condition_set"):
        test(0, 1, (0, 2))
    with pytest.raises(AssertionError, match="condition_set"):
        test.batched_call(0, 1, [(2,), (1, 3)])


# ---- independently verifiable known value ------------------------------------------------


def test_empty_condition_set_is_marginal_pearson(data):
    # With no conditioning the partial correlation reduces to the marginal Pearson r of columns
    # x and y, so p = 2 * (1 - Phi(sqrt(n - 3) * |arctanh(r)|)). This closed form is computed
    # straight from np.corrcoef and np.arctanh, independent of the test's internal inversion path.
    n = data.shape[0]
    for x, y in [(0, 1), (2, 5), (3, 4)]:
        r = np.corrcoef(data[:, x], data[:, y])[0, 1]
        expected = 2.0 * (1.0 - ndtr(np.sqrt(n - 3) * abs(np.arctanh(r))))
        assert FisherZ(data)(x, y) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("condition_set", [(), (2,), (2, 3), (2, 3, 4)])
def test_statistic_is_the_residual_correlation_and_the_p_value_is_the_call(data, condition_set):
    # Each closed form (|S| = 0, 1) and the inversion path (|S| >= 2) gives the correlation of the
    # least-squares residuals of x and y on the conditioning columns, and the p-value of a call.
    test = FisherZ(data)
    design = np.column_stack([np.ones(len(data)), data[:, list(condition_set)]])
    residuals = [data[:, k] - design @ np.linalg.lstsq(design, data[:, k])[0] for k in (0, 5)]
    r, p = test.statistic_pvalue(5, 0, condition_set)
    assert r == pytest.approx(np.corrcoef(*residuals)[0, 1], abs=1e-12)
    assert p == test(0, 5, condition_set)


def test_independent_columns_give_large_p_dependent_give_small_p():
    # A strong linear dependence drives the p-value toward 0 (reject independence); an unrelated
    # column pair leaves it well above any usual threshold. Directional sanity.
    rng = np.random.default_rng(7)
    z = rng.standard_normal(500)
    dependent = np.column_stack([z, z + 0.01 * rng.standard_normal(500)])
    independent = rng.standard_normal((500, 2))
    assert FisherZ(dependent)(0, 1) < 1e-6
    assert FisherZ(independent)(0, 1) > 0.05


# ---- inline baseline -----------------------------------------------------------------------

# Fixed seed-3 standard-normal sample, shape (200, 6). Values captured from the current numpy-path
# outputs and cross-checked against the closed-form marginal test above; they pin the p-values so a
# regression is caught with no reference implementation on the path.
_BASELINE = {
    (1, 3, ()): 0.21578957156142087,
    (0, 1, (2, 3)): 0.32926308886248035,
    (2, 5, (0, 1, 3)): 0.2732338572135704,
}


def test_matches_inline_baseline(data):
    test = FisherZ(data)
    for (x, y, cond), expected in _BASELINE.items():
        assert test(x, y, cond) == pytest.approx(expected, abs=1e-12)


# ---- Student-t test -------------------------------------------------------------------------


def _residual_correlation(data, x, y, condition_set):
    """Correlation of the least-squares residuals of columns ``x`` and ``y`` on the conditions."""
    design = np.column_stack([np.ones(len(data)), data[:, list(condition_set)]])
    residuals = [data[:, k] - design @ np.linalg.lstsq(design, data[:, k])[0] for k in (x, y)]
    return np.corrcoef(*residuals)[0, 1]


@pytest.mark.parametrize(
    ("x", "y", "condition_set"),
    [(0, 1, ()), (2, 5, ()), (1, 4, (2,)), (5, 0, (3,)), (0, 1, (2, 3)), (4, 2, (0, 1, 3))],
)
def test_student_t_p_value_is_the_t_tail_of_the_residual_correlation(data, x, y, condition_set):
    r = _residual_correlation(data, x, y, condition_set)
    df = len(data) - len(condition_set) - 2
    expected = 2 * stats.t.sf(abs(r * np.sqrt(df / (1 - r * r))), df)
    statistic, p = StudentT(data).statistic_pvalue(x, y, condition_set)
    assert statistic == pytest.approx(r, abs=1e-12)
    assert p == pytest.approx(expected, rel=1e-10)
    assert StudentT(data)(x, y, condition_set) == p


def test_student_t_fast_paths_are_the_t_tail_of_the_partial_correlation(data):
    test = StudentT(data)
    n, d = data.shape

    def expected(x, y, condition_set):
        r = _residual_correlation(data, x, y, condition_set)
        df = n - len(condition_set) - 2
        return 2 * stats.t.sf(abs(r * np.sqrt(df / (1 - r * r))), df)

    sets = [(), (2,), (3,), (2, 3), (3, 4), (2, 3, 4)]
    np.testing.assert_allclose(
        test.batched_call(0, 1, sets), [expected(0, 1, s) for s in sets], rtol=1e-10
    )
    marginal = test.marginal_pvalues()
    first_order = test.first_order_pvalues(0, 5, [1, 2, 3, 4])
    for x, y in combinations(range(d), 2):
        assert marginal[x, y] == pytest.approx(expected(x, y, ()), rel=1e-10)
    np.testing.assert_allclose(
        first_order, [expected(0, 5, (z,)) for z in (1, 2, 3, 4)], rtol=1e-10
    )


def test_student_t_reports_the_statistic_fisher_z_reports(data):
    for x, y, condition_set in [(0, 1, ()), (1, 4, (2,)), (4, 2, (0, 1, 3))]:
        assert (
            StudentT(data).statistic_pvalue(x, y, condition_set)[0]
            == (FisherZ(data).statistic_pvalue(x, y, condition_set)[0])
        )


def test_student_t_p_value_is_independent_of_argument_order(data):
    test = StudentT(data)
    assert test(4, 1, (3, 2)) == test(1, 4, (2, 3, 3))


def test_student_t_is_not_a_registered_indep_test():
    assert "student_t" not in ci.available_indep_tests()
    assert "student_t" not in ci.SHIPPED_INDEP_TESTS


def test_student_t_needs_more_samples_than_conditions_plus_two():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="degrees of freedom"):
        StudentT(rng.standard_normal((2, 4)))(0, 1)
    with pytest.raises(ValueError, match="degrees of freedom"):
        StudentT(rng.standard_normal((4, 4)))(0, 3, (1, 2))
    r, p = StudentT(rng.standard_normal((3, 4))).statistic_pvalue(0, 1)  # one degree of freedom
    assert abs(r) < 1 and 0 < p <= 1


def test_student_t_on_exactly_dependent_columns_gives_a_finite_tiny_p_value():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(50)
    for sign in (1.0, -1.0):
        r, p = StudentT(np.column_stack([x, sign * x, rng.standard_normal(50)])).statistic_pvalue(
            0, 1
        )
        assert abs(r) < 1 and np.isclose(abs(r), 1.0)
        assert 0 <= p < 1e-12


def test_student_t_shares_the_fisher_z_error_paths():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    X[:, 4] = X[:, 0]
    with pytest.raises(ValueError, match="singular"):
        StudentT(X)(1, 2, (0, 4))
    with pytest.raises(ValueError, match="NaN or Inf"):
        StudentT(np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]))
    with pytest.raises(AssertionError, match="condition_set"):
        StudentT(X)(0, 1, (0, 2))
