"""Fisher-Z and Student-t conditional-independence tests: equivalence, error, and known values.

Covers the contract both tests share through :class:`PartialCorrelationTest`: exact agreement of
the scalar, batched, many-query, canonical, all-pairs, and first-order paths, argument-order
independence, no per-query state retained, and fail-fast error paths. :class:`FisherZ` adds an
independently derived closed-form p-value for the empty conditioning set and a small inline
baseline pinning current outputs. The numpy path is pinned as the exact float64 oracle.
:class:`StudentT` reports the same partial correlation with a Student-t p-value, checked against a
direct scipy computation.
"""

from __future__ import annotations

import re
import sys
from itertools import combinations

import numpy as np
import pytest
from scipy import stats
from scipy.special import ndtr

import andrey
from andrey.core import ci
from andrey.core.ci import FisherZ, NormalScores, StudentT


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
    # and the p-value of row i equal the scalar answer for that row, bit for bit, and so does the
    # canonical path's p-value given the row's pair ordered and its set sorted.
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
        canonical = test._canonical_pvalues(
            np.minimum(x, y), np.maximum(x, y), np.sort(conditions, axis=1)
        )
        scalar = [test.statistic_pvalue(*row) for row in rows]
        np.testing.assert_array_equal(statistics, [r for r, _ in scalar])
        np.testing.assert_array_equal(pvalues, [p for _, p in scalar])
        np.testing.assert_array_equal(canonical, [p for _, p in scalar])


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


@pytest.mark.parametrize("test_class", PARTIAL_CORRELATION_TESTS)
def test_the_residual_path_gives_the_inverse_paths_partial_correlation(test_class, data):
    # On a well-conditioned design both paths compute the same residual correlation.
    test = test_class(data)
    x, y = np.array([0, 1, 2]), np.array([3, 4, 5])
    conditions = np.array([[1, 2], [0, 5], [1, 3]])
    r, p = test.statistic_pvalues(x, y, conditions)
    r_res, p_res = test.residual_statistic_pvalues(data, x, y, conditions)
    np.testing.assert_allclose(r_res, r, rtol=0, atol=1e-12)
    np.testing.assert_allclose(p_res, p, rtol=1e-9)


@pytest.mark.parametrize("test_class", PARTIAL_CORRELATION_TESTS)
def test_the_residual_path_answers_a_query_whose_conditions_are_collinear(test_class):
    # A conditioning column that repeats another makes the submatrix singular; the residuals are
    # those of the fit on the distinct columns, so the partial correlation is that one's, and the
    # degrees of freedom still count every conditioning column.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 5))
    X[:, 1] += 0.5 * X[:, 0]
    X[:, 4] = 2.0 * X[:, 0] + 1.0  # an affine copy of column 0
    test = test_class(X)
    with pytest.raises(ci.SingularCorrelationError):
        test.statistic_pvalues(np.array([1]), np.array([2]), np.array([[0, 4]]))
    r, p = test.residual_statistic_pvalues(X, np.array([1]), np.array([2]), np.array([[0, 4]]))
    r_one, _ = test.statistic_pvalues(np.array([1]), np.array([2]), np.array([[0]]))
    np.testing.assert_allclose(r, r_one, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(p, test._pvalue(r, test._degrees_of_freedom(2)))


def test_the_residual_path_refuses_a_column_in_the_span_of_its_conditions():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 4))
    X[:, 0] = X[:, 2] - 3.0 * X[:, 3]
    test = StudentT(X)
    with pytest.raises(ci.SingularCorrelationError, match="span"):
        test.residual_statistic_pvalues(X, np.array([0]), np.array([1]), np.array([[2, 3]]))


def test_a_singular_submatrix_raises_the_singular_error_type():
    # Callers catch the type to answer such a query another way; it stays a ValueError.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    X[:, 4] = X[:, 0]
    with pytest.raises(ci.SingularCorrelationError):
        StudentT(X).statistic_pvalues(np.array([1]), np.array([2]), np.array([[0, 4]]))
    assert issubclass(ci.SingularCorrelationError, ValueError)


@pytest.mark.parametrize("test_class", PARTIAL_CORRELATION_TESTS)
def test_a_singular_submatrix_from_copies_names_every_group(test_class):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 8))
    X[:, 2] = 2.0 * X[:, 1] + 1.0
    X[:, 7] = -X[:, 1]
    X[:, 5] = 3.0 * X[:, 3] - 2.0
    test = test_class(X)
    for query in (lambda: test(0, 4, [1, 2]), lambda: test.batched_call(0, 4, [[1, 2], [3, 6]])):
        with pytest.raises(ci.SingularCorrelationError) as error:
            query()
        assert error.value.groups == ((1, 2, 7), (3, 5))
        assert re.search(r"\b2 and 7 repeat 1\b", str(error.value))
        assert re.search(r"\b5 repeats 3\b", str(error.value))
        assert "keep one column" in str(error.value)


def test_a_singular_submatrix_without_copies_names_no_group():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    X[:, 4] = X[:, 2] - X[:, 3]
    with pytest.raises(ci.SingularCorrelationError, match="singular") as error:
        StudentT(X)(0, 1, [2, 3, 4])
    assert error.value.groups == ()


def test_pc_names_copies_by_column_name_before_a_test_meets_them():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    b = x + rng.standard_normal(1000)
    y = x + b + rng.standard_normal(1000)
    frame = pd.DataFrame({"x": x, "b": b, "y": y, "b2": 2.0 * b + 1.0})
    with pytest.raises(ValueError, match=r"'b2' repeats 'b'"):
        andrey.pc(frame)


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


# ---- singularity guard -------------------------------------------------------------------


def _equicorrelation(size: int, smallest: float) -> np.ndarray:
    """A unit-diagonal correlation matrix whose smallest eigenvalue is ``smallest``.

    ``(1 - a) I + a J`` has the eigenvalue ``1 - a`` (``size - 1`` times) and ``1 + a (size - 1)``.
    """
    a = -(1.0 - smallest) / (size - 1)
    return (1.0 - a) * np.eye(size) + a * np.ones((size, size))


def _eigenvalue_checks(monkeypatch) -> list[np.ndarray]:
    """Record every stack the guard hands to its eigenvalue test."""
    seen: list[np.ndarray] = []
    check = ci._raise_if_singular

    def spy(stack):
        seen.append(stack.copy())
        check(stack)

    monkeypatch.setattr(ci, "_raise_if_singular", spy)
    return seen


def _eigenvalues_first(corr, x, y, conditions):
    """Whether the eigenvalue test rejects some query row, else the rows' partial correlations."""
    var = np.column_stack([np.minimum(x, y), np.maximum(x, y), np.sort(conditions, axis=1)])
    stack = corr[var[:, :, None], var[:, None, :]]
    w = np.linalg.eigvalsh(stack)
    if (w[:, 0] <= w[:, -1] * 1e-12).any():
        return True, None
    inv = np.linalg.inv(stack)
    r = -inv[:, 0, 1] / np.sqrt(np.abs(inv[:, 0, 0] * inv[:, 1, 1]))
    return False, np.clip(r, -ci._ONE_MINUS_EPS, ci._ONE_MINUS_EPS)


@pytest.mark.parametrize("test_class", PARTIAL_CORRELATION_TESTS)
@pytest.mark.parametrize("kind", ["duplicate", "combination"])
@pytest.mark.parametrize("scale", [0.0, 1e-13, 1e-9, 1e-5, 1e-3])
def test_guard_raises_exactly_when_the_eigenvalues_find_a_singular_row(test_class, kind, scale):
    # A column that duplicates another or combines three others, plus noise of relative size
    # `scale`: exactly singular (an inversion that raises, or one that returns huge finite values),
    # rejected, or near-singular but accepted. Small batches mix singular and healthy rows. A batch
    # raises the singular ValueError, never a LinAlgError, exactly when the eigenvalue test rejects
    # one of its rows; otherwise it returns the partial correlations of the inverse. Each row on
    # the scalar path decides on its own.
    rng = np.random.default_rng(7)
    X = rng.standard_normal((150, 7))
    base = X[:, 2] if kind == "duplicate" else X[:, 1] + 2.0 * X[:, 3] - 0.5 * X[:, 5]
    noisy = base + scale * np.std(base) * rng.standard_normal(150)
    test = test_class(np.column_stack([X, noisy]))
    corr = test.to_seed()[0]
    outcomes = set()
    for size in (2, 3, 4):
        for _ in range(15):
            rows = np.array([rng.permutation(8)[: size + 2] for _ in range(4)])
            x, y, conditions = rows[:, 0], rows[:, 1], rows[:, 2:]
            raises, r = _eigenvalues_first(corr, x, y, conditions)
            outcomes.add(raises)
            if raises:
                with pytest.raises(ValueError, match="singular") as info:
                    test.statistic_pvalues(x, y, conditions)
                assert type(info.value) is ci.SingularCorrelationError
            else:
                np.testing.assert_array_equal(test.statistic_pvalues(x, y, conditions)[0], r)
            for row in rows:
                row_raises, row_r = _eigenvalues_first(corr, row[:1], row[1:2], row[None, 2:])
                if row_raises:
                    with pytest.raises(ValueError, match="singular") as info:
                        test.statistic_pvalue(row[0], row[1], row[2:])
                    assert type(info.value) is ci.SingularCorrelationError
                else:
                    assert test.statistic_pvalue(row[0], row[1], row[2:])[0] == row_r[0]
    assert outcomes == ({True, False} if scale <= 1e-9 else {False})


def test_guard_settles_healthy_rows_without_eigenvalues(monkeypatch, data):
    seen = _eigenvalue_checks(monkeypatch)
    test = StudentT(data)
    rng = np.random.default_rng(2)
    for size in (2, 3, 4):
        rows = np.array([rng.permutation(6)[: size + 2] for _ in range(30)])
        test.statistic_pvalues(rows[:, 0], rows[:, 1], rows[:, 2:])
        test.statistic_pvalue(rows[0, 0], rows[0, 1], rows[0, 2:])
    assert seen == []


@pytest.mark.parametrize("test_class", PARTIAL_CORRELATION_TESTS)
def test_guard_bound_edge(test_class, monkeypatch):
    # The largest entry of the equicorrelation inverse is about 1 / (size * smallest), so these two
    # matrices sit just inside and just outside the inverse bound. Only the outside one takes the
    # eigenvalue test; both pass it, since the bound stays far above the eigenvalue threshold.
    size = 4
    limit = 1.0 / (ci._BOUND_MARGIN * ci._SINGULAR_RTOL * size * size)
    edge = 1.0 / (size * limit)
    inside, outside = _equicorrelation(size, 1.1 * edge), _equicorrelation(size, 0.9 * edge)
    assert np.abs(np.linalg.inv(inside)).max() < limit < np.abs(np.linalg.inv(outside)).max()
    seen = _eigenvalue_checks(monkeypatch)
    stack = np.stack([inside, outside, inside])
    np.testing.assert_array_equal(ci._invert_nonsingular(stack), np.linalg.inv(stack))
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], outside[None])
    test_class.from_corr(inside, 100).statistic_pvalue(0, 1, (2, 3))
    assert len(seen) == 1
    test_class.from_corr(outside, 100).statistic_pvalue(0, 1, (2, 3))
    assert len(seen) == 2


@pytest.mark.parametrize("test_class", PARTIAL_CORRELATION_TESTS)
def test_guard_keeps_the_eigenvalue_threshold(test_class, monkeypatch):
    # Block-diagonal correlation: columns 0-3 are healthy, 4-7 have a smallest eigenvalue just above
    # the threshold, 8-11 just below it. Rows from the first two blocks pass; a batch holding a row
    # from the last block raises, whatever else it holds.
    size = 4
    largest = 1.0 + 1.0 / (size - 1)  # the largest eigenvalue, to within the smallest
    corr = np.zeros((3 * size, 3 * size))
    for block, smallest in enumerate((0.5, 2e-12 * largest, 0.5e-12 * largest)):
        span = slice(block * size, (block + 1) * size)
        corr[span, span] = _equicorrelation(size, smallest)
    test = test_class.from_corr(corr, 100)
    healthy, just_above, just_below = ([4 * b, 4 * b + 1, 4 * b + 2, 4 * b + 3] for b in range(3))
    seen = _eigenvalue_checks(monkeypatch)

    def query(*rows):
        rows = np.array(rows)
        return test.statistic_pvalues(rows[:, 0], rows[:, 1], rows[:, 2:])

    query(healthy, just_above, healthy)
    assert [len(stack) for stack in seen] == [1]
    for rows in [(just_below,), (healthy, just_below), (just_below, healthy, just_above)]:
        with pytest.raises(ValueError, match="singular"):
            query(*rows)
    with pytest.raises(ValueError, match="singular"):
        test.statistic_pvalue(just_below[0], just_below[1], just_below[2:])


def test_guard_decides_by_eigenvalues_whatever_the_inversion_returns(monkeypatch):
    # An exactly singular stack can make np.linalg.inv raise (an exactly zero pivot) or return the
    # huge inverse of a nearby nonsingular matrix, depending on the BLAS. Either way the guard
    # raises the singular ValueError; and when the inversion raises on a stack the eigenvalues
    # accept, the LinAlgError propagates.
    healthy = _equicorrelation(4, 0.5)
    singular = np.stack([healthy, _equicorrelation(4, 1e-15)])
    exact_inv = np.linalg.inv

    def raising_inv(a):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(np.linalg, "inv", raising_inv)
    with pytest.raises(ValueError, match="singular") as info:
        ci._invert_nonsingular(singular)
    assert type(info.value) is ci.SingularCorrelationError
    with pytest.raises(np.linalg.LinAlgError):
        ci._invert_nonsingular(healthy[None])

    monkeypatch.setattr(np.linalg, "inv", lambda a: exact_inv(a + 1e-14 * np.eye(a.shape[-1])))
    with pytest.raises(ValueError, match="singular"):
        ci._invert_nonsingular(singular)


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


def test_student_t_is_a_registered_indep_test(data):
    assert "student_t" in ci.available_indep_tests()
    assert "student_t" in ci.SHIPPED_INDEP_TESTS
    assert isinstance(ci.make_indep_test("student_t", data), StudentT)


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


def _latent_fork(rng, n, size):
    """Columns ``[x, y, *z]`` with ``x`` and ``y`` independent given the ``size`` columns ``z``."""
    z = rng.standard_normal((n, size))
    weights = rng.uniform(0.5, 1.0, (size, 2))
    x = z @ weights[:, 0] + rng.standard_normal(n)
    y = z @ weights[:, 1] + rng.standard_normal(n)
    return np.column_stack([x, y, z])


def test_normal_scores_p_values_do_not_change_under_increasing_transforms_of_columns():
    rng = np.random.default_rng(3)
    x = _latent_fork(rng, 150, 2)
    x[:, 1] += 0.4 * x[:, 0]
    transformed = np.column_stack([np.exp(x[:, 0]), x[:, 1] ** 3, 5.0 * x[:, 2] - 2.0, x[:, 3]])
    before, after = ci.NormalScores(x), ci.NormalScores(transformed)
    for a, b, cond in [(0, 1, ()), (0, 2, ()), (0, 1, (2,)), (0, 1, (2, 3)), (1, 3, (0, 2))]:
        assert after(a, b, cond) == before(a, b, cond)


def test_normal_scores_is_fisher_z_on_the_normal_scores_of_each_column():
    rng = np.random.default_rng(4)
    x = np.exp(_latent_fork(rng, 120, 2))
    x[:, 3] = np.round(x[:, 3])  # tied values share their average rank
    ranks = stats.rankdata(x, axis=0)
    scores = stats.norm.ppf(ranks / (len(x) + 1.0))
    for a, b, cond in [(0, 1, ()), (0, 1, (2,)), (0, 1, (2, 3))]:
        assert NormalScores(x)(a, b, cond) == pytest.approx(
            FisherZ(scores)(a, b, cond), rel=1e-12, abs=0
        )


def test_normal_scores_query_paths_return_the_scalar_p_values():
    rng = np.random.default_rng(5)
    x = np.exp(_latent_fork(rng, 200, 3))
    test = NormalScores(x)
    sets = [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4), ()]
    assert test.batched_call(0, 1, sets) == [test(0, 1, s) for s in sets]


@pytest.mark.parametrize("size", [0, 2])
@pytest.mark.parametrize("transform", [np.exp, lambda v: v**3], ids=["exp", "cubic"])
def test_normal_scores_rejects_a_true_independence_at_about_the_level_on_monotone_marginals(
    size, transform
):
    rng = np.random.default_rng(6)
    n_data, n_rows, level = 400, 200, 0.05
    rejected = sum(
        ci.NormalScores(transform(_latent_fork(rng, n_rows, size)))(0, 1, tuple(range(2, 2 + size)))
        < level
        for _ in range(n_data)
    )
    # Four binomial standard deviations around the level.
    tolerance = 4 * np.sqrt(level * (1 - level) / n_data)
    assert abs(rejected / n_data - level) < tolerance


def test_normal_scores_finds_a_monotone_dependence_that_pearson_correlation_dilutes():
    rng = np.random.default_rng(7)
    latent = rng.standard_normal((300, 1))
    x = np.column_stack(
        [np.exp(3 * latent[:, 0]), np.exp(3 * (0.4 * latent[:, 0] + rng.standard_normal(300)))]
    )
    assert NormalScores(x)(0, 1) < 1e-4 < FisherZ(x)(0, 1)


def test_normal_scores_shares_the_fisher_z_error_paths():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    X[:, 4] = X[:, 0]
    with pytest.raises(ValueError, match="singular"):
        NormalScores(X)(1, 2, (0, 4))
    with pytest.raises(ValueError, match="NaN or Inf"):
        NormalScores(np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]))
    with pytest.raises(ValueError, match="2-D"):
        NormalScores(np.arange(5.0))
    with pytest.raises(ValueError, match="degrees of freedom"):
        NormalScores(X[:3])(0, 1, (2,))
