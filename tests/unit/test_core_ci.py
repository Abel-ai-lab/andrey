"""Fisher-Z conditional-independence test: equivalence, error, and known-value checks.

Covers the :class:`FisherZ` contract: exact scalar-vs-batched agreement,
argument-order independence, no per-query state retained, fail-fast error paths, an
independently derived closed-form p-value for the empty conditioning set, and a small inline
baseline pinning current outputs. The numpy path is pinned as the exact float64 oracle.
"""

from __future__ import annotations

import sys
from itertools import combinations

import numpy as np
import pytest
from scipy.special import ndtr

from andrey.core.ci import FisherZ


@pytest.fixture(autouse=True)
def force_numpy(monkeypatch):
    """numpy is the exact float64 oracle for the correlation matrix and every p-value."""
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


@pytest.fixture
def data():
    """A moderate Gaussian sample: enough rows to invert every submatrix up to |S| = 3."""
    rng = np.random.default_rng(3)
    return rng.standard_normal((200, 6))


# ---- scalar-vs-batched agreement --------------------------------------------------------


@pytest.mark.parametrize("seed", range(15))
def test_batched_matches_scalar_exactly(seed):
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

    scalar_test = FisherZ(data)
    batched_test = FisherZ(data)
    scalar_p: list[float] = []
    batched_p: list[float] = []
    for (x, y), conds in queries.items():
        scalar_p.extend(scalar_test(x, y, c) for c in conds)
        batched_p.extend(batched_test.batched_call(x, y, conds))

    np.testing.assert_array_equal(batched_p, scalar_p)


@pytest.mark.parametrize("seed", range(15))
def test_prescreen_pvalues_match_canonical_scalar_exactly(seed):
    data = np.random.default_rng(seed).standard_normal((200, 10))
    test = FisherZ(data)
    marginal = test.marginal_pvalues()
    for x, y in combinations(range(data.shape[1]), 2):
        assert marginal[x, y] == test(x, y)
        zs = [z for z in range(data.shape[1]) if z not in (x, y)]
        np.testing.assert_array_equal(
            test.first_order_pvalues(x, y, zs), [test(x, y, (z,)) for z in zs]
        )


def test_no_per_query_state_is_retained(data):
    # A skeleton search issues one query per conditioning subset it visits -- combinatorially many
    # at large d -- so the test object must hold exactly the same state after answering them as
    # before (the correlation matrix and the sample size). Any per-query memo would grow with the
    # subsets visited and take the search's memory with it.
    test = FisherZ(data)

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


def test_empty_batch_returns_empty_list(data):
    assert FisherZ(data).batched_call(0, 1, []) == []


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
