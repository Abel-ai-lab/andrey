"""The additive-model fit and the pruning built on it.

The thin plate basis spans the constant-free linear function, unpenalized; a zero smoothing
parameter gives least squares on the basis, an infinite one the linear regression; the GCV fit
recovers an additive model's parents and drops an independent variable; and every result is the
same on every run and for any order of the parents.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from andrey.fcm import gam
from andrey.fcm.pruning import prune


def _additive(seed: int, n: int = 300) -> np.ndarray:
    """Columns ``x0, x1, x2`` independent and ``y = sin(2 x0) + x1^2 / 2 + e`` last."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.4 * rng.standard_normal(n)
    return np.column_stack([X, y])


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    return _additive(0)


@pytest.fixture(scope="module")
def fit(data) -> gam.GAMFit:
    return gam.fit_gam(gam.BasisCache(data), 3, [0, 1, 2])


@pytest.mark.parametrize(("n", "max_knots"), [(200, 2000), (80, 50)])
def test_the_basis_is_centered_and_leaves_the_linear_function_unpenalized(n, max_knots):
    x = np.random.default_rng(1).uniform(-2.0, 2.0, n)
    b = gam.tp_basis(x, 10, max_knots=max_knots)
    assert b.X.shape == (n, 9) and b.S.shape == (9, 9) and b.root.shape == (8, 9)
    np.testing.assert_allclose(b.X.sum(axis=0), 0.0, atol=1e-9 * np.abs(b.X).max())
    np.testing.assert_allclose(b.root.T @ b.root, b.S, atol=1e-10 * np.abs(b.S).max())
    assert np.linalg.matrix_rank(b.S, tol=1e-9 * np.abs(b.S).max()) == 8
    beta = np.linalg.lstsq(b.X, x - x.mean(), rcond=None)[0]
    np.testing.assert_allclose(b.X @ beta, x - x.mean(), atol=1e-8)
    assert abs(beta @ b.S @ beta) < 1e-8 * np.abs(b.S).max()


def test_a_basis_has_at_most_one_function_per_distinct_value():
    rng = np.random.default_rng(4)
    data = np.column_stack([rng.integers(0, 5, 200).astype(float), rng.standard_normal(200)])
    cache = gam.BasisCache(data)
    assert cache(0, 10).X.shape == (200, 4) and cache(1, 10).X.shape == (200, 9)
    assert cache.n_coef([0, 1], 10) == 1 + 4 + 9
    with pytest.raises(ValueError):
        gam.BasisCache(np.column_stack([np.repeat([0.0, 1.0], 100), data[:, 1]]))(0, 10)


def test_the_basis_needs_k_distinct_values():
    with pytest.raises(ValueError):
        gam.tp_basis(np.repeat(np.arange(5.0), 10), 10)
    with pytest.raises(ValueError):
        gam.tp_basis(np.arange(10.0), 2)


def test_zero_smoothing_is_least_squares_on_the_basis(data):
    cache = gam.BasisCache(data)
    bases = [cache(j, 10) for j in range(3)]
    f = gam.fit_additive(data[:, 3], bases, sp=np.zeros(3))
    design = np.column_stack([np.ones(len(data)), *(b.X for b in bases)])
    lstsq = design @ np.linalg.lstsq(design, data[:, 3], rcond=None)[0]
    np.testing.assert_allclose(f.fitted, lstsq, atol=1e-10)
    np.testing.assert_allclose(f.edf, 9.0, atol=1e-9)
    assert f.residual_df == pytest.approx(len(data) - 28.0)


def test_infinite_smoothing_is_the_linear_regression(data):
    cache = gam.BasisCache(data)
    f = gam.fit_additive(data[:, 3], [cache(j, 10) for j in range(3)], sp=np.full(3, 1e12))
    design = np.column_stack([np.ones(len(data)), data[:, :3]])
    linear = design @ np.linalg.lstsq(design, data[:, 3], rcond=None)[0]
    np.testing.assert_allclose(f.fitted, linear, atol=1e-6)
    np.testing.assert_allclose(f.edf, 1.0, atol=1e-6)


def test_gcv_fit_keeps_the_parents_and_drops_an_independent_variable(fit, data):
    p = fit.p_values()
    assert p.shape == (3,) and fit.edf.shape == (3,) and fit.fitted.shape == (len(data),)
    assert p[0] < 1e-10 and p[1] < 1e-10 and p[2] > 0.01
    assert fit.edf[0] > 3.0
    np.testing.assert_allclose(fit.residuals, data[:, 3] - fit.fitted)
    assert abs(fit.residuals.sum()) < 1e-8
    assert fit.scale == pytest.approx(fit.residuals @ fit.residuals / fit.residual_df)
    assert fit.gcv == pytest.approx(len(data) * fit.scale / fit.residual_df)


def test_gcv_search_keeps_the_lowest_of_its_starts(fit, data):
    cache = gam.BasisCache(data)
    bases = [cache(j, 10) for j in range(3)]
    shifts = (0.0, -6.0, 6.0)
    ends = [gam.fit_additive(data[:, 3], bases, starts=(s,)).gcv for s in shifts]
    assert gam.fit_additive(data[:, 3], bases, starts=shifts).gcv == min(ends)
    assert fit.gcv == ends[0]  # one start by default


def test_gcv_fit_is_a_local_minimum_of_the_score(fit, data):
    cache = gam.BasisCache(data)
    bases = [cache(j, 10) for j in range(3)]
    for j in range(2):  # the independent term sits at the bound, where the score is flat
        for factor in (0.8, 1.25):
            sp = fit.sp.copy()
            sp[j] *= factor
            assert gam.fit_additive(data[:, 3], bases, sp=sp).gcv > fit.gcv


def test_the_fit_is_deterministic_and_free_of_parent_order(fit, data):
    again = gam.fit_gam(gam.BasisCache(data), 3, [0, 1, 2])
    np.testing.assert_array_equal(again.fitted, fit.fitted)
    np.testing.assert_array_equal(again.p_values(), fit.p_values())
    reverse = gam.fit_gam(gam.BasisCache(data), 3, [2, 1, 0])
    np.testing.assert_allclose(reverse.fitted, fit.fitted, atol=1e-7)
    np.testing.assert_allclose(reverse.edf[::-1], fit.edf, atol=1e-6)
    np.testing.assert_allclose(reverse.p_values()[::-1][2], fit.p_values()[2], rtol=1e-5)


@pytest.mark.parametrize("unit", [1e-6, 1e6])
def test_the_fit_is_the_same_in_any_unit(fit, data, unit):
    scaled = data * np.array([unit, 1.0, 1.0, 1.0])
    again = gam.fit_gam(gam.BasisCache(scaled), 3, [0, 1, 2])
    np.testing.assert_allclose(again.fitted, fit.fitted, atol=1e-6)
    np.testing.assert_allclose(again.edf, fit.edf, atol=1e-5)


@pytest.mark.parametrize(
    ("n", "parents", "k"), [(1000, 5, 10), (200, 10, 7), (1000, 99, 4), (40, 15, 3)]
)
def test_the_rank_keeps_three_rows_per_basis_function(n, parents, k):
    assert gam.effective_k(n, parents, 10) == k


def test_a_duplicated_parent_splits_one_term_equally(data):
    duplicated = np.column_stack([data[:, 0], data[:, 0], data[:, 1], data[:, 3]])
    f = gam.fit_gam(gam.BasisCache(duplicated), 3, [0, 1, 2])
    assert f.edf[0] == pytest.approx(f.edf[1], rel=1e-6)
    assert f.p_values()[0] == pytest.approx(f.p_values()[1], rel=1e-6)
    single = gam.fit_gam(gam.BasisCache(duplicated), 3, [0, 2])
    np.testing.assert_allclose(f.fitted, single.fitted, atol=1e-4)


def test_a_failed_search_refits_without_penalty(data, monkeypatch):
    def fail(self, starts):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(gam._Problem, "optimize", fail)
    f = gam.fit_gam(gam.BasisCache(data), 3, [0, 1])
    np.testing.assert_array_equal(f.sp, 0.0)
    np.testing.assert_allclose(f.edf, 9.0, atol=1e-9)


# The last case's statistic is near zero: its slowly decaying integrand takes the quadrature path.
@pytest.mark.parametrize(
    ("d", "df", "tol"),
    [(0.5, 20, 1e-13), (12.0, 200, 1e-13), (30.0, 990, 1e-13), (1e-6, 200, 1e-9)],
)
def test_the_reference_tail_is_the_f_tail_for_one_unit_weight(d, df, tol):
    p = gam._quadratic_form_sf([1.0, -d / df], [1.0, float(df)])
    assert p == pytest.approx(stats.f.sf(d, 1, df), abs=tol)


def test_the_knot_subsample_is_fixed():
    x = np.random.default_rng(2).uniform(-2.0, 2.0, 80)
    first, second = gam.tp_basis(x, 10, max_knots=50), gam.tp_basis(x, 10, max_knots=50)
    np.testing.assert_array_equal(first.X, second.X)
    assert np.abs(first.X - gam.tp_basis(x, 10).X).max() > 1e-6  # a subset, not every value


def test_prune_keeps_the_parents_and_drops_the_spurious_edges(data):
    candidates = np.zeros((4, 4), dtype=int)
    candidates[[0, 1, 2], 3] = 1
    candidates[0, 1] = candidates[0, 2] = 1  # independent columns
    kept = prune(data, candidates)
    assert kept.shape == (4, 4) and kept.dtype == np.int64
    assert np.all(kept <= candidates)
    assert {tuple(e) for e in np.argwhere(kept)} == {(0, 3), (1, 3)}


def test_prune_reuses_bases_only_for_their_data(data):
    cache = gam.BasisCache(data)
    candidates = np.zeros((4, 4), dtype=int)
    candidates[[0, 1], 3] = 1
    np.testing.assert_array_equal(prune(data, candidates, bases=cache), prune(data, candidates))
    with pytest.raises(ValueError):
        prune(data + 1.0, candidates, bases=cache)
    with pytest.raises(ValueError):
        prune(data, np.eye(4, dtype=int))
