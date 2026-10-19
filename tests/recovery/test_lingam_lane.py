from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

import andrey
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output


def _linear_sem_group(seed: int, n: int, B: np.ndarray) -> np.ndarray:
    """Sample ``n`` rows from a strictly-lower-triangular linear SEM (true order 0..d-1)."""
    rng = np.random.default_rng(seed)
    d = B.shape[0]
    e = rng.uniform(-1.0, 1.0, (n, d))
    X = np.zeros((n, d))
    for j in range(d):
        X[:, j] = e[:, j] + X @ B[j]
    return X


def test_adaptive_lasso_refit_is_least_squares_on_selected_parents():
    from sklearn.linear_model import LinearRegression

    from andrey.lingam.adjacency import predict_adaptive_lasso

    B = np.zeros((4, 4))
    B[2, 0] = 0.8
    B[3, 0] = 0.5
    B[3, 2] = -0.6
    X = _linear_sem_group(0, 500, B) + 3.0  # an offset the refit's intercept absorbs
    shrunk = predict_adaptive_lasso(X, [0, 1, 2], 3)
    refit = predict_adaptive_lasso(X, [0, 1, 2], 3, refit=True)

    assert np.flatnonzero(shrunk).tolist() == [0, 2]
    assert np.flatnonzero(refit).tolist() == [0, 2]
    ols = LinearRegression().fit(X[:, [0, 2]], X[:, 3]).coef_
    np.testing.assert_allclose(refit[[0, 2]], ols, rtol=0, atol=1e-12)


def test_ica_lingam_recovery():
    out = andrey.ica_lingam(datagen.generate("lingam_5v_uniform"))
    assert_recorded_output(out, "ICALiNGAM", case="lingam_5v_uniform")


@pytest.fixture(scope="module")
def mgdl_fit() -> list:
    return andrey.multi_group_direct_lingam(datagen.generate_groups("mgdl_2groups"))


def test_multi_group_direct_lingam_recovery(mgdl_fit):
    assert_recorded_output(mgdl_fit, "MultiGroupDirectLiNGAM", case="mgdl_2groups")


def test_multi_group_recovers_each_groups_planted_weights(mgdl_fit):
    """Each group gets its own planted edges and weights, under one order the planted DAGs allow."""
    order = datagen.column_order("mgdl_2groups")
    for group, out in zip(datagen.GROUP_CASES["mgdl_2groups"]["groups"], mgdl_fit, strict=True):
        planted = np.asarray(group["weights"]).T[np.ix_(order, order)]  # weights column i -> j
        position = {v: k for k, v in enumerate(out.ordering)}
        assert all(position[i] < position[j] for i, j in zip(*np.nonzero(planted), strict=True))
        estimated = np.asarray(out.weighted_adjacency)
        np.testing.assert_array_equal(estimated != 0, planted != 0)
        np.testing.assert_allclose(estimated, planted, atol=0.15)


def test_multi_group_finds_no_edge_between_independent_columns():
    rng = np.random.default_rng(0)
    groups = [rng.uniform(-1.0, 1.0, (500, 4)), rng.uniform(-1.0, 1.0, (300, 4))]
    for out in andrey.multi_group_direct_lingam(groups):
        assert out.structure.oriented_edges() == []
        assert not np.any(out.weighted_adjacency)


def test_ica_lingam_output_contract():
    out = andrey.ica_lingam(datagen.generate("lingam_5v_uniform"))
    assert out.ordering is not None and out.weighted_adjacency is not None


def test_multi_group_output_contract(mgdl_fit):
    assert len(mgdl_fit) == 2
    assert all(o.ordering == mgdl_fit[0].ordering for o in mgdl_fit)  # one shared order


def test_multi_group_requires_at_least_two_groups():
    with pytest.raises(ValueError, match="at least two"):
        andrey.multi_group_direct_lingam([datagen.generate("lingam_5v_uniform")])


def test_multi_group_rejects_mismatched_features():
    a = datagen.generate("lingam_5v_uniform")
    with pytest.raises(ValueError, match="data_groups needs the same columns"):
        andrey.multi_group_direct_lingam([a, a[:, :-1]])


def test_multi_group_unequal_size_groups_recovers_shared_order():
    # Unequal groups exercise size weighting (n_g / total_n). Both groups must recover the
    # shared order 0..d-1 from a strictly lower-triangular SEM, deterministically.
    rng = np.random.default_rng(0)
    d = 4
    B = np.tril(rng.uniform(0.5, 1.0, (d, d)) * rng.choice([-1.0, 1.0], (d, d)), -1)
    groups = [_linear_sem_group(1, 600, B), _linear_sem_group(2, 250, B)]
    out = andrey.multi_group_direct_lingam(groups)
    assert [list(o.ordering) for o in out] == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert list(andrey.multi_group_direct_lingam(groups)[0].ordering) == list(out[0].ordering)


def test_multi_group_rejects_non_2d_and_nonfinite_groups():
    good = datagen.generate("lingam_5v_uniform")
    with pytest.raises(ValueError, match="2-D"):
        andrey.multi_group_direct_lingam([good, good[:, 0]])  # a 1-D group
    nan = good.copy()
    nan[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"data_groups\[1\] holds NaN or inf"):
        andrey.multi_group_direct_lingam([good, nan])


def test_ica_lingam_random_state_and_max_iter_accepted():
    # random_state defaults to 0 (via resolve_seed) and is honored; max_iter is accepted. The
    # returned B is the adaptive-Lasso adjacency on the ICA-derived order, so the output is
    # seed-robust on clean data by construction -- this asserts the plumbing (default == explicit-0,
    # reproducible) rather than a seed-dependent output that does not exist.
    X = datagen.generate("lingam_5v_uniform")
    default = andrey.ica_lingam(X)
    explicit0 = andrey.ica_lingam(X, random_state=0)
    assert list(default.ordering) == list(explicit0.ordering)
    np.testing.assert_array_equal(default.weighted_adjacency, explicit0.weighted_adjacency)
    other = andrey.ica_lingam(X, random_state=123, max_iter=500)
    assert list(andrey.ica_lingam(X, random_state=123, max_iter=500).ordering) == list(
        other.ordering
    )


def _ica_in_original_labels(X: np.ndarray, order: tuple[int, ...]) -> tuple[list[int], np.ndarray]:
    """Fit ``ica_lingam`` on the columns in ``order``; return its ordering and weights relabeled."""
    out = andrey.ica_lingam(X[:, order], random_state=0)
    weights = np.zeros_like(out.weighted_adjacency)
    weights[np.ix_(order, order)] = out.weighted_adjacency
    return [order[k] for k in out.ordering], weights


def test_ica_lingam_tied_sources_follow_the_columns(monkeypatch):
    """Two independent sources can both come first; reordering the columns keeps the choice."""
    rng = np.random.default_rng(0)
    a, b = rng.uniform(-1, 1, 2000), rng.uniform(-1, 1, 2000)
    X = np.column_stack([a, b, 0.8 * a + 0.8 * b + rng.uniform(-1, 1, 2000)])
    orders = list(permutations(range(3)))
    want = _ica_in_original_labels(X, orders[0])
    for order in orders[1:]:
        got = _ica_in_original_labels(X, order)
        assert got[0] == want[0]
        np.testing.assert_array_equal(got[1] != 0, want[1] != 0)
        np.testing.assert_allclose(got[1], want[1], rtol=0, atol=1e-6)

    import andrey.lingam.ica as ica

    monkeypatch.setattr(ica, "tie_ranks", lambda x: np.arange(x.shape[1]))
    by_position = [_ica_in_original_labels(X, order)[0] for order in orders]
    assert any(o != by_position[0] for o in by_position[1:])
