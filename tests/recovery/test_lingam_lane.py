from __future__ import annotations

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


def test_ica_lingam_recovery():
    out = andrey.ica_lingam(datagen.generate("lingam_5v_uniform"))
    assert_recorded_output(out, "ICALiNGAM", case="lingam_5v_uniform")


def test_multi_group_direct_lingam_recovery():
    out = andrey.multi_group_direct_lingam(datagen.generate_groups("mgdl_2groups"))
    assert_recorded_output(out, "MultiGroupDirectLiNGAM", case="mgdl_2groups")


def test_ica_lingam_output_contract():
    out = andrey.ica_lingam(datagen.generate("lingam_5v_uniform"))
    assert out.ordering is not None and out.weighted_adjacency is not None


def test_multi_group_output_contract():
    out = andrey.multi_group_direct_lingam(datagen.generate_groups("mgdl_2groups"))
    assert len(out) == 2
    assert all(o.ordering == out[0].ordering for o in out)  # one shared order across groups


def test_multi_group_requires_at_least_two_groups():
    with pytest.raises(ValueError, match="at least two"):
        andrey.multi_group_direct_lingam([datagen.generate("lingam_5v_uniform")])


def test_multi_group_rejects_mismatched_features():
    a = datagen.generate("lingam_5v_uniform")
    with pytest.raises(ValueError, match="same number of features"):
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
    with pytest.raises(ValueError, match="NaN or Inf"):
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
