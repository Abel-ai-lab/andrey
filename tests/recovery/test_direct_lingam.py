from __future__ import annotations

import numpy as np
import pytest

import andrey
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output


def _random_linear_sem(seed: int, n: int, d: int, noise: str) -> np.ndarray:
    """A random strictly-lower-triangular linear SEM (uniform, or near-tie Gaussian, errors)."""
    rng = np.random.default_rng(seed)
    B = np.tril(rng.uniform(0.3, 1.2, (d, d)) * rng.choice([-1.0, 1.0], (d, d)), -1)
    e = rng.uniform(-1.0, 1.0, (n, d)) if noise == "uniform" else rng.standard_normal((n, d))
    X = np.zeros((n, d))
    for j in range(d):
        X[:, j] = e[:, j] + X @ B[j]
    return X


def test_direct_lingam_recovery():
    out = andrey.direct_lingam(datagen.generate("lingam_5v_uniform"))
    assert_recorded_output(out, "DirectLiNGAM", case="lingam_5v_uniform")


def test_direct_lingam_output_contract():
    out = andrey.direct_lingam(datagen.generate("lingam_5v_uniform"))
    assert out.ordering is not None and out.weighted_adjacency is not None


def test_direct_lingam_rejects_kernel():
    with pytest.raises(NotImplementedError):
        andrey.direct_lingam(datagen.generate("lingam_5v_uniform"), measure="kernel")


def test_pwling_serial_and_vectorized_agree():
    # Code-path equivalence on the shared pwling foundation: the batched scoring must pick the
    # same causal order as the readable per-pair reference -- float associativity in the batched
    # mean stays below any order-flipping gap. Guards the acceleration against a silent order flip.
    from andrey.lingam import pwling

    X = datagen.generate("lingam_5v_uniform")
    vectorized = pwling.find_causal_order(X, scores=pwling.score_candidates)
    reference = pwling.find_causal_order(X, scores=pwling.score_candidates_serial)
    assert vectorized == reference


@pytest.mark.parametrize("d", [8, 20])
@pytest.mark.parametrize("noise", ["uniform", "gaussian"])
def test_pwling_serial_vectorized_property(d: int, noise: str):
    # Beyond the single baseline: batched and serial pwling must agree on random SEMs up to d=20,
    # including near-tie Gaussian inputs (where pwling scores cluster) -- the FP-equivalence guard
    # that a one-input test can't give. `_random_linear_sem` is seeded, so this is reproducible.
    from andrey.lingam import pwling

    X = _random_linear_sem(seed=0, n=600, d=d, noise=noise)
    vectorized = pwling.find_causal_order(X, scores=pwling.score_candidates)
    reference = pwling.find_causal_order(X, scores=pwling.score_candidates_serial)
    assert vectorized == reference


def test_score_candidates_single_candidate():
    # The foundation must survive a lone candidate (MultiGroup calls it directly in its peel loop);
    # DirectLiNGAM is shielded by causal_order's guard, but the primitive must not crash.
    from andrey.lingam import pwling

    scores = pwling.score_candidates(_random_linear_sem(seed=1, n=200, d=3, noise="uniform"), [1])
    assert scores.shape == (1,)
