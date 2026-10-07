from __future__ import annotations

import numpy as np
import pytest

import andrey
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


_BASELINE_LAMBDA = 2.0


def test_hc_recovery():
    assert_recorded_output(andrey.hc(datagen.generate("gauss_5v")), "HC", case="gauss_5v")


def test_boss_recovery():
    assert_recorded_output(
        andrey.boss(datagen.generate("gauss_5v"), lambda_value=_BASELINE_LAMBDA),
        "BOSS",
        case="gauss_5v",
    )


def test_grasp_recovery():
    out = andrey.grasp(datagen.generate("gauss_5v"), lambda_value=_BASELINE_LAMBDA)
    assert_recorded_output(out, "GRaSP", case="gauss_5v")


def test_hc_output_contract():
    out = andrey.hc(datagen.generate("gauss_5v"))
    assert out.structure.kind == "cpdag" and "score" in out.metadata


def test_boss_grasp_output_contract():
    x = datagen.generate("gauss_5v")
    assert andrey.boss(x).structure.kind == "cpdag"
    assert andrey.grasp(x).structure.kind == "cpdag"


def test_boss_grasp_deterministic():
    x = datagen.generate("gauss_5v")
    assert np.array_equal(andrey.boss(x).structure.to_numpy(), andrey.boss(x).structure.to_numpy())
    assert np.array_equal(
        andrey.grasp(x).structure.to_numpy(), andrey.grasp(x).structure.to_numpy()
    )


def test_boss_equal_scores_are_stable_under_input_rounding():
    # Four centered rows give every variable variance 18 and every pair covariance 15.
    # Each single parent has the same BIC; a second parent does not improve it.
    x = np.array([[6, 6, 6], [0, -3, -3], [-3, 0, -3], [-3, -3, 0]], dtype=float)
    rng = np.random.default_rng(10_000)
    perturbed = x * (1 + 1e-12 * rng.standard_normal(x.shape))
    base = andrey.boss(x, seed=0).structure
    moved = andrey.boss(perturbed, seed=0).structure

    assert base.kind == moved.kind == "cpdag"
    # Both inputs give the undirected edges 0 -- 1 and 0 -- 2.
    expected = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(base.to_numpy(), expected)
    np.testing.assert_array_equal(moved.to_numpy(), expected)


def test_hc_rejects_unknown_score():
    with pytest.raises(NotImplementedError):
        andrey.hc(datagen.generate("gauss_5v"), score_func="local_score_BDeu")
