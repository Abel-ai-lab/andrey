from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey import data
from andrey.constraint.pc import pc as pc_engine
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    # The BIC/Fisher-Z primitives route through core.stats; pin the numpy oracle so an ambient
    # ANDREY_DEVICE=cpu cannot compare torch numerics against the numpy baselines.
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


_BASELINE_LAMBDA = 2.0


def test_pc_recovery():
    assert_recorded_output(andrey.pc(datagen.generate("gauss_5v")), "PC", case="gauss_5v")


def test_pc_is_invariant_to_column_order():
    # A seed on which colliders conflict, so a higher-ranked collider makes PC skip another.
    x = data.sample_scm(d=20, n=200, seed=0, density=2.0, scale="standardize").data
    trace: list[dict] = []
    pc_engine(x, _trace=trace)
    assert any(
        r["step"] == "collider" and not r["oriented"] and r["triple"][1] not in r["sepset"]
        for r in trace
    )
    marks = andrey.pc(x).structure.to_numpy()
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(x.shape[1])
        back = np.argsort(order)
        permuted = andrey.pc(x[:, order]).structure.to_numpy()
        assert np.array_equal(permuted[np.ix_(back, back)], marks)


def test_ges_recovery():
    assert_recorded_output(
        andrey.ges(datagen.generate("gauss_5v"), lambda_value=_BASELINE_LAMBDA),
        "GES",
        case="gauss_5v",
    )


def test_pc_output_contract():
    out = andrey.pc(datagen.generate("gauss_5v"))
    assert out.structure.kind == "cpdag"


def test_ges_output_contract():
    out = andrey.ges(datagen.generate("gauss_5v"))
    assert out.structure.kind == "cpdag" and "score" in out.metadata


def test_pc_rejects_non_fisherz():
    with pytest.raises(NotImplementedError):
        andrey.pc(datagen.generate("gauss_5v"), indep_test="kci")


def test_ges_rejects_unknown_score():
    with pytest.raises(NotImplementedError):
        andrey.ges(datagen.generate("gauss_5v"), score_func="local_score_CV_general")


def test_pc_rejects_bad_alpha():
    with pytest.raises(ValueError, match="alpha"):
        andrey.pc(datagen.generate("gauss_5v"), alpha=1.5)
