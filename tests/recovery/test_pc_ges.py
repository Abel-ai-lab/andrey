from __future__ import annotations

import numpy as np
import pytest

import andrey
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


def test_ges_recovery():
    assert_recorded_output(
        andrey.ges(datagen.generate("gauss_5v"), lambda_value=_BASELINE_LAMBDA),
        "GES",
        case="gauss_5v",
    )


def test_ges_does_not_depend_on_the_data_scale():
    """GES returns the same CPDAG after the data is rescaled so far that squared covariances
    underflow."""
    x = datagen.generate("gauss_5v")
    want = andrey.ges(x).structure.to_numpy()
    np.testing.assert_array_equal(andrey.ges(x * 1e-85).structure.to_numpy(), want)


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
