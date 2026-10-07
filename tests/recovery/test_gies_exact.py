from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.metrics import markov_equivalent
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output, assert_recorded_output_mec


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


_BASELINE_LAMBDA = 2.0


def test_gies_recovery():
    assert_recorded_output(
        andrey.gies(datagen.generate("gauss_5v"), lambda_value=_BASELINE_LAMBDA),
        "GIES",
        case="gauss_5v",
    )


def test_exact_search_recovery():
    assert_recorded_output_mec(
        andrey.exact_search(datagen.generate("gauss_5v")),
        "ExactSearch",
        case="gauss_5v",
    )


def test_gies_output_contract():
    out = andrey.gies(datagen.generate("gauss_5v"))
    assert out.structure.kind == "cpdag" and "score" in out.metadata


def test_exact_search_output_contract():
    assert andrey.exact_search(datagen.generate("gauss_5v")).structure.kind == "dag"


@pytest.mark.parametrize("search_method", ["astar", "dp"])
def test_exact_search_ignores_the_variables_means(search_method):
    """Independent columns have no edges whatever their means, and a shift keeps the class."""
    rng = np.random.default_rng(0)
    independent = rng.normal(5.0, 1.0, size=(500, 3))
    found = andrey.exact_search(independent, search_method=search_method).structure
    assert found.oriented_edges() == []
    # A shift can change which score-tied DAG is returned, not its Markov equivalence class.
    x = datagen.generate("gauss_5v")
    centered = andrey.exact_search(x, search_method=search_method).structure
    shifted = andrey.exact_search(x + 7.0, search_method=search_method).structure
    assert markov_equivalent(shifted, centered)


def test_gfci_recovery():
    out = andrey.gfci(datagen.generate("gauss_5v"), lambda_value=_BASELINE_LAMBDA)
    assert_recorded_output(out, "GFCI", case="gauss_5v")


def test_gfci_output_contract():
    assert andrey.gfci(datagen.generate("gauss_5v")).structure.kind == "pag"


def test_gfci_rejects_non_fisherz():
    with pytest.raises(NotImplementedError):
        andrey.gfci(datagen.generate("gauss_5v"), indep_test="kci")
