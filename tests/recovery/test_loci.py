from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from andrey.api.irregular import loci
from andrey.data import sample_scm
from tests.recovery.helpers import datagen
from tests.recovery.helpers.harness import assert_recorded_output

CRITERIA = ("independence", "likelihood")


def test_loci_matches_the_recorded_output():
    out = loci(datagen.generate("lsnm_pair"))
    assert_recorded_output(out, "LOCI", case="lsnm_pair")


def test_independent_columns_give_no_edge_at_a_level():
    rng = np.random.default_rng(3)
    out = loci(rng.standard_normal((300, 2)), alpha=0.05)
    assert out.structure.oriented_edges() == []
    assert min(out.metadata["p_values"].values()) > 0.05


@pytest.mark.parametrize(("criterion", "at_least"), [("independence", 8), ("likelihood", 9)])
def test_loci_names_the_direction_of_location_scale_pairs(criterion, at_least):
    # Ten pairs of the paper's model: a Gaussian cause, a noise scale that changes with it.
    right = 0
    for seed in range(10):
        dataset = sample_scm(
            functional="location_scale", d=2, n=500, seed=seed, density=1.0, scale="rescale"
        )
        edges = loci(dataset.data, criterion=criterion).structure.oriented_edges()
        right += edges == dataset.graph.oriented_edges()
    assert right >= at_least


@pytest.mark.parametrize("criterion", CRITERIA)
def test_loci_names_the_direction_in_any_units_and_column_order(criterion):
    names = [("cause", "effect")[v] for v in datagen.column_order("lsnm_pair")]
    planted = pd.DataFrame(datagen.generate("lsnm_pair"), columns=names)
    first = loci(planted, criterion=criterion)
    assert first.structure.oriented_edges() == [("cause", "effect", "directed")]
    for columns in (["cause", "effect"], ["effect", "cause"]):
        scaled = planted[columns] * np.array([2.0**-30, 2.0**20])  # exact products
        out = loci(scaled, criterion=criterion)
        assert out.structure.oriented_edges() == [("cause", "effect", "directed")]
        if columns == ["cause", "effect"]:
            assert out.metadata == first.metadata
