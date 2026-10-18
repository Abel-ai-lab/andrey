"""The mixed-network sampler: categorical columns first in causal order, seeded, validated."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.data import sample_mixed


def test_categorical_columns_hold_codes_and_have_only_categorical_parents():
    ds = sample_mixed(d=12, n=500, seed=1, categorical_share=0.5, categories=(2, 4))
    config = ds.provenance["config"]
    columns, cards = config["categorical"], config["categories"]
    assert len(columns) == 6 and all(2 <= k <= 4 for k in cards)
    for column, k in zip(columns, cards):
        assert set(np.unique(ds.data[:, column])) <= set(range(k))
    edges = ds.params.edges
    into_categorical = np.isin(edges[:, 1], columns)
    assert np.isin(edges[into_categorical, 0], columns).all()
    assert np.isnan(ds.params.weights[into_categorical]).all()
    assert np.isfinite(ds.params.weights[~into_categorical]).all()


def test_the_same_seed_gives_the_same_data():
    first, second = sample_mixed(d=6, n=100, seed=7), sample_mixed(d=6, n=100, seed=7)
    assert np.array_equal(first.data, second.data)
    assert not np.array_equal(first.data, sample_mixed(d=6, n=100, seed=8).data)


@pytest.mark.parametrize("share", [0.0, 1.0])
def test_the_share_of_categorical_columns_runs_from_none_to_all(share):
    ds = sample_mixed(d=5, n=50, seed=0, categorical_share=share)
    assert len(ds.provenance["config"]["categorical"]) == round(5 * share)


@pytest.mark.parametrize(
    "settings",
    [
        {"categorical_share": 1.5},
        {"categories": (1, 3)},
        {"categories": (3, 2)},
        {"concentration": 0.0},
    ],
)
def test_bad_settings_raise(settings):
    with pytest.raises(ValueError):
        sample_mixed(d=5, n=50, seed=0, **settings)
