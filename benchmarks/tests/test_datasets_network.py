"""A published network's rows, stored and keyed like any generated dataset."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from andrey.data import load_dataset
from andrey_bench import datasets

ASIA = Path(__file__).resolve().parents[2] / "tests" / "unit" / "fixtures" / "asia.bif.gz"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A data cache holding the asia network's file, so no load reaches the network."""
    (tmp_path / "cache" / "asia").mkdir(parents=True)
    shutil.copy(ASIA, tmp_path / "cache" / "asia" / "asia.bif.gz")
    monkeypatch.setenv("ANDREY_DATA_DIR", str(tmp_path / "cache"))


def test_a_network_dataset_holds_the_loaded_rows_and_graph(cache, tmp_path):
    key = datasets.network_key("asia", n=300, seed=1)
    registry = datasets.materialize([key], tmp_path / "store")
    data, marks = datasets.load(tmp_path / "store", key)
    loaded = load_dataset("asia", n=300, seed=1, discrete=True, as_frame=False)
    assert key.d == len(loaded.feature_names)
    np.testing.assert_array_equal(data, loaded.data)
    np.testing.assert_array_equal(marks, loaded.graph.to_numpy())
    # Column j holds the file's column column_order[j].
    columns = registry["keys"][key.label]["column_order"]
    plain = load_dataset("asia", n=300, seed=1, discrete=True, shuffle=False, as_frame=False)
    np.testing.assert_array_equal(data, plain.data[:, columns])


def test_network_keys_differ_by_seed_rows_and_order(cache):
    keys = [
        datasets.network_key("asia", 300, 0),
        datasets.network_key("asia", 300, 1),
        datasets.network_key("asia", 500, 0),
        datasets.network_key("asia", 300, 0, shuffle=False),
    ]
    assert len({key.digest for key in keys}) == len(keys)
    assert len({key.label for key in keys}) == len(keys)
