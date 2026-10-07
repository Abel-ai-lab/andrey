"""Benchmark-registry tests: every named preset builds valid, finite data."""

from __future__ import annotations

import numpy as np
import pytest

import andrey.data as data


def test_names_nonempty():
    assert set(data.benchmarks.names()) >= {"linear_gauss_er", "lingam_sf", "nonlinear_hub"}


@pytest.mark.parametrize("name", data.benchmarks.names())
def test_each_benchmark_generates_valid_data(name):
    from andrey.core.structure import TemporalStructure

    scm = data.benchmarks.scm(name, d=20)
    ds = scm.sample(n=200, seed=0)
    assert np.all(np.isfinite(ds.data))
    if scm.functional.is_temporal:
        assert isinstance(ds.graph, TemporalStructure)
        assert ds.graph.n_lags >= 1
    else:
        ds.graph.validate()
        assert ds.graph.kind == "dag"


def test_unknown_benchmark_raises():
    with pytest.raises(KeyError, match="unknown benchmark"):
        data.benchmarks.get("nope")
