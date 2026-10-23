"""RFCI on finite data: fixed cases, ground truth, recorded output, and column order."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey.data import sample_scm
from andrey.data.latent import marginal
from andrey.metrics import shd
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output


@pytest.fixture(autouse=True)
def _quiet(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        yield


def test_rfci_recovery():
    assert_recorded_output(andrey.rfci(datagen.generate("gauss_5v")), "RFCI", case="gauss_5v")


def test_independent_columns_have_no_edges():
    data = np.random.default_rng(1).standard_normal((2000, 5))
    assert andrey.rfci(data).structure.oriented_edges() == []


def test_a_collider_and_an_independent_column():
    """``x -> z <- y`` with ``w`` apart: the PAG is ``x o-> z <-o y``, and ``w`` has no edge."""
    rng = np.random.default_rng(0)
    x, y, w = rng.standard_normal((3, 2000))
    z = x + y + rng.standard_normal(2000)
    edges = andrey.rfci(np.column_stack([x, y, z, w])).structure.oriented_edges(index=True)
    assert sorted(edges) == [(0, 2, "partially_directed"), (1, 2, "partially_directed")]


def test_fewer_endpoint_errors_than_the_empty_graph():
    """On five linear-Gaussian graphs of 12 nodes, 2 hidden, the median endpoint errors against
    the true PAG stay below 80% of the empty graph's."""
    errors, empty = [], []
    for seed in range(5):
        dataset = sample_scm(d=12, n=2000, seed=seed, latents=2, density=3.0)
        truth = marginal(dataset.graph, "pag")
        errors.append(shd(andrey.rfci(dataset.data).structure, truth, endpoint_aware=True))
        empty.append(shd(np.zeros((10, 10), dtype=np.int8), truth, endpoint_aware=True))
    assert np.median(errors) < 0.8 * np.median(empty)


def test_pag_ignores_column_order():
    """Five shuffles of 17 columns give the same PAG once mapped back. On these data the
    unshielded-triple step removes an edge the adjacency search kept."""
    data = sample_scm(d=20, n=2000, seed=4, latents=3, density=4.0).data
    reference = andrey.rfci(data).structure.to_numpy()
    skeleton = andrey.pc(data).structure.to_numpy() != 0
    assert np.count_nonzero(reference) < np.count_nonzero(skeleton)
    rng = np.random.default_rng(0)
    for _ in range(5):
        order = rng.permutation(data.shape[1])
        pag = andrey.rfci(data[:, order]).structure.to_numpy()
        back = np.empty_like(pag)
        back[np.ix_(order, order)] = pag
        np.testing.assert_array_equal(back, reference)
