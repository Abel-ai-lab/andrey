"""Recovery and behavior checks for XGES.

Fixed cases have a CPDAG that can be checked by hand. Ground-truth cases are linear-Gaussian SCMs on
Erdos-Renyi graphs from ``andrey.data``, where the BIC score is consistent: on sparse graphs with
many samples GES and XGES both return the true CPDAG, and on denser graphs with fewer samples XGES's
extended search recovers graphs GES misses, as its paper reports. The queue-based search must
return the graph the rescanning reference path in ``andrey.search.xges`` returns.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
import andrey.data as data
from andrey.api._adapt import dag_from_adjacency
from andrey.core.orient import dag2cpdag, from_structure, pdag2dag
from andrey.metrics import shd, to_cpdag
from andrey.search.xges import xges as xges_search
from tests.recovery.helpers import datagen

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


def _er_sample(d: int, avg_degree: float, n: int, seed: int) -> data.CausalDataset:
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=d, avg_degree=avg_degree),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    return scm.sample(n=n, seed=seed)


# ---- fixed cases ---------------------------------------------------------------------------------


@pytest.mark.parametrize("extended_search", [True, False])
def test_independent_columns_give_no_edges(extended_search):
    x = np.random.default_rng(0).standard_normal((1000, 4))
    out = andrey.xges(x, extended_search=extended_search)
    assert out.structure.oriented_edges() == []


@pytest.mark.parametrize("extended_search", [True, False])
@pytest.mark.parametrize("replicate", [0, 1, 2])
def test_recovers_the_cpdag_of_gauss_7v(replicate, extended_search):
    """The collider 0 -> 2 <- 1 orients 2 -> 3 -> 4; nothing orients 5 -- 6."""
    weights = np.asarray(datagen.CASES["gauss_7v"]["weights"])
    truth = to_cpdag(dag_from_adjacency(weights.T != 0))
    x = datagen.generate("gauss_7v", replicate=replicate)
    out = andrey.xges(x, extended_search=extended_search)
    assert shd(out.structure, truth) == 0


def test_output_contract():
    x = datagen.generate("gauss_7v")
    out = andrey.xges(x)
    assert out.structure.kind == "cpdag"
    assert out.metadata["algorithm"] == "XGES"
    assert np.isfinite(out.metadata["score"])
    assert andrey.xges(x, extended_search=False).metadata["algorithm"] == "XGES-0"


# ---- ground truth --------------------------------------------------------------------------------

# Sparse graphs with many samples: GES recovers every true CPDAG, and so must XGES.
_SPARSE_SEEDS = range(4)
# Denser graphs at 2000 rows: GES stops in local optima on most seeds, while XGES finds all but a
# few edges (3 errors over the 8 graphs against GES's 92).
_DENSE_SEEDS = range(8)


def _sparse(seed: int) -> data.CausalDataset:
    return _er_sample(10, 2.0, 10_000, seed)


def _dense(seed: int) -> data.CausalDataset:
    return _er_sample(12, 3.0, 2000, seed)


@pytest.fixture(scope="module")
def dense_fits():
    fits = []
    for seed in _DENSE_SEEDS:
        ds = _dense(seed)
        ges, xges = andrey.ges(ds.data), andrey.xges(ds.data)
        fits.append((to_cpdag(ds.graph), ges, xges, andrey.xges(ds.data, extended_search=False)))
    return fits


@pytest.mark.parametrize("seed", _SPARSE_SEEDS)
def test_recovers_the_true_cpdag_where_ges_does(seed):
    ds = _sparse(seed)
    truth = to_cpdag(ds.graph)
    assert shd(andrey.ges(ds.data).structure, truth) == 0
    assert shd(andrey.xges(ds.data).structure, truth) == 0


def test_beats_ges_on_denser_graphs(dense_fits):
    errors = np.array(
        [
            [shd(ges.structure, truth), shd(xges.structure, truth)]
            for truth, ges, xges, _ in dense_fits
        ]
    )
    ges_total, xges_total = errors.sum(axis=0)
    assert xges_total <= 10
    assert xges_total < ges_total / 3


def test_extended_search_never_scores_worse(dense_fits):
    """XGES keeps a branch only when it lowers the score, so it ends at or below XGES-0."""
    for _truth, _ges, xges, xges0 in dense_fits:
        assert xges.metadata["score"] <= xges0.metadata["score"]


def test_returns_a_cpdag(dense_fits):
    """Each result is the CPDAG of its own equivalence class."""
    for _truth, _ges, xges, xges0 in dense_fits:
        for out in (xges, xges0):
            marks = from_structure(out.structure)
            assert np.array_equal(dag2cpdag(pdag2dag(marks)), marks)


# ---- what the search promises --------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 4])
@pytest.mark.parametrize("extended_search", [True, False])
def test_queue_search_matches_the_rescanning_reference(seed, extended_search):
    """The queues, the incremental updates, and the local CPDAG completion pick the same moves as
    rescanning every operator and re-forming the CPDAG after each move.

    Seeds 0 and 4 are graphs where the extended search keeps branches that change the result.
    """
    x = _dense(seed).data
    fast, fast_score = xges_search(x, extended_search=extended_search)
    slow, slow_score = xges_search(x, extended_search=extended_search, _rescan=True)
    assert np.array_equal(from_structure(fast), from_structure(slow))
    assert fast_score == slow_score


def test_repeat_runs_are_identical():
    x = _dense(0).data
    first, second = andrey.xges(x), andrey.xges(x)
    assert np.array_equal(from_structure(first.structure), from_structure(second.structure))
    assert first.metadata["score"] == second.metadata["score"]


def test_lambda_value_is_the_penalty_ges_uses():
    """At the same lambda_value, XGES-0 and GES search the same score: on a graph both recover,
    the reported scores agree."""
    ds = _sparse(0)
    for lam in (1.0, 2.0):
        ges = andrey.ges(ds.data, lambda_value=lam)
        xges = andrey.xges(ds.data, lambda_value=lam, extended_search=False)
        assert shd(ges.structure, xges.structure) == 0
        assert xges.metadata["score"] == pytest.approx(ges.metadata["score"], rel=1e-12)
