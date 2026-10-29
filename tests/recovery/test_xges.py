"""Recovery and behavior checks for XGES.

Fixed cases have a CPDAG that can be checked by hand. Ground-truth cases are linear-Gaussian SCMs on
Erdos-Renyi graphs from ``andrey.data``, where the BIC score is consistent: on sparse graphs with
many samples GES and XGES both return the true CPDAG, and on denser graphs with fewer samples XGES's
extended search recovers graphs GES misses, as its paper reports. The queue-based search must
return the graph the rescanning reference in ``tests/recovery/helpers/xges_rescan.py`` returns.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

import andrey
import andrey.data as data
from andrey.api._adapt import dag_from_adjacency
from andrey.core import ARROW, TAIL
from andrey.core.orient import dag2cpdag, from_structure, pdag2dag
from andrey.core.score import BICScore
from andrey.metrics import shd, to_cpdag
from andrey.search.ges import _recompute_objective
from andrey.search.hc import tie_ranks
from andrey.search.xges import (
    _UNDIRECTED,
    _edge_key,
    _extension_parents,
    _Graph,
    _Search,
)
from andrey.search.xges import xges as xges_search
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output
from tests.recovery.helpers.xges_rescan import _marks_total, _sets, xges_rescan


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


def _er_sample(d: int, avg_degree: float, n: int, seed: int) -> data.CausalDataset:
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=d, avg_degree=avg_degree),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    return scm.sample(n=n, seed=seed, scale="raw")


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
    order = datagen.column_order("gauss_7v", replicate=replicate)
    truth = to_cpdag(dag_from_adjacency((weights.T != 0)[np.ix_(order, order)]))
    x = datagen.generate("gauss_7v", replicate=replicate)
    out = andrey.xges(x, extended_search=extended_search)
    assert shd(out.structure, truth) == 0


def test_matches_the_recorded_output_on_gauss_5v():
    assert_recorded_output(andrey.xges(datagen.generate("gauss_5v")), "XGES", case="gauss_5v")


def test_output_contract():
    x = datagen.generate("gauss_7v")
    out = andrey.xges(x)
    assert out.structure.kind == "cpdag"
    assert out.metadata["algorithm"] == "xges"
    assert np.isfinite(out.metadata["score"])
    assert andrey.xges(x, extended_search=False).metadata["algorithm"] == "xges"


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
    """GES, XGES, and XGES without the extended search on the dense graphs, at the standard BIC,
    where GES stops in local optima."""
    fits = []
    for seed in _DENSE_SEEDS:
        ds = _dense(seed)
        ges, xges = andrey.ges(ds.data, ebic_gamma=0.0), andrey.xges(ds.data, ebic_gamma=0.0)
        base = andrey.xges(ds.data, ebic_gamma=0.0, extended_search=False)
        fits.append((to_cpdag(ds.graph), ges, xges, base))
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
    """The extended search keeps a branch only when it lowers the score, so the extended result ends
    at or below the base search's."""
    for _truth, _ges, xges, base in dense_fits:
        assert xges.metadata["score"] <= base.metadata["score"]


def test_returns_a_cpdag(dense_fits):
    """Each result is the CPDAG of its own equivalence class."""
    for _truth, _ges, xges, base in dense_fits:
        for out in (xges, base):
            marks = from_structure(out.structure)
            assert np.array_equal(dag2cpdag(pdag2dag(marks)), marks)


# ---- what the search promises --------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 4])
@pytest.mark.parametrize("extended_search", [True, False])
@pytest.mark.parametrize("ebic_gamma", [0.0, 1.0])
def test_queue_search_matches_the_rescanning_reference(seed, extended_search, ebic_gamma):
    """The queues, the incremental updates, and the local CPDAG completion pick the same moves as
    rescanning every operator and re-forming the CPDAG after each move.

    At ``ebic_gamma=0.0``, seeds 0 and 4 are graphs where the extended search keeps branches that
    change the result.
    """
    x = _dense(seed).data
    fast, fast_score = xges_search(x, ebic_gamma=ebic_gamma, extended_search=extended_search)
    slow, slow_score = xges_rescan(x, ebic_gamma=ebic_gamma, extended_search=extended_search)
    assert np.array_equal(from_structure(fast), from_structure(slow))
    assert fast_score == slow_score


def test_repeat_runs_are_identical():
    x = _dense(0).data
    first, second = andrey.xges(x), andrey.xges(x)
    assert np.array_equal(from_structure(first.structure), from_structure(second.structure))
    assert first.metadata["score"] == second.metadata["score"]


def test_penalty_discount_is_the_penalty_ges_uses():
    """At the same penalty_discount, XGES without the extended search and GES search the same
    score: on a graph both recover, the reported scores agree."""
    ds = _sparse(0)
    for lam in (1.0, 2.0):
        ges = andrey.ges(ds.data, penalty_discount=lam)
        xges = andrey.xges(ds.data, penalty_discount=lam, extended_search=False)
        assert shd(ges.structure, xges.structure) == 0
        assert xges.metadata["score"] == pytest.approx(ges.metadata["score"], rel=1e-12)


def _extension_dag(cpdag: np.ndarray) -> np.ndarray:
    """The DAG whose BIC the extended search scores for ``cpdag``, as a mark matrix."""
    d = cpdag.shape[0]
    sets = [_sets(cpdag, v) for v in range(d)]
    dag = np.zeros_like(cpdag)
    for v, ps in _extension_parents([s[0] for s in sets], [s[1] for s in sets], range(d)).items():
        for u in ps:
            dag[u, v], dag[v, u] = TAIL, ARROW
    return dag


@pytest.mark.parametrize("seed", range(6))
def test_branches_are_scored_on_a_member_of_the_class(seed):
    """The extended search compares branches by the BIC of one DAG it picks from each CPDAG. That
    DAG must belong to the CPDAG's class, so its BIC is the class's BIC."""
    ds = _er_sample(15, 1.5 + seed % 3, 500, seed)
    cpdag = from_structure(to_cpdag(ds.graph))
    assert np.array_equal(dag2cpdag(_extension_dag(cpdag)), cpdag)
    score = BICScore(ds.data)
    assert _marks_total(cpdag, score) == pytest.approx(_recompute_objective(cpdag, score, 15))


@pytest.mark.parametrize("seed", [0, 4])
def test_the_extended_search_scores_the_graph_it_returns(seed):
    """The extended search keeps a running total, rescoring only the nodes a kept branch changed.
    On the returned graph it must equal the total over the whole graph, scored on a DAG in that
    graph's class. On seeds 0 and 4 the search keeps branches at the standard BIC."""
    x = _dense(seed).data
    score = BICScore(x, ebic_gamma=0.0)
    search = _Search(score, x.shape[1], tie_ranks(x).tolist())
    with threadpool_limits(limits=1):
        search.inserts_on_empty_graph()
        search.base_search()
        base_total = _marks_total(search.g.marks(), score)
        total = search.extend(x.shape[0])
    marks = search.g.marks()
    assert np.array_equal(marks, from_structure(xges_search(x, ebic_gamma=0.0)[0]))
    assert total < base_total
    assert total == _marks_total(marks, score)
    assert np.array_equal(dag2cpdag(_extension_dag(marks)), marks)


def test_a_stale_snapshot_cannot_seed_a_branch():
    """A branch shares its snapshot's sets, so it needs a snapshot of the graph as it is now."""
    g = _Graph(4)
    g.set_state(0, 1, _UNDIRECTED)
    snapshot = g.copy()
    branch = g.branch(snapshot)
    branch.set_state(1, 2, _UNDIRECTED)  # copies node 1 and 2 from `g`; the snapshot is unchanged
    assert snapshot.ne[1] == {0} and g.ne[1] == {0}
    g.set_state(2, 3, _UNDIRECTED)
    with pytest.raises(RuntimeError, match="no longer matches"):
        g.branch(snapshot)
    stale = g.copy()
    stale.set_state(0, 3, _UNDIRECTED)
    with pytest.raises(RuntimeError, match="no longer matches"):
        g.branch(stale)


@pytest.mark.parametrize("extended_search", [True, False])
def test_singular_parent_sets_raise_no_floating_point_warning(extended_search):
    """With fewer rows than columns some parent sets fit their child exactly and are singular;
    the search scores them without a numpy warning, as the whole run, including the extended
    search's trials, ignores divisions by zero."""
    x = np.random.default_rng(0).standard_normal((8, 12))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        graph, total = xges_search(x, extended_search=extended_search)
        assert np.isfinite(_marks_total(from_structure(graph), BICScore(x)))
    assert np.isfinite(total)


def _colliding_pairs(pairs: int) -> list[int]:
    """Indices ``j`` whose undirected edges ``2j -- 2j+1`` have edge keys that XOR to zero.

    Among more 128-bit keys than bits some subset XORs to zero; Gaussian elimination over GF(2)
    returns the first such subset, no proper part of which XORs to zero.
    """
    basis: dict[int, tuple[int, int]] = {}  # leading bit -> (key combination, the indices in it)
    for j in range(pairs):
        key, used = _edge_key(2 * j, 2 * j + 1, _UNDIRECTED), 1 << j
        while key:
            top = key.bit_length() - 1
            if top not in basis:
                basis[top] = (key, used)
                break
            key, used = key ^ basis[top][0], used ^ basis[top][1]
        else:
            return [i for i in range(pairs) if used >> i & 1]
    raise AssertionError("no subset of the edge keys XORs to zero")


@pytest.mark.parametrize("extended_search", [True, False])
def test_a_fingerprint_collision_does_not_undo_a_move(extended_search):
    """The search looks up visited graphs by a fingerprint, the XOR of per-edge keys, which
    distinct graphs can share. The columns are exactly uncorrelated except within the pairs whose
    edge keys XOR to zero: each of those edges improves the score, and the last one inserted gives
    the graph the empty graph's fingerprint."""
    pairs = _colliding_pairs(129)
    z = np.random.default_rng(0).standard_normal((512, 258))
    x = np.linalg.qr(z - z.mean(axis=0))[0] * np.sqrt(512)
    x[:, [2 * j + 1 for j in pairs]] += x[:, [2 * j for j in pairs]]
    marks = from_structure(xges_search(x, extended_search=extended_search)[0])
    edges = {(int(a), int(b)) for a, b in zip(*np.nonzero(np.triu(marks)), strict=True)}
    assert edges == {(2 * j, 2 * j + 1) for j in pairs}


@pytest.mark.parametrize("extended_search", [True, False])
def test_fewer_rows_than_columns_keep_parent_sets_that_can_be_scored(extended_search):
    """With fewer rows than columns, a parent set of ``n - 1`` or more fits its child exactly. The
    search never picks one, so every node of the graph's DAG extension has a finite score."""
    n, d = 8, 12
    x = np.random.default_rng(0).standard_normal((n, d))
    graph, total = xges_search(x, extended_search=extended_search)
    dag = pdag2dag(from_structure(graph))
    assert max(int(np.count_nonzero(dag[:, v] == TAIL)) for v in range(d)) <= n - 2
    assert np.isfinite(total)
