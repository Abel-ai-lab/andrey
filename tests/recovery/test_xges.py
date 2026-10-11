"""Recovery and behavior checks for XGES.

Fixed cases have a CPDAG that can be checked by hand. Ground-truth cases are linear-Gaussian SCMs on
Erdos-Renyi graphs from ``andrey.data``, where the BIC score is consistent: on sparse graphs with
many samples GES and XGES both return the true CPDAG, and on denser graphs with fewer samples XGES's
extended search recovers graphs GES misses, as its paper reports. The queue-based search must
return the graph the rescanning reference path in ``andrey.search.xges`` returns.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

import andrey
import andrey.data as data
from andrey.api._adapt import dag_from_adjacency
from andrey.core import ARROW, TAIL
from andrey.core.orient import dag2cpdag, from_structure, pdag2dag
from andrey.core.score import BICScore
from andrey.metrics import shd, to_cpdag
from andrey.search.ges import _recompute_objective
from andrey.search.xges import (
    _UNDIRECTED,
    _edge_key,
    _extension_parents,
    _marks_total,
    _Search,
    _sets,
)
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


@pytest.mark.parametrize("seed", range(6))
def test_branches_are_scored_on_a_member_of_the_class(seed):
    """The extended search compares branches by the BIC of one DAG it picks from each CPDAG. That
    DAG must belong to the CPDAG's class, so its BIC is the class's BIC."""
    ds = _er_sample(15, 1.5 + seed % 3, 500, seed)
    cpdag = from_structure(to_cpdag(ds.graph))
    sets = [_sets(cpdag, v) for v in range(15)]
    parents = _extension_parents([s[0] for s in sets], [s[1] for s in sets], range(15))
    dag = np.zeros_like(cpdag)
    for v, ps in parents.items():
        for u in ps:
            dag[u, v], dag[v, u] = TAIL, ARROW
    assert np.array_equal(dag2cpdag(dag), cpdag)
    score = BICScore(ds.data)
    assert _marks_total(cpdag, score) == pytest.approx(_recompute_objective(cpdag, score, 15))


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


@pytest.mark.parametrize(("n", "d"), [(8, 12), (12, 20)], ids=["-inf", "nan"])
def test_no_branch_is_tried_when_the_score_is_not_finite(n, d, monkeypatch):
    """With fewer rows than columns, parent sets fit some columns exactly and XGES-0's total is
    -inf or nan. No branch can score below that, so the extended search tries none and returns
    XGES-0's graph."""
    x = np.random.default_rng(0).standard_normal((n, d))
    branches = []
    branch = _Search.branch
    monkeypatch.setattr(_Search, "branch", lambda self: branches.append(1) or branch(self))
    with np.errstate(divide="ignore", invalid="ignore"):
        xges_graph, xges_score = xges_search(x)
        xges0_graph, _ = xges_search(x, extended_search=False)
    assert not np.isfinite(xges_score)
    assert branches == []
    assert np.array_equal(from_structure(xges_graph), from_structure(xges0_graph))


def _rank_deficient() -> dict[str, tuple[np.ndarray, float, bool]]:
    """Data with exactly collinear columns, with the ``lambda_value`` and ``extended_search``."""
    cases = {}
    for name, (n, d, seed, column, weights) in {
        "x5 = x2 + x3": (200, 10, 0, 5, {2: 1.0, 3: 1.0}),
        "x7 = 2 x2 - x3 + x4": (500, 8, 2, 7, {2: 2.0, 3: -1.0, 4: 1.0}),
        "x5 = x2 + x3, d=20": (1000, 20, 1, 5, {2: 1.0, 3: 1.0}),
    }.items():
        x = np.random.default_rng(seed).standard_normal((n, d))
        x[:, column] = sum(w * x[:, j] for j, w in weights.items())
        cases[name] = (x, 1.0, True)
    cases["x5 = x2 + x3, lambda 2"] = (cases["x5 = x2 + x3"][0], 2.0, True)
    rng = np.random.default_rng(6)
    x = rng.standard_normal((300, 6))
    cases["one-hot block"] = (np.column_stack([x, np.eye(3)[rng.integers(0, 3, 300)]]), 1.0, True)
    rng = np.random.default_rng(26)
    hot = np.eye(3)[rng.integers(0, 3, 300)]
    x = rng.standard_normal((300, 5))
    x[:, 0] += hot @ np.array([1.0, -1.0, 0.5])
    x[:, 1] += x[:, 0]
    cases["one-hot block, XGES-0"] = (np.column_stack([x, hot]), 1.0, False)
    return cases


_FIT_EACH = """
import json, sys
import numpy as np
from andrey.search.xges import xges
for path, lambda_value, extended_search in json.loads(sys.argv[1]):
    xges(np.load(path), lambda_value=lambda_value, extended_search=extended_search)
    print(path, flush=True)
"""


def test_the_search_ends_on_rank_deficient_data(tmp_path):
    """Collinear columns make parent sets singular, where the Schur update and two full scores
    disagree and a cycle of moves can each look improving. Every search must still end; a child
    process runs the fits so that one that does not is stopped."""
    runs = []
    for name, (x, lambda_value, extended_search) in _rank_deficient().items():
        path = tmp_path / f"{len(runs)}.npy"
        np.save(path, x)
        runs.append((str(path), lambda_value, extended_search, name))
    try:
        subprocess.run(
            [sys.executable, "-c", _FIT_EACH, json.dumps([r[:3] for r in runs])],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as stopped:
        done = len((stopped.stdout or b"").splitlines())
        pytest.fail(f"no result within 60 s for {runs[done][3]!r}")
