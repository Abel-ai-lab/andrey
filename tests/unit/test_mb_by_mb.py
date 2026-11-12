"""Checks of ``mb_by_mb``, MB-by-MB (Wang, Zhou, Zhao, and Geng 2014).

Under a d-separation oracle the answer at every target of random graphs is the target's edges in
the true CPDAG (Theorem 3), with the paper's blanket learner, IAMB's grow phase, and with IAMB;
the paper's chessboard needs two blankets for every target off its top and left sides (Section
4.2), and an orientation four blankets away reaches the target. The grow phase keeps what IAMB's
shrink drops. On data the answer does not depend on the column order or the columns' units, and
the result holds the target's sets, the blankets learned, and the test count.
"""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import andrey
from andrey import data
from andrey.constraint import sequential as sequential_engine
from andrey.constraint.blanket import LEARNERS, LocalSearch
from andrey.core import ci
from andrey.core.ci import register_indep_test
from andrey.data.latent import DSepOracle, causal_digraph
from andrey.metrics import target_oracle


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        yield


@pytest.fixture
def dsep():
    """Register ``"dsep"``: d-separation in ``box["dag"]``, column ``k`` as node ``k``."""
    box: dict[str, nx.DiGraph] = {}
    register_indep_test("dsep", lambda x: DSepOracle(box["dag"]))
    yield box
    ci._CI_REGISTRY.pop("dsep", None)


def _dag(edges, d: int) -> nx.DiGraph:
    dag = nx.DiGraph(edges)
    dag.add_nodes_from(range(d))
    return dag


def _oracle_fit(box: dict, dag: nx.DiGraph, target: int, seed: int = 0, **settings):
    """``mb_by_mb`` under the oracle, uncapped, the setting of Theorem 3; the random columns give
    the ranks that break the oracle's ties."""
    box["dag"] = dag
    d = dag.number_of_nodes()
    x = np.random.default_rng(seed).standard_normal((40, d))
    return andrey.mb_by_mb(x, target=target, indep_test="dsep", local_max_depth=None, **settings)


def _truth(dag: nx.DiGraph, target: int) -> np.ndarray:
    adjacency = nx.to_numpy_array(dag, nodelist=range(dag.number_of_nodes()))
    return target_oracle(adjacency, target, "cpdag").to_numpy()


def _random_dag(seed: int, d: int) -> nx.DiGraph:
    truth = data.sample_scm(graph="erdos_renyi", d=d, n=2, seed=seed, density=2.5).graph
    return causal_digraph(truth)


# ---- the answer under the oracle -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("blanket", "seed", "d"),
    [(b, s, 8 + s % 7) for b in ("iamb_forward", "iamb") for s in range(10)] + [("iamb", 40, 30)],
)
def test_the_answer_is_the_cpdag_at_every_target_under_the_oracle(dsep, blanket, seed, d):
    dag = _random_dag(seed, d)
    for target in range(d):
        out = _oracle_fit(dsep, dag, target, seed, blanket=blanket)
        truth = _truth(dag, target)
        assert np.array_equal(out.structure.to_numpy(), truth), target
        meta = out.metadata
        parents = set(np.flatnonzero((truth[target] == 2) & (truth[:, target] == 1)).tolist())
        assert set(meta["parents"]) == parents
        assert set(meta["parents"]) | set(meta["children"]) | set(meta["undirected"]) == set(
            np.flatnonzero(truth[target]).tolist()
        )
        assert meta["blankets_learned"][0] == target
        assert len(set(meta["blankets_learned"])) == len(meta["blankets_learned"])


def _grid(k: int) -> nx.DiGraph:
    """The ``k x k`` chessboard of Section 4.2: each node points right and down; node ``r * k + c``
    sits in row ``r`` and column ``c``."""
    edges = [(r * k + c, r * k + c + 1) for r in range(k) for c in range(k - 1)]
    edges += [(r * k + c, (r + 1) * k + c) for r in range(k - 1) for c in range(k)]
    return _dag(edges, k * k)


# Section 4 supposes each blanket learned exactly, so the chessboard runs IAMB; the grow phase
# alone can admit the shared parent 1 of 2 and 4 into the center's set, which then orients both
# edges in one step.


def test_the_3x3_chessboard_takes_two_blankets_at_its_center(dsep):
    # Fig. 6: the center 5 (position 4); after MB(5), 3 and 7 are left behind directed edges, and
    # MB(4) or MB(2) orients 2 -> 5 <- 4 through the shared parent 1.
    out = _oracle_fit(dsep, _grid(3), 4, blanket="iamb")
    meta = out.metadata
    assert set(meta["parents"]) == {1, 3} and set(meta["children"]) == {5, 7}
    assert meta["undirected"] == []
    assert len(meta["blankets_learned"]) == 2 and meta["blankets_learned"][1] in (1, 3)


@pytest.mark.parametrize("seed", range(3))
def test_a_chessboard_target_off_the_top_and_left_sides_takes_two_blankets(dsep, seed):
    k = 5
    dag = _grid(k)
    for target in range(k * k):
        out = _oracle_fit(dsep, dag, target, seed, blanket="iamb")
        assert np.array_equal(out.structure.to_numpy(), _truth(dag, target)), target
        if target >= k and target % k:
            assert len(out.metadata["blankets_learned"]) == 2, target


def test_an_orientation_four_blankets_away_reaches_the_target(dsep):
    # p -> q <- r, q -> s -> t -> T: only the collider at q orients t -> T.
    target, t, s, q, p, r = range(6)
    out = _oracle_fit(dsep, _dag([(p, q), (r, q), (q, s), (s, t), (t, target)], 6), target)
    assert out.metadata["parents"] == [t] and out.metadata["children"] == []
    assert out.metadata["blankets_learned"] == [target, t, s, q]


def test_fig_2a_directs_both_edges_at_the_target(dsep):
    # Example 1, Fig. 2(a): 2 -> T -> 1, 3 -> 2, 4 -> 3 <- 5, 6 -> 4, 6 -> 5; T is column 0.
    dag = _dag([(2, 0), (0, 1), (3, 2), (4, 3), (5, 3), (6, 4), (6, 5)], 7)
    out = _oracle_fit(dsep, dag, 0)
    assert out.metadata["parents"] == [2] and out.metadata["children"] == [1]
    assert out.metadata["blankets_learned"][:1] == [0]


# ---- IAMB's grow phase ---------------------------------------------------------------------------


def test_the_grow_phase_keeps_what_the_shrink_drops():
    # a -> b -> t: a joins first on the oracle's tie, and only IAMB's shrink removes it.
    a, b, t = 0, 1, 2
    dag = _dag([(a, b), (b, t)], 3)
    found = {}
    for learner in ("iamb_forward", "iamb"):
        search = LocalSearch(DSepOracle(dag), [0, 1, 2], 0.05)
        found[learner] = LEARNERS[learner].search(search, t, {})
    assert found == {"iamb_forward": [a, b], "iamb": [b]}


def test_the_grow_phase_stops_once_the_blanket_holds_more_than_the_cap():
    # Six parents of t: the grow phase conditions on the whole set, so a cap of 2 stops it at 3.
    dag = _dag([(p, 6) for p in range(6)], 7)
    search = LocalSearch(DSepOracle(dag), list(range(7)), 0.05, max_depth=2)
    assert LEARNERS["iamb_forward"].search(search, 6, {}) == [0, 1, 2]


def test_local_max_depth_caps_the_local_structures_alone(monkeypatch):
    # Six parents of t: the blanket learner conditions on up to five of them, while the local
    # structures test their pairs given at most local_max_depth members.
    dag = _dag([(p, 6) for p in range(6)], 7)
    sizes: dict[str, list[int]] = {"blanket": [], "local": []}
    step = ["blanket"]

    class Recorded:
        def __init__(self, data, **_):
            self._test = DSepOracle(dag)

        def __call__(self, x, y, condition_set=()):
            sizes[step[0]].append(len(condition_set))
            return self._test(x, y, condition_set)

    def learn_local(*args, **kwargs):
        step[0] = "local"
        try:
            return learn(*args, **kwargs)
        finally:
            step[0] = "blanket"

    learn = sequential_engine.learn_local
    monkeypatch.setattr(sequential_engine, "learn_local", learn_local)
    monkeypatch.setitem(ci._CI_REGISTRY, "recorded", Recorded)
    x = np.random.default_rng(0).standard_normal((40, 7))
    andrey.mb_by_mb(x, target=6, indep_test="recorded", local_max_depth=1)
    assert max(sizes["local"]) == 1 < max(sizes["blanket"])


# ---- the result on data --------------------------------------------------------------------------


def _collider() -> pd.DataFrame:
    """a -> c <- b, c -> e: the target c has two parents and a child."""
    rng = np.random.default_rng(0)
    a, b, u, v = rng.standard_normal((4, 2000))
    c = a + b + 0.5 * u
    return pd.DataFrame({"a": a, "b": b, "c": c, "e": c + 0.5 * v})


def test_the_result_names_the_target_sets_by_label():
    out = andrey.mb_by_mb(_collider(), target="c")
    meta = out.metadata
    assert out.structure.kind == "cpdag"
    assert (meta["target"], meta["parents"], meta["children"], meta["undirected"]) == (
        "c",
        ["a", "b"],
        ["e"],
        [],
    )
    assert meta["blankets_learned"][0] == "c" and meta["n_tests"] > 0
    assert sorted(out.structure.oriented_edges()) == [
        ("a", "c", "directed"),
        ("b", "c", "directed"),
        ("c", "e", "directed"),
    ]


def test_a_target_is_required():
    with pytest.raises(ValueError, match="mb_by_mb: pass target"):
        andrey.mb_by_mb(_collider())


def test_a_learner_of_parents_and_children_is_refused():
    with pytest.raises(ValueError, match="blanket"):
        andrey.mb_by_mb(_collider(), target="c", blanket="mmpc")


def _answer(x: np.ndarray, target: int, **settings) -> np.ndarray:
    return andrey.mb_by_mb(x, target=target, **settings).structure.to_numpy()


@pytest.mark.parametrize("seed", range(2))
def test_the_answer_does_not_depend_on_the_column_order_or_the_units(seed):
    sample = data.sample_scm(graph="erdos_renyi", d=12, n=1500, seed=seed, density=2.0)
    x = sample.data
    d = x.shape[1]
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    scale = 10 ** rng.uniform(-2, 2, d)
    for target in range(d):
        answer = _answer(x, target)
        moved = _answer(x[:, order], int(np.flatnonzero(order == target)[0]))
        assert np.array_equal(moved, answer[np.ix_(order, order)]), target
        assert np.array_equal(_answer(x * scale, target), answer), target


def test_a_fit_repeats():
    x = data.sample_scm(graph="erdos_renyi", d=10, n=800, seed=3, density=2.0).data
    first = andrey.mb_by_mb(x, target=4)
    again = andrey.mb_by_mb(x, target=4)
    assert np.array_equal(first.structure.to_numpy(), again.structure.to_numpy())
    assert first.metadata == again.metadata


def test_a_tier_directs_the_edge_the_data_leave_undirected():
    # a -> b -> c: the chain is one class, so the edges at b stay undirected without knowledge.
    rng = np.random.default_rng(1)
    a = rng.standard_normal(2000)
    b = a + 0.5 * rng.standard_normal(2000)
    frame = pd.DataFrame({"a": a, "b": b, "c": b + 0.5 * rng.standard_normal(2000)})
    plain = andrey.mb_by_mb(frame, target="b")
    assert sorted(plain.metadata["undirected"]) == ["a", "c"]
    tiered = andrey.mb_by_mb(frame, target="b", knowledge=andrey.Knowledge(tiers=[["a"], ["b"]]))
    assert tiered.metadata["parents"] == ["a"] and tiered.metadata["children"] == ["c"]
    forbidden = andrey.mb_by_mb(
        frame, target="b", knowledge=andrey.Knowledge(forbidden=[("a", "b"), ("b", "a")])
    )
    assert "a" not in forbidden.metadata["undirected"] + forbidden.metadata["parents"]


def test_paths_are_not_honored():
    with pytest.raises(NotImplementedError, match="required_paths"):
        andrey.mb_by_mb(
            _collider(), target="c", knowledge=andrey.Knowledge(required_paths=[("a", "e")])
        )
