"""Checks of ``mmb_by_mmb``, MMB-by-MMB (Xie, Li, Wu, Zeng, Liu, and Geng 2024).

Under a d-separation oracle, total conditioning finds every target's Markov blanket in the maximal
ancestral graph of random graphs with hidden common causes, as IAMB does; the paper's Example 4
gives the PAG's marks at its target; the answer at every target of random graphs is the PAG's; the
stop guard decides the circle where the paper's R3 stops early, and the collider check drops the
edge the PC step keeps on the graph where it does. Total conditioning tests at the level of the
authors' code, MB-by-MB's at alpha.
On data the answer does not depend on the column order or the columns' units, knowledge sets
marks, and the result holds the target's sets, the blankets learned, and the test count.
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
from andrey.core import ARROW, TAIL, ci
from andrey.core.ci import counting_tests, register_indep_test
from andrey.data.latent import DSepOracle, causal_digraph
from andrey.metrics import target_oracle


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        yield


@pytest.fixture
def dsep():
    """Register ``"dsep"``: d-separation in ``box["dag"]``, column ``k`` as node ``k``; the hidden
    nodes come after the columns and are never asked about."""
    box: dict[str, nx.DiGraph] = {}
    register_indep_test("dsep", lambda x: DSepOracle(box["dag"]))
    yield box
    ci._CI_REGISTRY.pop("dsep", None)


def _dag(edges, d: int) -> nx.DiGraph:
    dag = nx.DiGraph(edges)
    dag.add_nodes_from(range(d))
    return dag


def _random_dag(seed: int, d: int, hidden: int, mediated: bool) -> nx.DiGraph:
    """An Erdos-Renyi DAG over ``d + hidden`` nodes, the last ``hidden`` of them hidden: each a
    common parent of two observed nodes and, with ``mediated``, a child of a third, earlier one."""
    rng = np.random.default_rng(seed)
    dag = causal_digraph(
        data.sample_scm(graph="erdos_renyi", d=d, n=2, seed=seed, density=2.5).graph
    )
    order = {v: k for k, v in enumerate(nx.topological_sort(dag))}
    for h in range(d, d + hidden):
        a, b, c = sorted(rng.choice(d, size=3, replace=False).tolist(), key=order.__getitem__)
        dag.add_edges_from([(h, b), (h, c)])
        if mediated:
            dag.add_edge(a, h)
    return dag


def _oracle_fit(box: dict, dag: nx.DiGraph, d: int, target: int, seed: int = 0, **settings):
    """``mmb_by_mmb`` under the oracle on the ``d`` observed columns, the local structures uncapped
    (the setting of Theorem 4); the random columns give the ranks that break the oracle's ties."""
    box["dag"] = dag
    x = np.random.default_rng(seed).standard_normal((40, d))
    return andrey.mmb_by_mmb(x, target=target, indep_test="dsep", local_max_depth=None, **settings)


def _truth(dag: nx.DiGraph, d: int, target: int) -> np.ndarray:
    adjacency = nx.to_numpy_array(dag, nodelist=range(dag.number_of_nodes()))
    hidden = range(d, dag.number_of_nodes())
    return target_oracle(adjacency, target, "pag", latent=hidden).to_numpy()


# ---- total conditioning --------------------------------------------------------------------------


@pytest.mark.parametrize(("seed", "d", "hidden"), [(s, 7 + s % 6, s % 4) for s in range(8)])
def test_total_conditioning_finds_the_blanket_of_the_maximal_ancestral_graph(seed, d, hidden):
    dag = _random_dag(seed, d, hidden, seed % 2 == 1)
    adjacency = nx.to_numpy_array(dag, nodelist=range(d + hidden))
    for target in range(d):
        truth = target_oracle(adjacency, target, "markov_blanket", latent=range(d, d + hidden))
        found = {}
        for learner in ("total_conditioning", "iamb"):
            search = LocalSearch(DSepOracle(dag), list(range(d)), 0.05)
            found[learner] = set(LEARNERS[learner].search(search, target, {}))
        assert found["total_conditioning"] == found["iamb"] == set(truth.neighbors(target)), target


def test_total_conditioning_tests_each_variable_given_all_the_others():
    # a -> t <- b, t -> c, e alone: e and nothing else is independent of t given the rest.
    dag = _dag([(0, 2), (1, 2), (2, 3)], 5)
    search = LocalSearch(DSepOracle(dag), list(range(5)), 0.05)
    sepsets: dict[int, tuple[int, ...]] = {}
    with counting_tests() as tests:
        blanket = LEARNERS["total_conditioning"].search(search, 2, sepsets)
    assert sorted(blanket) == [0, 1, 3] and sepsets == {4: (0, 1, 3)}
    assert tests.n == 4


class _GivenTheRest:
    """The p-value ``p`` for a query given all the other ``d - 2`` columns, 0 for any other."""

    def __init__(self, d: int, p: float) -> None:
        self.d, self.p = d, p

    def __call__(self, x: int, y: int, condition_set=()) -> float:
        return self.p if len(tuple(condition_set)) == self.d - 2 else 0.0


def _given_the_rest_fit(method, d: int, p: float, **settings):
    """``method`` at ``alpha=0.01`` on ``d`` columns, every total-conditioning test at ``p``."""
    register_indep_test("given_rest", lambda x: _GivenTheRest(d, p))
    try:
        x = np.random.default_rng(d).standard_normal((40, d))
        return method(
            x, target=0, alpha=0.01, indep_test="given_rest", local_max_depth=0, **settings
        )
    finally:
        ci._CI_REGISTRY.pop("given_rest", None)


# The level of the authors' code: alpha / (10 * floor(d / 10)) over d >= 10 columns, alpha below.
@pytest.mark.parametrize(
    ("d", "level"), [(5, 0.01), (9, 0.01), (10, 0.001), (19, 0.001), (20, 0.0005), (45, 0.00025)]
)
def test_total_conditioning_tests_at_the_level_of_the_authors_code(d, level):
    for p, joins in ((level, True), (level * 1.01, False)):
        out = _given_the_rest_fit(andrey.mmb_by_mmb, d, p)
        assert sorted(out.metadata["markov_blanket"]) == (list(range(1, d)) if joins else []), p


def test_mb_by_mb_tests_total_conditioning_at_alpha():
    # Between MMB-by-MMB's level at d = 20 (0.0005) and alpha: MB-by-MB keeps every variable.
    out = _given_the_rest_fit(andrey.mb_by_mb, 20, 0.005, blanket="total_conditioning")
    assert np.count_nonzero(out.structure.to_numpy()[0]) == 19
    assert _given_the_rest_fit(andrey.mmb_by_mmb, 20, 0.005).metadata["markov_blanket"] == []


def test_below_a_cap_on_the_rest_total_conditioning_runs_no_test_and_keeps_every_variable():
    dag = _dag([(0, 2), (1, 2), (2, 3)], 5)
    search = LocalSearch(DSepOracle(dag), [4, 3, 2, 1, 0], 0.05, max_depth=2)
    with counting_tests() as tests:
        blanket = LEARNERS["total_conditioning"].search(search, 2, {})
    assert blanket == [4, 3, 1, 0] and tests.n == 0  # every other variable, in rank order


# ---- the answer under the oracle -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("seed", "blanket"),
    [(s, "total_conditioning") for s in range(14)] + [(s, "iamb") for s in range(4)],
)
def test_the_answer_is_the_pags_at_every_target(dsep, seed, blanket):
    d, hidden = 7 + seed % 9, seed % 5
    dag = _random_dag(seed, d, hidden, seed % 2 == 1)
    for target in range(d):
        out = _oracle_fit(dsep, dag, d, target, seed, blanket=blanket)
        answer = out.structure.to_numpy()
        assert np.array_equal(answer, _truth(dag, d, target)), target
        meta = out.metadata
        assert meta["blankets_learned"][0] == target
        keys = ("parents", "children", "bidirected", "undetermined")
        listed = [j for key in keys for j in meta[key]]
        assert sorted(listed) == np.flatnonzero(answer[target]).tolist()


# Fig. 1 and Example 4 of Xie et al. (2024), the target V5. Columns: V2, V3, V4, V5, V7, V8, V9,
# V10, V11, V12; hidden V1 (into V3, V4) and V6 (into V5, V8). V7 -> V2 and V9 -> V7 are not stated
# in the text; every independence its examples state holds.
_V = {name: k for k, name in enumerate([2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 1, 6])}
_XIE = [
    (11, 3), (1, 3), (1, 4), (2, 4), (7, 2), (9, 7), (4, 5), (6, 5), (6, 8), (5, 10), (3, 10),
    (7, 8), (12, 8),
]  # fmt: skip


@pytest.mark.parametrize("blanket", ["total_conditioning", "iamb"])
def test_example_4_gives_the_pags_marks(dsep, blanket):
    dag = _dag([(_V[a], _V[b]) for a, b in _XIE], 12)
    out = _oracle_fit(dsep, dag, 10, _V[5], blanket=blanket)
    meta = out.metadata
    assert sorted(meta["markov_blanket"]) == [_V[v] for v in (3, 4, 7, 8, 10, 12)]
    # The paper's walk stops after V4's blanket decides V4 -> V5 (R1); the stop guard reads the
    # rules only once more blankets are learned than V5 has neighbors (V4, V8, V10).
    learned = meta["blankets_learned"]
    assert learned[0] == _V[5] and {_V[4], _V[8], _V[10]} <= set(learned)
    assert (meta["parents"], meta["children"]) == ([_V[4]], [_V[10]])
    assert (meta["bidirected"], meta["undetermined"]) == ([_V[8]], [])
    assert np.array_equal(out.structure.to_numpy(), _truth(dag, 10, _V[5]))


# The stop rule R3 on the graph 0 -> 3 <- 1 -> 4 <- 2 -> 3, 0 -> 5 <- 1, 5 -> 4: at the target 1,
# every edge already ends in an arrowhead after its own blanket, so the paper's R3 stops there,
# while the PAG decides the circle at 1 on 1 -> 4 through the discriminating path
# 0 *-> 5 <-* 1, 5 -> 4, which only the blankets of 4 or 5 bring in.
_R3 = [(0, 3), (1, 3), (2, 3), (1, 4), (2, 4), (0, 5), (1, 5), (5, 4)]


def test_the_stop_guard_decides_the_circle_r3_leaves(dsep):
    dag = _dag(_R3, 6)
    truth = _truth(dag, 6, 1)
    assert truth[1, 4] == TAIL
    # 1 has three neighbors after its own step, so the rules are read after four blankets.
    at_1 = _oracle_fit(dsep, dag, 6, 1)
    assert len(at_1.metadata["blankets_learned"]) >= 4
    assert (at_1.metadata["children"], at_1.metadata["undetermined"]) == ([4], [3, 5])
    assert np.array_equal(at_1.structure.to_numpy(), truth)
    at_4 = _oracle_fit(dsep, dag, 6, 4)
    assert (at_4.metadata["parents"], at_4.metadata["undetermined"]) == ([1, 5], [2])
    assert np.array_equal(at_4.structure.to_numpy(), _truth(dag, 6, 4))


def test_the_collider_check_drops_an_edge_the_pc_step_keeps(dsep):
    # In this graph only {4, 6, 9, 10, 12} and {4, 9, 10, 12, 13} separate 1 and 7, sets the PC
    # step over MB+(10) never tests (neither end keeps all of them as neighbors), so L_10 holds
    # 1 - 7 and the v-structure 1 *-> 7 <-* 10, a v-structure that contains 10. 7's own step tests
    # the pair against every subset of MB(7) without 1 and separates it, so the check drops the
    # v-structure, and the answer at 7 is the PAG's.
    dag = _random_dag(53, 15, 3, True)
    out = _oracle_fit(dsep, dag, 15, 7, 53)
    assert np.array_equal(out.structure.to_numpy(), _truth(dag, 15, 7))
    search = LocalSearch(DSepOracle(dag), list(range(15)), 0.05)
    blanket = search.local("iamb", 10)[1]
    assert (1, 7) in sequential_engine.learn_local(search, 10, blanket, "pc").adjacent
    assert (1, 7) not in sequential_engine.learn_local(search, 10, blanket, "ic").adjacent


def test_local_max_depth_caps_the_local_structures_alone(monkeypatch):
    # Six parents of t: total conditioning conditions on the five other variables, while the local
    # structures test their pairs given at most local_max_depth members; a blanket learned during
    # a local structure, to compare the two ends of a pair, counts as the blanket learner's.
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

    def learn_blanket(*args, **kwargs):
        previous, step[0] = step[0], "blanket"
        try:
            return learn_mb(*args, **kwargs)
        finally:
            step[0] = previous

    learn = sequential_engine.learn_local
    learn_mb = sequential_engine.learn_blanket
    monkeypatch.setattr(sequential_engine, "learn_local", learn_local)
    monkeypatch.setattr(sequential_engine, "learn_blanket", learn_blanket)
    monkeypatch.setitem(ci._CI_REGISTRY, "recorded", Recorded)
    x = np.random.default_rng(0).standard_normal((40, 7))
    andrey.mmb_by_mmb(x, target=6, indep_test="recorded", local_max_depth=1)
    assert max(sizes["local"]) == 1 < max(sizes["blanket"]) == 5


# ---- the result on data --------------------------------------------------------------------------


def _confounded(n: int = 3000) -> pd.DataFrame:
    """a -> c <- b, c -> e, g -> f, and a hidden cause of e and f: the PAG has a o-> c <-o b,
    c -> e, e <-> f, and g o-> f."""
    rng = np.random.default_rng(0)
    a, b, g, h, u, v, w = rng.standard_normal((7, n))
    c = a + b + 0.5 * u
    frame = {"a": a, "b": b, "c": c, "e": c + h + 0.5 * v, "f": h + g + 0.5 * w, "g": g}
    return pd.DataFrame(frame)


def test_the_result_names_the_target_sets_by_label():
    out = andrey.mmb_by_mmb(_confounded(), target="e")
    meta = out.metadata
    assert out.structure.kind == "pag"
    assert (meta["target"], meta["parents"], meta["children"]) == ("e", ["c"], [])
    assert (meta["bidirected"], meta["undetermined"]) == (["f"], [])
    assert sorted(meta["markov_blanket"]) == ["c", "f", "g"]
    assert meta["blankets_learned"][0] == "e" and meta["n_tests"] > 0
    at_c = andrey.mmb_by_mmb(_confounded(), target="c").metadata
    assert (at_c["children"], sorted(at_c["undetermined"])) == (["e"], ["a", "b"])


def test_a_target_is_required():
    with pytest.raises(ValueError, match="mmb_by_mmb: pass target"):
        andrey.mmb_by_mmb(_confounded())


@pytest.mark.parametrize("learner", ["mmpc", "iamb_forward", "hiton_mb"])
def test_a_learner_the_proof_does_not_cover_is_refused(learner):
    with pytest.raises(ValueError, match="blanket"):
        andrey.mmb_by_mmb(_confounded(), target="c", blanket=learner)


def _answer(x: np.ndarray, target: int, **settings) -> np.ndarray:
    return andrey.mmb_by_mmb(x, target=target, **settings).structure.to_numpy()


@pytest.mark.parametrize(("seed", "blanket"), [(0, "total_conditioning"), (1, "iamb")])
def test_the_answer_does_not_depend_on_the_column_order_or_the_units(seed, blanket):
    sample = data.sample_scm(
        graph="erdos_renyi", d=13, n=1500, seed=seed, density=2.0, latents=2, scale="raw"
    )
    x = sample.data
    d = x.shape[1]
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    scale = 10 ** rng.uniform(-2, 2, d)
    for target in range(d):
        answer = _answer(x, target, blanket=blanket)
        moved = _answer(x[:, order], int(np.flatnonzero(order == target)[0]), blanket=blanket)
        assert np.array_equal(moved, answer[np.ix_(order, order)]), target
        assert np.array_equal(_answer(x * scale, target, blanket=blanket), answer), target


def test_a_fit_repeats():
    x = data.sample_scm(graph="erdos_renyi", d=10, n=800, seed=3, density=2.0, latents=1).data
    first = andrey.mmb_by_mmb(x, target=4)
    again = andrey.mmb_by_mmb(x, target=4)
    assert np.array_equal(first.structure.to_numpy(), again.structure.to_numpy())
    assert first.metadata == again.metadata


def test_knowledge_sets_the_marks_it_decides():
    # a -> b -> c: the chain leaves circles at b's edges without knowledge.
    rng = np.random.default_rng(1)
    a = rng.standard_normal(2000)
    b = a + 0.5 * rng.standard_normal(2000)
    frame = pd.DataFrame({"a": a, "b": b, "c": b + 0.5 * rng.standard_normal(2000)})
    plain = andrey.mmb_by_mmb(frame, target="b")
    assert sorted(plain.metadata["undetermined"]) == ["a", "c"]
    # Tiers put an arrowhead at b on a - b and at c on b - c; the FCI rules then direct b -> c.
    tiered = andrey.mmb_by_mmb(
        frame, target="b", knowledge=andrey.Knowledge(tiers=[["a"], ["b"], ["c"]])
    )
    marks = tiered.structure.to_numpy()
    assert marks[1, 0] == ARROW and tiered.metadata["children"] == ["c"]
    required = andrey.mmb_by_mmb(
        frame, target="b", knowledge=andrey.Knowledge(required=[("a", "b")])
    )
    assert required.metadata["parents"] == ["a"]
    forbidden = andrey.mmb_by_mmb(
        frame, target="b", knowledge=andrey.Knowledge(forbidden=[("a", "b"), ("b", "a")])
    )
    assert "a" not in forbidden.metadata["undetermined"] + forbidden.metadata["parents"]


def test_paths_are_not_honored():
    with pytest.raises(NotImplementedError, match="required_paths"):
        andrey.mmb_by_mmb(
            _confounded(), target="c", knowledge=andrey.Knowledge(required_paths=[("a", "e")])
        )
