"""Checks of ``cmb``, Causal Markov Blanket discovery (Gao and Ji 2015).

Under a d-separation oracle every edge and every direction the answer gives at a target of random
graphs is the true CPDAG's, with each of the three blanket learners; the paper's two examples come
out as the CPDAG at every target, and its case study learns the blankets it names, in its order.
The rule for a neighbor of two parents fires only when that neighbor is not a collider between
them. Edges the CPDAG directs by a test the steps never run, or by acyclicity, stay undirected.
On data the answer
does not depend on the column order or the columns' units, and the result holds the target's sets,
the blankets learned, and the test count.
"""

from __future__ import annotations

import itertools
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import andrey
from andrey import data
from andrey.constraint.blanket import LocalSearch
from andrey.constraint.cmb import CHILD, PARENT, UNDECIDED, cmb
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


def _dag(edges, names: str) -> nx.DiGraph:
    """The DAG over the one-letter nodes ``names`` with ``edges`` given as two-letter strings."""
    dag = nx.DiGraph([(names.index(a), names.index(b)) for a, b in edges])
    dag.add_nodes_from(range(len(names)))
    return dag


def _truth(dag: nx.DiGraph, target: int) -> np.ndarray:
    adjacency = nx.to_numpy_array(dag, nodelist=range(dag.number_of_nodes()))
    return target_oracle(adjacency, target, "cpdag").to_numpy()


def _codes(dag: nx.DiGraph, target: int, ranks: list[int], blanket: str = "hiton_mb"):
    """The engine under the oracle with the ranks ``ranks``."""
    return cmb(LocalSearch(DSepOracle(dag), ranks, 0.5), target, blanket)


def _star(codes: dict[int, int], target: int, d: int) -> np.ndarray:
    """The target's edges as the CPDAG's endpoint marks (1 tail, 2 arrowhead) hold them."""
    star = np.zeros((d, d), dtype=int)
    for x, code in codes.items():
        star[target, x] = 2 if code == PARENT else 1
        star[x, target] = 2 if code == CHILD else 1
    return star


# ---- the answer under the oracle -----------------------------------------------------------------


def _oracle_fit(box: dict, dag: nx.DiGraph, target: int, seed: int, **settings):
    """``cmb`` under the oracle; the random columns give the ranks that break the oracle's ties."""
    box["dag"] = dag
    x = np.random.default_rng(seed).standard_normal((40, dag.number_of_nodes()))
    return andrey.cmb(x, target=target, indep_test="dsep", **settings)


@pytest.mark.parametrize(
    ("blanket", "seed"),
    [("hiton_mb", s) for s in range(12)] + [("mmmb", 20), ("pcmb", 21)],
)
def test_every_edge_and_direction_under_the_oracle_is_the_cpdags(dsep, blanket, seed):
    d = 8 + seed % 7
    truth = data.sample_scm(graph="erdos_renyi", d=d, n=2, seed=seed, density=2.5).graph
    dag = causal_digraph(truth)
    for target in range(d):
        out = _oracle_fit(dsep, dag, target, seed, blanket=blanket)
        answer, cpdag = out.structure.to_numpy(), _truth(dag, target)
        assert np.array_equal(answer != 0, cpdag != 0), target
        directed = (answer[target] != answer[:, target]) & (answer[target] != 0)
        assert np.array_equal(answer[target, directed], cpdag[target, directed]), target
        assert np.array_equal(answer[directed, target], cpdag[directed, target]), target
        meta = out.metadata
        assert meta["blankets_learned"][0] == target
        assert len(set(meta["blankets_learned"])) == len(meta["blankets_learned"])
        children = set(dag.succ[target])
        spouses = {p for c in children for p in dag.pred[c]}
        assert set(meta["markov_blanket"]) == (set(dag.pred[target]) | children | spouses) - {
            target
        }


FIG_1A = _dag(["AC", "BC", "CD", "DG", "DE", "EF", "FG"], "ABCDEFG")
FIG_1B = _dag(["DA", "DT", "DB", "ED", "ET", "AC", "TC", "BC"], "ABCDET")


@pytest.mark.parametrize("dag", [FIG_1A, FIG_1B], ids=["fig_1a", "fig_1b"])
def test_the_papers_examples_give_the_cpdag_at_every_target(dag):
    d = dag.number_of_nodes()
    for ranks in ([*range(d)], [*range(d)][::-1]):
        for target in range(d):
            result = _codes(dag, target, ranks)
            assert np.array_equal(_star(result.codes, target, d), _truth(dag, target)), target


def test_the_case_study_learns_the_blankets_of_e_d_and_c():
    # Section 3: at E, D and F are undecided; at D, C, E, and G are; C finds A and B its parents
    # and D its child, so D finds C its parent and E its child, and E finds D its parent, F its
    # child.
    a, b, c, d, e, f, _ = range(7)
    result = _codes(FIG_1A, e, list(range(7)))
    assert result.codes == {d: PARENT, f: CHILD}
    assert result.learned == [e, d, c]


def test_a_neighbor_two_parents_point_into_is_not_made_a_parent():
    # x -> t <- y and x -> z <- y with z adjacent to t: z is a collider between the parents, so
    # Meek's rule R3 does not apply and z - t stays undirected, whatever the rank order.
    for zt in (("z", "t"), ("t", "z")):
        dag = _dag(["xt", "yt", "xz", "yz", "".join(zt)], "zxyt")
        for ranks in itertools.permutations(range(4)):
            assert _codes(dag, 3, list(ranks)).codes == {0: UNDECIDED, 1: PARENT, 2: PARENT}


def test_a_neighbor_between_two_parents_is_made_a_parent():
    # z -> x, z -> y, z -> t, x -> t <- y: z separates the parents, so R3 directs z -> t.
    dag = _dag(["zx", "zy", "zt", "xt", "yt"], "zxyt")
    for ranks in itertools.permutations(range(4)):
        assert _codes(dag, 3, list(ranks)).codes == {0: PARENT, 1: PARENT, 2: PARENT}


def test_edges_the_cpdag_directs_can_stay_undirected():
    # k -> j <- i, j -> t, i -> t. The CPDAG directs j -> t (k _||_ t given i and j) and then
    # i -> t (acyclicity, Meek's rule R2). The steps have no acyclicity rule, so i - t stays
    # undecided in every rank order; in the column order they also never test k and t given i and
    # j, so j - t stays undecided too. The edges are still the CPDAG's.
    dag = _dag(["kj", "ij", "jt", "it"], "kijt")
    assert _codes(dag, 3, [0, 1, 2, 3]).codes == {1: UNDECIDED, 2: UNDECIDED}
    for ranks in itertools.permutations(range(4)):
        assert _codes(dag, 3, list(ranks)).codes[1] == UNDECIDED
    truth = _truth(dag, 3)
    assert truth[3, 1] == truth[3, 2] == 2 and truth[1, 3] == truth[2, 3] == 1


# ---- the result on data --------------------------------------------------------------------------


def _collider() -> pd.DataFrame:
    """a -> c <- b, c -> e: the target c has two parents and a child."""
    rng = np.random.default_rng(0)
    a, b, u, v = rng.standard_normal((4, 2000))
    c = a + b + 0.5 * u
    return pd.DataFrame({"a": a, "b": b, "c": c, "e": c + 0.5 * v})


def test_the_result_names_the_target_sets_by_label():
    out = andrey.cmb(_collider(), target="c")
    meta = out.metadata
    assert out.structure.kind == "cpdag"
    assert (meta["target"], meta["parents"], meta["children"], meta["undirected"]) == (
        "c",
        ["a", "b"],
        ["e"],
        [],
    )
    assert sorted(meta["markov_blanket"]) == ["a", "b", "e"]
    assert meta["blankets_learned"] == ["c"] and meta["n_tests"] > 0
    assert sorted(out.structure.oriented_edges()) == [
        ("a", "c", "directed"),
        ("b", "c", "directed"),
        ("c", "e", "directed"),
    ]


def test_a_pair_past_the_cap_decides_nothing():
    # With max_depth=0 every test with the target conditions on one variable too many, so no pair
    # decides a direction; uncapped, a and b are parents and e a child.
    capped = andrey.cmb(_collider(), target="c", max_depth=0).metadata
    assert capped["parents"] == capped["children"] == []
    assert sorted(capped["undirected"]) == ["a", "b", "e"]


def test_a_target_is_required():
    with pytest.raises(ValueError, match="cmb: pass target"):
        andrey.cmb(_collider())


@pytest.mark.parametrize("blanket", ["iamb", "hiton_pc", "iamb_forward"])
def test_a_learner_without_both_sets_is_refused(blanket):
    with pytest.raises(ValueError, match="blanket"):
        andrey.cmb(_collider(), target="c", blanket=blanket)


def _answer(x: np.ndarray, target: int, **settings) -> np.ndarray:
    return andrey.cmb(x, target=target, **settings).structure.to_numpy()


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
    first = andrey.cmb(x, target=4)
    again = andrey.cmb(x, target=4)
    assert np.array_equal(first.structure.to_numpy(), again.structure.to_numpy())
    assert first.metadata == again.metadata


def test_a_tier_directs_the_edge_and_the_tests_direct_the_next():
    # a -> b -> c: the chain is one class, so the edges at b stay undirected without knowledge;
    # a tier makes a a parent, and the pair (a, c), independent given b, then makes c a child.
    rng = np.random.default_rng(1)
    a = rng.standard_normal(2000)
    b = a + 0.5 * rng.standard_normal(2000)
    frame = pd.DataFrame({"a": a, "b": b, "c": b + 0.5 * rng.standard_normal(2000)})
    plain = andrey.cmb(frame, target="b")
    assert sorted(plain.metadata["undirected"]) == ["a", "c"]
    tiered = andrey.cmb(frame, target="b", knowledge=andrey.Knowledge(tiers=[["a"], ["b"]]))
    assert tiered.metadata["parents"] == ["a"] and tiered.metadata["children"] == ["c"]
    forbidden = andrey.cmb(
        frame, target="b", knowledge=andrey.Knowledge(forbidden=[("a", "b"), ("b", "a")])
    )
    assert "a" not in forbidden.metadata["undirected"] + forbidden.metadata["parents"]


def test_paths_are_not_honored():
    with pytest.raises(NotImplementedError, match="required_paths"):
        andrey.cmb(_collider(), target="c", knowledge=andrey.Knowledge(required_paths=[("a", "e")]))
