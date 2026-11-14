"""The sequential engine of the target-centered local methods.

Under a d-separation oracle, the CPDAG rules return at every target of random DAGs the target's
edges and marks in the true CPDAG, and the PAG rules with R1 and R2 those in the true PAG over the
observed variables, hidden variables included; the walks of the papers' examples are reproduced, an
orientation that waits several steps away is found, and the stop rules end early with the answer
still right. Each step is checked alone: the local structure decides a pair at the node by the
node's blanket or, under the skeleton rule of LOAD, by the adjacencies of the pair's ends, the merge
keeps the node's edges and the v-structures that contain it, the orientation reads a non-adjacency
only where a test separated the pair, and the stop rules read the paths. On data the answer does not
depend on the column order or the columns' units.
"""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from andrey import data
from andrey.constraint.blanket import LocalSearch, local_search
from andrey.constraint.fci import circle_pag, fci_orient, orient_colliders
from andrey.constraint.sequential import (
    LOAD,
    MB_BY_MB,
    MMB_BY_MMB,
    Cache,
    LocalGraph,
    Network,
    Rules,
    extend,
    learn_local,
    merge,
    open_reach,
    orient,
    sequential,
    stop_rule,
    unknown_triples,
)
from andrey.core import ARROW, CIRCLE, NULL, TAIL
from andrey.core.ci import counting_tests
from andrey.core.orient import dag2cpdag
from andrey.data.latent import (
    DSepOracle,
    _AncestorSepsets,
    _ancestral_adjacency,
    causal_digraph,
    observed_ancestors,
    pag_projection,
)
from andrey.metrics import target_oracle

#: MMB-by-MMB with the stop rules R1 and R2, which stop only once the target's marks are decided.
MMB_R1_R2 = Rules(local="pc", expand="adjacent", orient="pag", stop="r1_r2")
#: MMB-by-MMB as its paper states it: every v-structure at the node merged, the stop rules read
#: after every step.
MMB_PAPER = Rules(local="pc", expand="adjacent", orient="pag", stop="r1_r3")


def _dag(edges, d: int) -> nx.DiGraph:
    dag = nx.DiGraph(edges)
    dag.add_nodes_from(range(d))
    return dag


def _search(dag: nx.DiGraph, ranks=None, max_depth=None) -> LocalSearch:
    """A search whose test is d-separation in ``dag``; column ``k`` is node ``k``."""
    d = len(ranks) if ranks is not None else dag.number_of_nodes()
    return LocalSearch(
        DSepOracle(dag), list(range(d)) if ranks is None else ranks, 0.05, max_depth=max_depth
    )


def _star(marks: np.ndarray, t: int) -> np.ndarray:
    out = np.zeros_like(marks)
    out[t], out[:, t] = marks[t], marks[:, t]
    return out


def _random_dag(seed: int, d: int, hidden: int = 0, mediated: bool = False) -> nx.DiGraph:
    """An Erdos-Renyi DAG over ``d + hidden`` nodes, the last ``hidden`` of them hidden: each a
    common parent of two observed nodes, so that it confounds them, and with ``mediated`` also a
    child of a third, earlier in a causal order, so that it carries that node's effect."""
    rng = np.random.default_rng(seed)
    truth = data.sample_scm(graph="erdos_renyi", d=d, n=2, seed=seed, density=2.5).graph
    dag = causal_digraph(truth)
    order = {v: k for k, v in enumerate(nx.topological_sort(dag))}
    for h in range(d, d + hidden):
        a, b, c = sorted(rng.choice(d, size=3, replace=False).tolist(), key=order.__getitem__)
        dag.add_edges_from([(h, b), (h, c)])
        if mediated:
            dag.add_edge(a, h)
    return dag


def _cpdag(dag: nx.DiGraph) -> np.ndarray:
    d = dag.number_of_nodes()
    marks = np.zeros((d, d), dtype=np.int8)
    for u, v in dag.edges:
        marks[u, v], marks[v, u] = TAIL, ARROW
    return dag2cpdag(marks)


def _pag(dag: nx.DiGraph, n_obs: int) -> np.ndarray:
    observed = range(n_obs)
    return pag_projection(dag, observed, observed_ancestors(dag, observed)).to_numpy()


def _ranks(seed: int, d: int) -> list[int]:
    """Ranks in a random order, so that the oracle's ties visit the variables in that order."""
    return np.random.default_rng(seed + 100).permutation(d).tolist()


# ---- the answer at every target ------------------------------------------------------------------


@pytest.mark.parametrize("rules", [MB_BY_MB, LOAD], ids=["mb_by_mb", "load"])
@pytest.mark.parametrize(("seed", "d"), [(s, 8 + s % 7) for s in range(12)] + [(40, 30), (41, 40)])
def test_the_cpdag_rules_give_the_true_cpdag_at_every_target(rules, seed, d):
    dag = _random_dag(seed, d)
    truth, adjacency = _cpdag(dag), nx.to_numpy_array(dag, nodelist=range(d))
    for target in range(d):
        result = sequential(_search(dag, _ranks(seed, d)), target, "iamb", rules)
        answer = result.network.star(target, d)
        assert np.array_equal(answer, _star(truth, target)), target
        assert np.array_equal(answer, target_oracle(adjacency, target, "cpdag").to_numpy())


@pytest.mark.parametrize(
    ("seed", "d", "hidden", "mediated"),
    [(s, 7 + s % 6, s % 4, s % 2 == 1) for s in range(14)]
    + [(50, 25, 4, False), (51, 20, 3, True)],
)
def test_the_pag_rules_give_the_true_pag_at_every_target(seed, d, hidden, mediated):
    dag = _random_dag(seed, d, hidden, mediated)
    truth, adjacency = _pag(dag, d), nx.to_numpy_array(dag, nodelist=range(d + hidden))
    hidden_nodes = range(d, d + hidden)
    for target in range(d):
        result = sequential(_search(dag, _ranks(seed, d)), target, "iamb", MMB_R1_R2)
        answer = result.network.star(target, d)
        assert np.array_equal(answer, _star(truth, target)), target
        oracle = target_oracle(adjacency, target, "pag", latent=hidden_nodes).to_numpy()
        assert np.array_equal(answer, oracle), target


def test_r3_stops_at_an_arrowhead_before_the_pag_decides_the_circle_unless_it_waits():
    # The PAG has 1 -> 4: the discriminating path 0 *-> 5 <-* 1, 5 -> 4 decides the circle.
    dag = _dag([(0, 3), (1, 3), (2, 3), (1, 4), (2, 4), (0, 5), (1, 5), (5, 4)], 6)
    assert _pag(dag, 6)[1, 4] == TAIL
    stopped = sequential(_search(dag), 1, "iamb", MMB_PAPER)
    assert stopped.stopped == "blocked" and stopped.done == [1]
    assert stopped.network.mark(1, 4) == CIRCLE
    # The target has three neighbors after its own step, so the guard reads R3 after four steps.
    guarded = sequential(_search(dag), 1, "iamb", MMB_BY_MMB)
    assert len(guarded.done) == 4
    full = sequential(_search(dag), 1, "iamb", MMB_R1_R2)
    for result in (guarded, full):
        assert np.array_equal(result.network.star(1, 6), _star(_pag(dag, 6), 1))


# ---- the papers' examples ------------------------------------------------------------------------

# MB-by-MB, Fig. 2(a) of Wang et al. (2014): 2 -> T -> 1, 3 -> 2, 4 -> 3 <- 5, 6 -> 4, 6 -> 5; the
# v-structure 4 -> 3 <- 5 two steps away orients both edges at T. T is column 0.
_WANG_FIG2 = [(2, 0), (0, 1), (3, 2), (4, 3), (5, 3), (6, 4), (6, 5)]


def test_mb_by_mb_walks_fig_2a_and_stops_once_the_edges_at_the_target_are_directed():
    result = sequential(_search(_dag(_WANG_FIG2, 7)), 0, "iamb", MB_BY_MB)
    assert result.done == [0, 1, 2, 3, 4] and result.learned == result.done
    assert result.network.mark(2, 0) == TAIL and result.network.mark(0, 2) == ARROW
    assert result.network.mark(0, 1) == TAIL and result.network.mark(1, 0) == ARROW
    assert result.stopped == "empty"


def test_mb_by_mb_stops_at_a_directed_edge_with_the_target_edges_undirected():
    # Fig. 2(a) with 2 -> 3: after MB(2), every path from T passes 2 -> 3 (condition 3).
    revised = [(2, 0), (0, 1), (2, 3), (4, 3), (5, 3), (6, 4), (6, 5)]
    result = sequential(_search(_dag(revised, 7)), 0, "iamb", MB_BY_MB)
    assert result.done == [0, 1, 2]
    expected = _star(_cpdag(_dag(revised, 7)), 0)
    assert np.array_equal(result.network.star(0, 7), expected)
    assert expected[0, 1] == expected[1, 0] == TAIL  # the class leaves 1 - T - 2 undirected


def test_mb_by_mb_stops_when_no_variable_is_left_to_learn():
    # Fig. 2(a) without 3 -> 2: T's component is 1 - T - 2 (condition 2).
    result = sequential(_search(_dag([(2, 0), (0, 1), (4, 3), (5, 3)], 6)), 0, "iamb", MB_BY_MB)
    assert result.done == [0, 1, 2]
    assert result.network.mark(0, 2) == TAIL and result.network.mark(2, 0) == TAIL


def test_an_orientation_several_steps_away_reaches_the_target():
    # p -> q <- r, q -> s -> t -> T: only the collider at q, three blankets away, orients t -> T.
    t_, t, s, q, p, r = range(6)
    dag = _dag([(p, q), (r, q), (q, s), (s, t), (t, t_)], 6)
    for rules in (MB_BY_MB, LOAD):
        result = sequential(_search(dag), t_, "iamb", rules)
        assert result.done == [t_, t, s, q]
        assert result.network.mark(t, t_) == TAIL and result.network.mark(t_, t) == ARROW


# MMB-by-MMB, Fig. 1 and Example 4 of Xie et al. (2024), the target V5. Columns: V2, V3, V4, V5,
# V7, V8, V9, V10, V11, V12; hidden V1 (into V3, V4) and V6 (into V5, V8). V7 -> V2 and V9 -> V7
# are not stated in the text; every independence its examples state holds.
_V = {name: k for k, name in enumerate([2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 1, 6])}
_XIE = [
    (11, 3), (1, 3), (1, 4), (2, 4), (7, 2), (9, 7), (4, 5), (6, 5), (6, 8), (5, 10), (3, 10),
    (7, 8), (12, 8),
]  # fmt: skip


@pytest.mark.parametrize("rules", [MMB_PAPER, MMB_R1_R2], ids=["r1_r3", "r1_r2"])
def test_mmb_by_mmb_walks_example_4(rules):
    dag = _dag([(_V[a], _V[b]) for a, b in _XIE], 12)
    search = _search(dag, list(range(10)))
    assert sorted(search.local("iamb", _V[5])[1]) == [_V[v] for v in (3, 4, 7, 8, 10, 12)]
    result = sequential(search, _V[5], "iamb", rules)
    assert result.done == [_V[5], _V[4]] and result.stopped == "determined"
    net = result.network
    assert (net.mark(_V[4], _V[5]), net.mark(_V[5], _V[4])) == (TAIL, ARROW)  # V4 -> V5
    assert (net.mark(_V[5], _V[10]), net.mark(_V[10], _V[5])) == (TAIL, ARROW)  # V5 -> V10
    assert (net.mark(_V[5], _V[8]), net.mark(_V[8], _V[5])) == (ARROW, ARROW)  # V5 <-> V8
    assert np.array_equal(net.star(_V[5], 10), _star(_pag(dag, 10), _V[5]))


# ---- the steps alone -----------------------------------------------------------------------------


def test_extend_appends_in_order_what_is_neither_waiting_nor_done():
    waitlist = [4]
    extend(waitlist, [3, 4, 5, 1, 5], done=[1])
    assert waitlist == [4, 3, 5]


@pytest.mark.parametrize("how", ["ic", "pc"])
def test_a_pair_at_the_node_is_decided_by_subsets_of_its_blanket(how):
    # Schubert et al. (2026) Ex. A.1: 0 and its spouse 4 are separated by {1, 2} only; 2 is not
    # adjacent to 0, so a search over 0's shrinking adjacencies keeps 0 - 4.
    dag = _dag([(0, 1), (0, 3), (1, 4), (2, 1), (2, 4), (4, 3)], 5)
    search = _search(dag)
    mb = search.local("iamb", 0)[1]
    local = learn_local(search, 0, mb, how)
    assert {p for p in local.adjacent if 0 in p} == {(0, 1), (0, 3)}
    assert {(0, 1, 2), (0, 3, 4)} <= local.colliders


def _at(local: LocalGraph, x: int):
    """The pairs that hold ``x`` and the v-structures that contain it."""
    return (
        {p for p in local.adjacent if x in p},
        {c for c in local.colliders if x in c},
    )


@pytest.mark.parametrize(("seed", "d"), [(s, 10 + s % 5) for s in range(12)])
def test_the_skeleton_rule_keeps_the_edges_and_v_structures_at_the_node_of_the_blanket_rule(
    seed, d
):
    # Under exact tests, testing a pair at the node against the adjacencies of both ends keeps the
    # node's edges and the v-structures that contain it, as testing it against its blanket does.
    dag = _random_dag(seed, d)
    search = _search(dag)
    for x in range(d):
        mb = search.local("iamb", x)[1]
        assert _at(learn_local(search, x, mb, "skeleton"), x) == _at(
            learn_local(search, x, mb, "pc"), x
        )


def test_the_skeleton_rule_separates_a_spouse_by_the_adjacencies_of_its_far_end():
    # Schubert et al. (2026) Ex. A.1: 0 and its spouse 4 are separated by {1, 2} only; 2 is not
    # adjacent to 0 but is to 4, so the far end's adjacencies hold the set.
    dag = _dag([(0, 1), (0, 3), (1, 4), (2, 1), (2, 4), (4, 3)], 5)
    search = _search(dag)
    local = learn_local(search, 0, search.local("iamb", 0)[1], "skeleton")
    assert {p for p in local.adjacent if 0 in p} == {(0, 1), (0, 3)}
    assert {(0, 1, 2), (0, 3, 4)} <= local.colliders


def test_the_skeleton_rule_tests_far_fewer_sets_where_the_blanket_is_wide():
    # 0 has two children, each with six other parents: |MB(0)| is 14 and no variable has more than
    # seven neighbors. An adjacent pair at 0 passes every subset of its blanket, 2 ** 13 of them,
    # while the adjacencies of its ends hold at most 2 ** 6.
    edges = [(0, c) for c in (1, 2)]
    edges += [(p, 1) for p in range(3, 9)] + [(p, 2) for p in range(9, 15)]
    dag = _dag(edges, 15)
    counts, found = {}, {}
    for how in ("pc", "skeleton"):
        search = _search(dag)
        mb = search.local("iamb", 0)[1]
        with counting_tests() as tests:
            found[how] = _at(learn_local(search, 0, mb, how), 0)
        counts[how] = tests.n
    assert found["skeleton"] == found["pc"]
    assert counts["skeleton"] * 10 < counts["pc"]


@pytest.mark.parametrize(("seed", "d", "hidden"), [(s, 10 + s % 5, s % 4) for s in range(12)])
def test_the_smaller_blanket_keeps_the_answer_of_the_node_blanket_at_every_target(seed, d, hidden):
    # Under exact tests a pair x - y is separated by a subset of MB(y) without x exactly when a
    # subset of MB(x) without y separates it, with hidden variables too, so the answer is the same.
    dag = _random_dag(seed, d, hidden, mediated=seed % 2 == 1)
    truth = _pag(dag, d)
    for target in range(d):
        stars = [
            sequential(_search(dag, _ranks(seed, d)), target, "iamb", rules).network.star(target, d)
            for rules in (MMB_BY_MMB, replace(MMB_BY_MMB, local="pc"))
        ]
        assert np.array_equal(*stars), target
        want = _star(truth, target)
        assert np.array_equal(stars[0] != NULL, want != NULL), target
        assert np.all((stars[0] == want) | (stars[0] == CIRCLE)), target


def test_the_smaller_blanket_tests_far_fewer_sets_at_a_hub():
    # 0 has two children, each with six other parents: |MB(0)| is 14 and each child's blanket has
    # 7 members. An adjacent pair at 0 passes every subset of 0's blanket, 2 ** 13 of them, and
    # every subset of the child's, 2 ** 6.
    edges = [(0, 1), (0, 2)] + [(p, 1) for p in range(3, 9)] + [(p, 2) for p in range(9, 15)]
    dag = _dag(edges, 15)
    counts, found = {}, {}
    for how in ("pc", "pc_smaller"):
        search = _search(dag)
        mb = search.local("iamb", 0)[1]
        with counting_tests() as tests:
            local = learn_local(search, 0, mb, how, of=lambda v: search.local("iamb", v)[1])
            found[how] = _at(local, 0)
        counts[how] = tests.n
    assert found["pc_smaller"] == found["pc"]
    assert counts["pc_smaller"] * 5 < counts["pc"]


def test_the_collider_check_drops_a_v_structure_the_smaller_blanket_brings_in():
    # 1's blanket {2, 6, 7, 10}: 2 <- 3 -> 7 -> 1 makes 2 - 7 an edge of L_1, and MB(2) = {1, 3,
    # 10} separates 1 from 2 by {3}, a set without 7, so L_1 holds 1 *-> 7 <-* 2. 7's own step
    # leaves 7 and 2 apart, and the check drops it.
    edges = [(0, 4), (1, 6), (1, 10), (2, 10), (3, 2), (3, 7), (5, 0), (6, 8), (7, 1), (7, 5)]
    dag = _dag(edges + [(7, 9), (9, 0), (10, 6), (10, 9)], 11)
    search = _search(dag)
    mb = search.local("iamb", 1)[1]
    local = learn_local(search, 1, mb, "pc_smaller", of=lambda v: search.local("iamb", v)[1])
    assert (1, 7, 2) in local.colliders
    truth = _pag(dag, 11)
    result = sequential(_search(dag), 1, "iamb", MMB_BY_MMB)
    assert np.array_equal(result.network.star(1, 11), _star(truth, 1))


def test_the_smaller_blanket_falls_back_to_the_node_blanket():
    # A far blanket that does not hold x, as a learner can return on data, is not read.
    edges = [(0, 1), (0, 2)] + [(p, 1) for p in range(3, 9)] + [(p, 2) for p in range(9, 15)]
    dag = _dag(edges, 15)
    counts, found = {}, {}
    for how, of in (("pc", None), ("pc_smaller", lambda v: [])):
        search = _search(dag)
        mb = search.local("iamb", 0)[1]
        with counting_tests() as tests:
            found[how] = learn_local(search, 0, mb, how, of=of)
        counts[how] = tests.n
    assert found["pc_smaller"] == found["pc"]
    assert counts["pc_smaller"] == counts["pc"]


def test_load_learns_its_local_structures_by_the_skeleton_rule():
    edges = [(0, 1), (0, 2)] + [(p, 1) for p in range(3, 9)] + [(p, 2) for p in range(9, 15)]
    dag = _dag(edges, 15)
    counts, marks = {}, {}
    for name, rules in (("load", LOAD), ("blanket", replace(LOAD, local="pc"))):
        search = _search(dag)
        result = sequential(search, 0, "gs", rules)
        counts[name], marks[name] = result.n_tests, result.network.marks.copy()
    assert np.array_equal(marks["load"], marks["blanket"])
    assert counts["load"] * 10 < counts["blanket"]


@pytest.mark.parametrize("how", ["ic", "pc"])
def test_the_merge_keeps_the_node_edges_and_its_v_structures_alone(how):
    # Wang et al. (2014) Fig. 1(a): 1 -> T <- 2, T -> 3 <- 4, 4 -> 2, and 1 -> 5 -> 2 outside
    # MB+(T), so L_T has the edge 1 - 2 and the v-structure 1 -> 2 <- 4, both wrong, and misses
    # 1 -> T <- 2.
    t = 0
    dag = _dag([(1, t), (2, t), (t, 3), (4, 3), (4, 2), (1, 5), (5, 2)], 6)
    search = _search(dag)
    local = learn_local(search, t, search.local("iamb", t)[1], how)
    assert (1, 2) in local.adjacent and (1, 2, 4) in local.colliders
    network = Network(TAIL)
    merge(network, local, t)
    edges = {(u, v) for u in network.nodes for v in network.neighbors(u) if u < v}
    assert edges == {(0, 1), (0, 2), (0, 3), (3, 4)}
    assert network.mark(3, 0) == ARROW and network.mark(3, 4) == ARROW
    assert network.mark(0, 1) == network.mark(0, 2) == TAIL


def test_a_substructure_keeps_the_pairs_and_colliders_inside():
    local = LocalGraph((0, 1, 2, 3), frozenset({(0, 1), (1, 2), (2, 3)}), frozenset({(0, 1, 2)}))
    inside = LocalGraph((1, 2, 3), frozenset({(1, 2), (2, 3)}), frozenset())
    assert local.restrict([1, 2, 3]) == inside


def _chain(end: int) -> Network:
    """``0 -> 1 - 2`` (``0 *-> 1`` in a PAG), the pair ``(0, 2)`` without an edge."""
    network = Network(end)
    network.add(0, 1)
    network.add(1, 2)
    network.arrow(1, 0)
    if end == TAIL:
        network.marks[0, 1] = TAIL
    return network


@pytest.mark.parametrize("end", [TAIL, CIRCLE], ids=["meek", "pag"])
def test_orientation_reads_a_non_adjacency_only_where_a_test_separated_the_pair(end):
    how = "meek" if end == TAIL else "pag"
    search = _search(_dag([(0, 1), (1, 2)], 3))
    unknown = _chain(end)
    assert unknown_triples(unknown, {}) == {(0, 1, 2)}
    orient(unknown, {}, how, search)
    assert unknown.mark(1, 2) == end  # 0 and 2 may be adjacent: R1 does not fire
    collider = _chain(end)
    orient(collider, {(0, 2): (3,)}, how, search)
    assert collider.mark(1, 2) == end  # separated without 1: not a non-collider
    separated = _chain(end)
    orient(separated, {(0, 2): (1,)}, how, search)
    assert (separated.mark(1, 2), separated.mark(2, 1)) == (TAIL, ARROW)


def test_fci_orient_reads_every_pair_without_an_edge_as_apart_by_default():
    for seed in range(8):
        dag = _random_dag(seed, 10, 2)
        observed = range(10)
        support = _ancestral_adjacency(dag, observed, ())
        sepsets = _AncestorSepsets(support, observed, observed_ancestors(dag, observed), set())
        marks = circle_pag(support)
        orient_colliders(marks, sepsets)
        default, apart = marks.copy(), marks.copy()
        test = DSepOracle(dag, observed)
        fci_orient(default, test, 0.5, sepsets, selection_bias=False)
        known = marks == NULL
        np.fill_diagonal(known, False)
        fci_orient(apart, test, 0.5, sepsets, selection_bias=False, apart=known)
        assert np.array_equal(default, apart)


def _network(end: int, edges) -> Network:
    """A network with each ``(u, v, mark at u, mark at v)``."""
    network = Network(end)
    for u, v, mu, mv in edges:
        network.add(u, v)
        network.marks[network.index[u], network.index[v]] = mu
        network.marks[network.index[v], network.index[u]] = mv
    return network


def test_open_reach_follows_the_edges_without_an_arrowhead():
    # T - 1 -> 2 - 3, and T - 4 <- 5: a directed edge either way blocks the path.
    network = _network(TAIL, [(0, 1, TAIL, TAIL), (1, 2, TAIL, ARROW), (2, 3, TAIL, TAIL),
                              (0, 4, TAIL, TAIL), (4, 5, ARROW, TAIL)])  # fmt: skip
    assert open_reach(network, 0) == {0, 1, 4}


def test_the_stop_rules():
    # Xie et al. (2024) Example 3: T o-> V1 is blocked by its own arrowhead (R3).
    assert stop_rule(_network(CIRCLE, [(0, 1, CIRCLE, ARROW)]), 0, [1]) == "blocked"
    assert stop_rule(_network(CIRCLE, [(0, 1, CIRCLE, ARROW)]), 0, [1], r3=False) is None
    assert stop_rule(_network(CIRCLE, [(0, 1, CIRCLE, CIRCLE)]), 0, [1]) is None
    # A path walks on while the far mark is not an arrowhead, and an arrowhead into it does not
    # block it: T o-o 1 <-o 2 reaches 2.
    walked = _network(CIRCLE, [(0, 1, CIRCLE, CIRCLE), (1, 2, ARROW, CIRCLE)])
    assert stop_rule(walked, 0, [2]) is None and stop_rule(walked, 0, [3]) == "blocked"
    assert stop_rule(_network(CIRCLE, [(0, 1, TAIL, ARROW)]), 0, [1]) == "determined"  # R1
    assert stop_rule(_network(TAIL, [(0, 1, TAIL, ARROW)]), 0, [1]) == "determined"
    assert stop_rule(_network(CIRCLE, [(0, 1, CIRCLE, CIRCLE)]), 0, []) == "empty"  # R2


def test_rules_refuse_unknown_choices_and_a_learner_of_parents_and_children():
    with pytest.raises(ValueError, match="stop"):
        Rules(stop="never")
    with pytest.raises(ValueError, match="colliders"):
        Rules(colliders="never")
    with pytest.raises(ValueError, match="stop_after"):
        Rules(stop_after="degree")
    with pytest.raises(ValueError, match="own_blanket"):
        Rules(local="pc_smaller")
    with pytest.raises(ValueError, match="parents and children"):
        sequential(_search(_dag([(0, 1)], 2)), 0, "mmpc")


# ---- caches and counts ---------------------------------------------------------------------------


@pytest.mark.parametrize("rules", [MB_BY_MB, LOAD, MMB_R1_R2], ids=["mb_by_mb", "load", "mmb"])
def test_a_cache_carried_across_targets_changes_no_answer_and_learns_each_blanket_once(rules):
    d = 14
    dag = _random_dag(3, d, 2 if rules.orient == "pag" else 0)
    search, cache = _search(dag, list(range(d))), Cache()
    learned = []
    for target in range(d):
        shared = sequential(search, target, "iamb", rules, cache)
        fresh = sequential(_search(dag, list(range(d))), target, "iamb", rules)
        assert np.array_equal(shared.network.star(target, d), fresh.network.star(target, d))
        assert shared.cache is cache and set(shared.learned) <= set(fresh.done)
        learned += shared.learned
    assert len(learned) == len(set(learned)) == len(cache.blankets)
    assert set(cache.local) <= set(cache.blankets)


def test_the_counts_are_the_tests_computed_and_the_blankets_learned():
    dag = _dag(_WANG_FIG2, 7)
    with counting_tests() as outer:
        result = sequential(_search(dag), 0, "iamb", MB_BY_MB)
    assert result.n_tests == outer.n > 0
    again = sequential(_search(dag), 0, "iamb", MB_BY_MB)
    assert again.n_tests == result.n_tests and again.done == result.done


# ---- data ----------------------------------------------------------------------------------------


def _answer(x: np.ndarray, target: int, rules: Rules) -> np.ndarray:
    search = local_search(x, alpha=0.05, indep_test="fisherz")
    return sequential(search, target, "iamb", rules).network.star(target, x.shape[1])


@pytest.mark.parametrize("rules", [MB_BY_MB, LOAD, MMB_BY_MMB], ids=["mb_by_mb", "load", "mmb"])
def test_the_answer_does_not_depend_on_the_column_order_or_the_units(rules):
    d = 12
    x = data.sample_scm(graph="erdos_renyi", d=d, n=1500, seed=4, density=3.0).data
    perm = np.random.default_rng(0).permutation(d)
    scale = np.random.default_rng(1).uniform(0.01, 100.0, d)
    for target in range(d):
        answer = _answer(x, target, rules)
        shuffled = _answer(x[:, perm], int(np.flatnonzero(perm == target)[0]), rules)
        assert np.array_equal(shuffled, answer[np.ix_(perm, perm)]), target
        assert np.array_equal(_answer(x * scale, target, rules), answer), target
