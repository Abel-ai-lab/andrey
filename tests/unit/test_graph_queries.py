"""Graph queries on ``GraphStructure``: each answer checked against an independent definition."""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest

import andrey.data as ad
from andrey import GraphStructure
from andrey.core.structure import ARROW, EDGE_DTYPE
from andrey.data.latent import DSepOracle, _causal_digraph
from andrey.metrics import to_cpdag


def _dag(d: int, seed: int) -> GraphStructure:
    scm = ad.SCM(
        graph=ad.graphs.erdos_renyi(d=d, avg_degree=2.5),
        functional=ad.functional.linear(),
        noise=ad.noise.gaussian(),
    )
    return scm.sample(n=2, seed=seed).graph


def _queries(d: int):
    """Every (x, y, given) of single nodes with given disjoint from both."""
    for x, y in itertools.combinations(range(d), 2):
        rest = [v for v in range(d) if v not in (x, y)]
        for r in range(len(rest) + 1):
            for given in itertools.combinations(rest, r):
                yield x, y, list(given)


def _open_path(graph: GraphStructure, x: int, y: int, given: list[int]) -> bool:
    """Whether a simple path between x and y is open given ``given``, by enumerating paths."""
    M = graph.to_numpy()
    d = graph.n_nodes
    directed = {(a, b) for a in range(d) for b in range(d) if M[a, b] == 1 and M[b, a] == ARROW}
    g = nx.DiGraph(directed)
    g.add_nodes_from(range(d))
    opened = set(given).union(*(nx.ancestors(g, z) for z in given))

    def extend(path: list[int], entered_arrow: bool) -> bool:
        v = path[-1]
        if v == y:
            return True
        for w in np.flatnonzero(M[v]).tolist():
            if w in path:
                continue
            if len(path) > 1:
                collider = entered_arrow and M[v, w] == ARROW
                if (v not in opened) if collider else (v in given):
                    continue
            if extend([*path, w], M[w, v] == ARROW):
                return True
        return False

    return extend([x], False)


@pytest.mark.parametrize("seed", range(6))
def test_d_separation_matches_the_oracle_and_path_enumeration(seed):
    dag = _dag(6, seed)
    oracle = DSepOracle(_causal_digraph(dag))
    for x, y, given in _queries(6):
        answer = dag.d_separated(x, y, given)
        assert answer == (oracle(x, y, given) == 1.0)
        assert answer == (not _open_path(dag, x, y, given))


@pytest.mark.parametrize("seed", range(6))
def test_mag_and_pag_separation_match_the_dag_with_hidden_nodes(seed):
    """m-separation in the MAG, and in the PAG, of a DAG with hidden nodes is d-separation in the
    DAG among the observed nodes (Richardson and Spirtes 2002; Zhang 2008)."""
    dag = _dag(7, seed)
    hidden = [0, 1]
    observed = [v for v in range(7) if v not in hidden]
    oracle = DSepOracle(_causal_digraph(dag))
    mag, pag = dag.to_mag(latent=hidden), dag.to_pag(latent=hidden)
    for x, y, given in _queries(len(observed)):
        truth = oracle(observed[x], observed[y], [observed[z] for z in given]) == 1.0
        assert mag.d_separated(x, y, given) == truth
        assert pag.d_separated(x, y, given) == truth
        assert truth == (not _open_path(mag, x, y, given))


@pytest.mark.parametrize("seed", range(4))
def test_selection_variables_join_the_conditioning_set(seed):
    """With selection nodes S, m-separation given Z in the MAG and the PAG is d-separation given
    Z and S in the DAG (Richardson and Spirtes 2002)."""
    dag = _dag(7, seed)
    hidden, selection = [0], [6]
    observed = [1, 2, 3, 4, 5]
    oracle = DSepOracle(_causal_digraph(dag))
    mag = dag.to_mag(latent=hidden, selection=selection)
    pag = dag.to_pag(latent=hidden, selection=selection)
    for x, y, given in _queries(len(observed)):
        z = [observed[k] for k in given] + selection
        truth = oracle(observed[x], observed[y], z) == 1.0
        assert mag.d_separated(x, y, given) == truth
        assert pag.d_separated(x, y, given) == truth


def test_a_graph_that_is_not_ancestral_is_refused():
    # 2 -> 0 -> 1 and 1 <-> 2: a bidirected edge whose end 2 is an ancestor of its end 1.
    two = np.zeros((3, 3), dtype=np.int8)
    two[0, 1], two[1, 0] = 1, ARROW  # 0 -> 1
    two[1, 2], two[2, 1] = ARROW, ARROW  # 1 <-> 2
    two[0, 2], two[2, 0] = ARROW, 1  # 2 -> 0
    with pytest.raises(ValueError, match="not ancestral"):
        GraphStructure.from_numpy(two, kind="mag").d_separated(0, 1)


@pytest.mark.parametrize("seed", range(4))
def test_a_cpdag_answers_as_every_dag_of_its_class(seed):
    dag = _dag(6, seed)
    cpdag = to_cpdag(dag)
    members = [dag, *(cpdag.to_dag(seed=s) for s in range(3))]
    for x, y, given in _queries(6):
        assert len({g.d_separated(x, y, given) for g in [cpdag, *members]}) == 1
    for v in range(6):
        assert len({tuple(g.markov_blanket(v)) for g in [cpdag, *members]}) == 1


@pytest.mark.parametrize("kind", ["dag", "mag", "cpdag", "pag"])
def test_markov_blanket_is_the_smallest_separating_set(kind):
    """Given its blanket, a node is separated from every other node, and every member is needed:
    without it, the node and that member are not separated."""
    for seed in range(4):
        dag = _dag(7, seed)
        graph = {
            "dag": dag,
            "cpdag": to_cpdag(dag),
            "mag": dag.to_mag(latent=[0, 1]),
            "pag": dag.to_pag(latent=[0, 1]),
        }[kind]
        for v in range(graph.n_nodes):
            blanket = graph.markov_blanket(v)
            rest = [w for w in range(graph.n_nodes) if w != v and w not in blanket]
            if rest:
                assert graph.d_separated(v, rest, blanket)
            for w in blanket:
                assert not graph.d_separated(v, w, [b for b in blanket if b != w])


def test_dag_queries_match_networkx():
    for seed in range(4):
        dag = _dag(8, seed)
        g = _causal_digraph(dag)
        for v in range(8):
            assert dag.parents(v) == sorted(g.predecessors(v))
            assert dag.children(v) == sorted(g.successors(v))
            assert dag.ancestors(v) == sorted(nx.ancestors(g, v))
            assert dag.descendants(v) == sorted(nx.descendants(g, v))
            spouses = {p for c in g.successors(v) for p in g.predecessors(c)} - {v}
            assert dag.markov_blanket(v) == sorted(
                set(g.predecessors(v)) | set(g.successors(v)) | spouses
            )


def test_directed_edges_alone_make_parents_and_ancestors():
    # a -> b -- c, b -> d: the undirected edge makes no parent, child, or ancestor.
    M = np.zeros((4, 4), dtype=np.int8)
    M[0, 1], M[1, 0] = 1, 2
    M[1, 2], M[2, 1] = 1, 1
    M[1, 3], M[3, 1] = 1, 2
    g = GraphStructure.from_numpy(M, kind="cpdag", labels=("a", "b", "c", "d"))
    assert g.parents("b") == ["a"] and g.children("b") == ["d"]
    assert g.ancestors("d") == ["a", "b"] and g.descendants("c") == []
    assert g.parents("c") == [] and g.neighbors(2).tolist() == [1]


def test_digraph_two_cycles_and_self_loops():
    # 0 -> 1 and 1 -> 0, 1 -> 2, and the self-loop 2 -> 2.
    edges = np.array([(0, 1, ARROW, ARROW), (1, 2, 1, ARROW), (2, 2, 1, ARROW)], dtype=EDGE_DTYPE)
    g = GraphStructure.from_edges(edges, n_nodes=3, kind="digraph")
    assert g.parents(0) == [1] and g.children(0) == [1]
    assert g.parents(2) == [1, 2] and g.children(2) == [2]
    assert g.ancestors(0) == [1] and g.descendants(0) == [1, 2] and g.descendants(2) == []
    with pytest.raises(ValueError, match="digraph"):
        g.d_separated(0, 2)
    with pytest.raises(ValueError, match="digraph"):
        g.markov_blanket(0)


def test_undirected_graph_separation_and_blanket():
    M = np.zeros((4, 4), dtype=np.int8)
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        M[a, b] = M[b, a] = 1
    g = GraphStructure.from_numpy(M, kind="undirected")
    assert g.d_separated(0, 3, [2]) and not g.d_separated(0, 3)
    assert g.markov_blanket(1) == [0, 2]
    assert g.parents(1) == [] and g.ancestors(3) == []


def test_node_arguments():
    M = np.array([[0, 1, 0], [2, 0, 2], [0, 1, 0]])  # a -> b <- c
    g = GraphStructure.from_numpy(M, labels=("a", "b", "c"))
    assert g.parents("b") == g.parents(1) == g.parents(np.int64(1)) == ["a", "c"]
    assert g.parents("b", index=True) == [0, 2]
    assert g.neighbors("b").tolist() == [0, 2] and g.endpoints("a", "b") == g.endpoints(0, 1)
    assert g.d_separated({"a"}, ["c"]) and not g.d_separated("a", "c", given=("b",))
    with pytest.raises(ValueError, match="lacks"):
        g.parents("z")
    with pytest.raises(ValueError, match="3 nodes"):
        g.parents(3)
    with pytest.raises(TypeError):
        g.parents(True)
    with pytest.raises(TypeError):
        g.parents(["a"])
    with pytest.raises(ValueError, match="disjoint"):
        g.d_separated("a", "c", given=["a"])
    with pytest.raises(ValueError, match="more than once"):
        g.d_separated(["a", "a"], "c")


def test_a_dag_with_a_directed_cycle_is_refused():
    M = np.zeros((3, 3), dtype=np.int8)
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        M[a, b], M[b, a] = 1, ARROW
    g = GraphStructure.from_numpy(M, kind="dag")
    with pytest.raises(ValueError, match="directed cycle"):
        g.d_separated(0, 1)


def test_answers_follow_labels_not_column_order():
    dag = _dag(6, 0)
    labels = tuple("uvwxyz")
    named = GraphStructure.from_numpy(dag.to_numpy(), kind="dag", labels=labels)
    perm = np.random.default_rng(0).permutation(6)
    moved = GraphStructure.from_numpy(
        dag.to_numpy()[np.ix_(perm, perm)], kind="dag", labels=tuple(labels[k] for k in perm)
    )
    for v in labels:
        assert sorted(named.markov_blanket(v)) == sorted(moved.markov_blanket(v))
        assert sorted(named.ancestors(v)) == sorted(moved.ancestors(v))
    for x, y, given in _queries(6):
        names = [labels[k] for k in given]
        assert named.d_separated(labels[x], labels[y], names) == moved.d_separated(
            labels[x], labels[y], names
        )
