"""Recovery tests for ``snap``: the examples and theorems of Schubert, Claassen, and Magliacane
(2025) under exact d-separation, and answers on linear-Gaussian data drawn from the paper's random
graphs by the benchmark lane's sampler, scored against the truth.

With exact tests (:class:`~andrey.data.latent.DSepOracle`), every order of the search keeps a
supergraph of the true skeleton over the variables left and a possibly ancestral set holding the
possible ancestors of the targets (Theorem 3.1); run to the end, it returns the true CPDAG over
exactly those possible ancestors (Theorem 3.2), in any visiting order. The records it reads off
that CPDAG equal the ones :func:`~andrey.metrics.effect_oracle` gives from the true DAG: the
optimal set where the class identifies the effect, the local parent sets where it does not.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from itertools import permutations
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from andrey import GraphStructure, snap
from andrey.constraint import snap as engine
from andrey.constraint.blanket import LocalSearch
from andrey.core import adjustment as adj
from andrey.core import ci
from andrey.core.orient import dag2cpdag
from andrey.core.structure import ARROW
from andrey.data.latent import DSepOracle
from andrey.metrics import effect_oracle, effect_scores, pooled_answer_scores

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")

#: Figure 1a: A and B share the causes C and D and have one more cause each, U and V.
FIGURE_1A = [("U", "A"), ("C", "A"), ("D", "A"), ("C", "B"), ("D", "B"), ("V", "B")]

#: Figure 1b: the graph of Example 3.1, where PC's v-structure rule after a search to order 3 can
#: make the edge A -> B bidirected.
FIGURE_1B = [
    *(("G", "E"), ("G", "X"), ("G", "B"), ("C", "E"), ("C", "A"), ("C", "B"), ("E", "A")),
    *(("E", "B"), ("X", "U"), ("U", "A"), ("A", "B"), ("V", "B"), ("V", "U"), ("V", "F")),
    *(("D", "A"), ("D", "F"), ("D", "B"), ("F", "A"), ("F", "B"), ("H", "F"), ("H", "B")),
]


def random_dag(rng: np.random.Generator, d: int, degree: float, max_degree: int = 10) -> np.ndarray:
    """A DAG as the paper draws its graphs: edges with probability ``degree / (d - 1)`` in a random
    order, a node over ``max_degree`` losing random incident edges."""
    order = rng.permutation(d)
    A = np.zeros((d, d), dtype=bool)
    upper = np.triu(rng.random((d, d)) < degree / (d - 1), 1)
    for i, j in zip(*np.nonzero(upper), strict=True):
        A[order[i], order[j]] = True
    for v in range(d):
        while A[v].sum() + A[:, v].sum() > max_degree:
            u = rng.choice(np.flatnonzero(A[v] | A[:, v]))
            A[v, u] = A[u, v] = False
    return A


def dag_marks(A: np.ndarray) -> np.ndarray:
    return (np.where(A, 1, 0) + np.where(A.T, 2, 0)).astype(np.int8)


def truth(A: np.ndarray, targets) -> tuple[set[int], np.ndarray]:
    """The possible ancestors of ``targets`` in the CPDAG of ``A``, and that CPDAG over them."""
    cpdag = dag2cpdag(dag_marks(A))
    kept = adj.possible_ancestors(cpdag, targets)
    out = [v for v in range(len(A)) if v not in kept]
    cpdag[out, :] = cpdag[:, out] = 0
    return kept, cpdag


def oracle_search(edges_or_matrix, nodes=None, ranks=None) -> LocalSearch:
    if isinstance(edges_or_matrix, np.ndarray):
        graph = nx.DiGraph(list(zip(*np.nonzero(edges_or_matrix), strict=True)))
        graph.add_nodes_from(range(len(edges_or_matrix)))
        d = len(edges_or_matrix)
        oracle = DSepOracle(graph)
    else:
        graph = nx.DiGraph(edges_or_matrix)
        d = len(nodes)
        oracle = DSepOracle(graph, nodes=list(nodes))
    return LocalSearch(oracle, list(range(d)) if ranks is None else list(ranks), 0.5)


def matrix(edges, nodes) -> np.ndarray:
    ix = {v: k for k, v in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=bool)
    for a, b in edges:
        A[ix[a], ix[b]] = True
    return A


# --- the paper's figures ------------------------------------------------------------------------


def test_figure_1a_conflicting_v_structures_at_order_0():
    """At order 0, ``U -> A <- B`` and ``A -> B <- V`` disagree on the edge ``A - B``, which ends
    with two arrowheads: neither is an ancestor of the other, so with A a target B is dropped."""
    nodes = ["A", "B", "C", "D", "U", "V"]
    ix = nodes.index
    first = engine.snap(oracle_search(FIGURE_1A, nodes), [ix("A"), ix("B")], max_depth=0)
    assert not first.complete
    assert first.marks[ix("A"), ix("B")] == first.marks[ix("B"), ix("A")] == ARROW
    alone = engine.snap(oracle_search(FIGURE_1A, nodes), [ix("A")], max_depth=0)
    assert {nodes[v] for v in alone.kept} == {"A", "C", "D", "U"}
    full = engine.snap(oracle_search(FIGURE_1A, nodes), [ix("A"), ix("B")])
    kept, cpdag = truth(matrix(FIGURE_1A, nodes), [ix("A"), ix("B")])
    assert full.complete and set(full.kept) == kept
    np.testing.assert_array_equal(full.marks, cpdag)


@pytest.mark.parametrize("targets", [("A", "B"), ("X", "B"), ("U", "F"), ("E", "H", "V")])
def test_figure_1b_recovers_the_cpdag_over_the_possible_ancestors(targets):
    """Example 3.1's graph: in every visiting order the search returns the true CPDAG over the
    possible ancestors of the targets, ``A -> B`` included."""
    nodes = sorted({v for e in FIGURE_1B for v in e})
    A = matrix(FIGURE_1B, nodes)
    found = [nodes.index(t) for t in targets]
    kept, cpdag = truth(A, found)
    rng = np.random.default_rng(0)
    for _ in range(20):
        result = engine.snap(oracle_search(FIGURE_1B, nodes, rng.permutation(len(nodes))), found)
        assert result.complete and set(result.kept) == kept
        np.testing.assert_array_equal(result.marks, cpdag)


# --- Theorems 3.1 and 3.2 on random graphs ------------------------------------------------------


def cases(count: int):
    for seed in range(count):
        rng = np.random.default_rng(seed)
        d = int(rng.integers(6, 26))
        A = random_dag(rng, d, float(rng.uniform(2, 4)))
        targets = sorted(rng.choice(d, int(rng.integers(2, min(8, d) + 1)), replace=False).tolist())
        yield A, targets, rng.permutation(d)


def test_snap_infinity_is_the_cpdag_over_the_possible_ancestors():
    """Theorem 3.2 on 60 random graphs, 6 to 25 nodes, expected degree 2 to 4, 2 to 8 targets,
    a random visiting order each."""
    for A, targets, ranks in cases(60):
        kept, cpdag = truth(A, targets)
        result = engine.snap(oracle_search(A, ranks=ranks), targets)
        assert result.complete and set(result.kept) == kept
        np.testing.assert_array_equal(result.marks, cpdag)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_each_order_keeps_the_possible_ancestors_and_the_true_edges(k):
    """Theorem 3.1 and Lemma 8.1 after order ``k``: the variables kept hold the possible ancestors
    of the targets and every possible ancestor of their own, and the graph over them every true
    edge."""
    for A, targets, ranks in cases(40):
        result = engine.snap(oracle_search(A, ranks=ranks), targets, max_depth=k)
        cpdag = dag2cpdag(dag_marks(A))
        kept = set(result.kept)
        assert adj.possible_ancestors(cpdag, targets) <= kept
        assert adj.possible_ancestors(cpdag, kept) <= kept
        inside = np.ix_(sorted(kept), sorted(kept))
        assert np.all((result.marks != 0)[inside] >= (A | A.T)[inside])


def test_tests_never_condition_on_more_than_the_cap():
    sizes: list[int] = []

    class Recorded(DSepOracle):
        def __call__(self, x, y, condition_set=()):
            sizes.append(len(condition_set))
            return super().__call__(x, y, condition_set)

    for A, targets, ranks in cases(20):
        graph = nx.DiGraph(list(zip(*np.nonzero(A), strict=True)))
        graph.add_nodes_from(range(len(A)))
        for k in (0, 1, 2):
            sizes.clear()
            engine.snap(LocalSearch(Recorded(graph), list(ranks), 0.5), targets, max_depth=k)
            assert max(sizes) <= k


# --- the records, through the facade -------------------------------------------------------------


@pytest.fixture
def dsep(monkeypatch):
    """Register ``indep_test="dsep"``, exact tests in the DAG ``dsep.graph``, for the facade."""

    class Factory:
        graph: nx.DiGraph | None = None

        def __call__(self, data, **_):
            return DSepOracle(self.graph)

    factory = Factory()
    monkeypatch.setitem(ci._CI_REGISTRY, "dsep", factory)
    return factory


def lane_oracle(A: np.ndarray, x: int, y: int) -> dict:
    """The record the benchmark lane expects of SNAP: the optimal set where the class identifies
    the effect, the local parent sets where it does not."""
    record = effect_oracle(A.astype(int), x, y, set_kind="optimal")
    if not record["identifiable"]:
        record = effect_oracle(A.astype(int), x, y, set_kind="local_parents")
    return record


def test_records_equal_the_oracle_records(dsep):
    """Every ordered pair of the targets, on 60 random graphs, the column order of each drawn from
    its data: the records equal the oracle's field by field, but for ``nonzero`` where the class
    leaves it open and the true DAG decides it. This covers the reverse direction of an edge
    between the targets: a zero effect that no adjustment set identifies."""
    reverse = 0
    for seed, (A, targets, _) in enumerate(cases(60)):
        graph = nx.DiGraph(list(zip(*np.nonzero(A), strict=True)))
        graph.add_nodes_from(range(len(A)))
        dsep.graph = graph
        data = np.random.default_rng(seed).standard_normal((30, len(A)))
        out = snap(data, variables=targets, indep_test="dsep")
        got = {(r["treatment"], r["outcome"]): r for r in out.metadata["effects"]}
        assert set(got) == set(permutations(targets, 2))
        cpdag = dag2cpdag(dag_marks(A))
        for (x, y), record in got.items():
            want = lane_oracle(A, x, y)
            # The truth may decide what the class cannot: an undecided nonzero is then right.
            if not want["identifiable"] and y in adj.possible_descendants(cpdag, [x]):
                want["nonzero"] = None
            assert record == want, (seed, x, y)
            reverse += bool(A[y, x])
        kept, _ = truth(A, targets)
        assert out.metadata["possible_ancestors"] == sorted(kept)
    assert reverse >= 5  # the reverse direction of an edge between targets is covered


def test_reverse_of_an_edge_between_the_variables(dsep):
    """``a -> y <- x`` makes ``x -> y`` compelled: the effect of ``y`` on ``x`` is zero, and no set
    adjusts for it in the one DAG of the class, since none blocks the edge."""
    A = np.zeros((3, 3), dtype=bool)
    A[0, 2] = A[1, 2] = True  # a -> y <- x, positions a = 0, x = 1, y = 2
    dsep.graph = nx.DiGraph([(0, 2), (1, 2)])
    data = np.random.default_rng(0).standard_normal((30, 3))
    out = snap(data, variables=[1, 2], indep_test="dsep")
    records = {(r["treatment"], r["outcome"]): r for r in out.metadata["effects"]}
    forward, backward = records[(1, 2)], records[(2, 1)]
    assert forward["identifiable"] and forward["nonzero"] and forward["adjustment_sets"] == [[0]]
    assert backward["nonzero"] is False and backward["identifiable"] is False
    assert backward["adjustment_sets"] == [[0]] and backward["set_kind"] == "local_parents"
    dag = GraphStructure.from_numpy(dag_marks(A), kind="dag")
    assert not any(dag.is_adjustment_set(2, 1, s) for s in ([], [0]))


# --- data ---------------------------------------------------------------------------------------


def lane_models():
    sys.path.insert(0, str(BENCHMARKS))
    try:
        return importlib.import_module("andrey_bench.adjustment_models")
    finally:
        sys.path.remove(str(BENCHMARKS))


def test_random_graph_data_beats_the_floor():
    """The paper's random graphs (20 nodes, expected degree 3, weights in [0.5, 3]) at 5,000 rows,
    the lane's target pairs, both directions scored against the oracle: whether the effect is
    nonzero and whether it is identifiable beat their floors. The exact optimal set is a harder
    answer on these data, where the empty set is valid for most pairs; the benchmark lane scores
    it, with the effect error."""
    lane = lane_models()
    pooled = {"nonzero": [], "identifiable": []}
    for seed in range(8):
        model = lane.random_graph(seed, nodes=20, degree=3.0)
        pair = model.pairs[0]
        x, y = pair.treatment, pair.outcome
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = snap(model.rows(5000, lane.DATA), variables=[x, y])
        oracle = [lane_oracle(model.dag, a, b) for a, b in ((x, y), (y, x))]
        got = effect_scores(out.metadata["effects"], oracle, true=model.dag)
        for key in pooled:
            pooled[key].append(got[key])
    for key, parts in pooled.items():
        scores = pooled_answer_scores(parts)
        assert scores["net_rate"] > scores["floor_rate"], key
