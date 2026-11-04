"""Recovery checks for FCI-JCI: FCI over data pooled from several domains, with the domain index.

A d-separation oracle in place of the tests checks the logic: every mark FCI-JCI sets agrees with
the true ancestral graph over the data and the index, and it keeps every mark the oracle PAG
without the index's background knowledge resolves. Seeded linear-Gaussian data check the fixed
cases a domain shift identifies and the paper's simulation setting against the oracle's PAG, and
random pooled models check determinism, column- and row-order invariance, and well-formed output.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.constraint.fci_jci import fci_jci as fci_jci_engine
from andrey.core import ARROW, CIRCLE, NULL, TAIL, GraphStructure, ci
from andrey.core.structure import LATENT, OBSERVED
from andrey.data.latent import DSepOracle, marginal
from andrey.metrics import shd, skeleton_scores
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")

_ORACLE = "dsep_oracle"
_INDEX_MARK = {"directed": TAIL, "partially_directed": CIRCLE, "bidirected": ARROW}


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


# --- models ---------------------------------------------------------------------------------------
#
# Nodes 0..d-1 are the data variables, d is the domain index, and the hidden variables follow. The
# index is a root that shifts a third of the data variables; each hidden variable causes two data
# variables, and with ``confounded`` the first one also causes the index.


def _random_dag(seed: int, d: int, n_latent: int, *, confounded: bool = False) -> nx.DiGraph:
    rng = np.random.default_rng(seed)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(d + 1 + n_latent))
    order = rng.permutation(d)
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < 2.0 / (d - 1):
                dag.add_edge(int(order[i]), int(order[j]))
    for latent in range(d + 1, d + 1 + n_latent):
        for child in rng.choice(d, 2, replace=False):
            dag.add_edge(latent, int(child))
    for target in rng.choice(d, max(1, d // 3), replace=False):
        dag.add_edge(d, int(target))
    if confounded:
        dag.add_edge(d + 1, d)
    return dag


def _sample(dag: nx.DiGraph, d: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Linear-Gaussian data from ``dag``, with the index two equal domains: (data, index)."""
    rng = np.random.default_rng([seed, 1])
    values = np.zeros((n, dag.number_of_nodes()))
    for node in nx.topological_sort(dag):
        if node == d:
            values[:, d] = np.repeat([0.0, 1.0], n // 2)
            continue
        values[:, node] = rng.standard_normal(n)
        for parent in dag.predecessors(node):
            if parent == d:
                weight = rng.uniform(1.0, 2.0) * rng.choice([-1.0, 1.0])
                values[:, node] += weight * (values[:, d] - 0.5)
            else:
                values[:, node] += rng.uniform(0.5, 1.5) * rng.choice([-1, 1]) * values[:, parent]
    return values[:, :d], values[:, d]


def _true_graphs(dag: nx.DiGraph, d: int) -> tuple[np.ndarray, np.ndarray]:
    """The true MAG and the oracle PAG over the data variables and the index, as marks."""
    n = dag.number_of_nodes()
    M = np.zeros((n, n), dtype=np.int8)
    for u, v in dag.edges:
        M[u, v], M[v, u] = TAIL, ARROW
    types = np.array([OBSERVED] * (d + 1) + [LATENT] * (n - d - 1), dtype=np.int8)
    truth = GraphStructure.from_numpy(M, kind="dag", node_types=types)
    return marginal(truth, "mag").to_numpy(), marginal(truth, "pag").to_numpy()


def _oracle_fit(
    dag: nx.DiGraph, d: int, monkeypatch: pytest.MonkeyPatch, **params
) -> tuple[GraphStructure, list]:
    """FCI-JCI over the graph's data variables and index, with d-separation in place of tests."""
    monkeypatch.setitem(ci._CI_REGISTRY, _ORACLE, lambda _data: DSepOracle(dag))
    placeholder = np.zeros((2, d))
    return fci_jci_engine(
        placeholder, np.array([0.0, 1.0]), indep_test=_ORACLE, possible_dsep="fast", **params
    )


def _with_index(pag: GraphStructure, index_edges: list) -> np.ndarray:
    """The marks over the data variables and the index, from the PAG and the index's edges."""
    d = pag.n_nodes
    marks = np.zeros((d + 1, d + 1), dtype=np.int8)
    marks[:d, :d] = pag.to_numpy()
    for j, kind in index_edges:
        marks[j, d], marks[d, j] = ARROW, _INDEX_MARK[kind]
    return marks


# --- oracle ---------------------------------------------------------------------------------------


def test_oracle_orients_a_chain_from_its_shifted_root(monkeypatch):
    """index -> 0 -> 1 -> 2: R1 directs the chain, which FCI without the index leaves o-o."""
    dag = nx.DiGraph([(3, 0), (0, 1), (1, 2)])
    pag, index_edges = _oracle_fit(dag, 3, monkeypatch)
    assert pag.oriented_edges() == [(0, 1, "directed"), (1, 2, "directed")]
    assert index_edges == [(0, "partially_directed")]
    _, randomized = _oracle_fit(dag, 3, monkeypatch, confounded_index=False)
    assert randomized == [(0, "directed")]


def test_oracle_marks_a_hidden_common_cause_the_index_reveals(monkeypatch):
    """index -> A <- L -> B <- D with L hidden: A <-> B, which needs the index to tell apart."""
    a, b, dd, index, hidden = range(5)
    dag = nx.DiGraph([(index, a), (hidden, a), (hidden, b), (dd, b)])
    pag, index_edges = _oracle_fit(dag, 3, monkeypatch)
    assert pag.oriented_edges() == [(a, b, "bidirected"), (dd, b, "partially_directed")]
    assert index_edges == [(a, "partially_directed")]


def test_oracle_removes_the_edge_pooling_adds(monkeypatch):
    """The index shifts two variables that share no other cause: no edge between them."""
    dag = nx.DiGraph([(2, 0), (2, 1)])
    pag, index_edges = _oracle_fit(dag, 2, monkeypatch)
    assert pag.oriented_edges() == []
    assert index_edges == [(0, "partially_directed"), (1, "partially_directed")]


@pytest.mark.parametrize("confounded", [False, True])
@pytest.mark.parametrize("seed", range(16))
def test_oracle_marks_agree_with_the_true_ancestral_graph(seed, confounded, monkeypatch):
    """Each mark FCI-JCI sets is the true MAG's, and it keeps every mark the plain PAG sets.

    ``confounded_index=False`` runs only where its assumption holds: no hidden cause of the index.
    """
    d, n_latent = 5 + seed % 8, 1 + seed % 2
    dag = _random_dag(seed, d, n_latent, confounded=confounded)
    mag, plain = _true_graphs(dag, d)
    for confounded_index in (True,) if confounded else (True, False):
        found = _with_index(*_oracle_fit(dag, d, monkeypatch, confounded_index=confounded_index))
        assert np.array_equal(found != NULL, mag != NULL)
        resolved = (found != NULL) & (found != CIRCLE)
        assert np.array_equal(found[resolved], mag[resolved])
        known = (plain != NULL) & (plain != CIRCLE)
        assert np.array_equal(found[known], plain[known])


def test_the_rules_without_selection_bias_leave_an_uncovered_circle_cycle():
    """R5 makes an uncovered o-o cycle undirected; without selection bias it stays o-o."""
    from andrey.constraint.fci import _fci_orient

    def cycle() -> np.ndarray:
        marks = np.zeros((4, 4), dtype=np.int8)
        for a in range(4):
            b = (a + 1) % 4
            marks[a, b] = marks[b, a] = CIRCLE
        return marks

    with_r5 = cycle()
    _fci_orient(with_r5, lambda *_: 0.0, 0.05, {})
    assert set(with_r5[with_r5 != NULL].tolist()) == {TAIL}
    without = cycle()
    _fci_orient(without, lambda *_: 0.0, 0.05, {}, selection_bias=False)
    assert np.array_equal(without, cycle())


# --- data -----------------------------------------------------------------------------------------


def _chain(seed: int = 0, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """index -> x0 -> x1 -> x2 from two domains."""
    return _sample(nx.DiGraph([(3, 0), (0, 1), (1, 2)]), 3, n, seed)


def test_fci_jci_orients_a_chain_from_a_domain_shift():
    x, index = _chain()
    assert andrey.fci(x).structure.oriented_edges() == [(0, 1, "circle"), (1, 2, "circle")]
    out = andrey.fci_jci(x, index)
    assert out.structure.oriented_edges() == [(0, 1, "directed"), (1, 2, "directed")]
    assert out.metadata["algorithm"] == "fci_jci"
    assert out.metadata["index_edges"] == [[0, "partially_directed"]]
    randomized = andrey.fci_jci(x, index, confounded_index=False)
    assert randomized.metadata["index_edges"] == [[0, "directed"]]


def test_fci_jci_removes_the_edge_pooling_adds():
    x, index = _sample(nx.DiGraph([(2, 0), (2, 1)]), 2, 2000, seed=1)
    assert andrey.fci(x).structure.oriented_edges() != []
    out = andrey.fci_jci(x, index)
    assert out.structure.oriented_edges() == []
    assert out.metadata["index_edges"] == [[0, "partially_directed"], [1, "partially_directed"]]


def test_fci_jci_marks_a_hidden_common_cause_the_index_reveals():
    dag = nx.DiGraph([(3, 0), (4, 0), (4, 1), (2, 1)])  # index -> A <- L -> B <- D, L hidden
    x, index = _sample(dag, 3, 2000, seed=0)
    assert andrey.fci(x).structure.oriented_edges() == [
        (0, 1, "partially_directed"),
        (2, 1, "partially_directed"),
    ]
    out = andrey.fci_jci(x, index)
    assert out.structure.oriented_edges() == [(0, 1, "bidirected"), (2, 1, "partially_directed")]


def test_fci_jci_finds_no_edge_between_independent_columns():
    rng = np.random.default_rng(0)
    out = andrey.fci_jci(rng.standard_normal((1000, 4)), np.repeat([0.0, 1.0], 500))
    assert out.structure.oriented_edges() == []
    assert out.metadata["index_edges"] == []


def test_fci_jci_with_one_domain_is_fci():
    x, _ = _chain()
    out = andrey.fci_jci(x, np.ones(len(x)))
    assert np.array_equal(out.structure.to_numpy(), andrey.fci(x).structure.to_numpy())
    assert out.metadata["index_edges"] == []


def test_fci_jci_takes_the_index_as_a_category_with_a_mixed_data_test():
    """A shift in the middle of three domains coded 0, 1, 2: linear correlation misses it."""
    rng = np.random.default_rng(0)
    index = np.repeat([0.0, 1.0, 2.0], 1000)
    x0 = 1.5 * (index == 1) + rng.standard_normal(3000)
    x1 = x0 + rng.standard_normal(3000)
    x = np.column_stack([x0, x1])
    assert andrey.fci_jci(x, index).metadata["index_edges"] == []
    out = andrey.fci_jci(x, index, indep_test="cond_gauss", categorical=[])
    assert out.metadata["index_edges"] == [[0, "partially_directed"]]
    assert out.structure.oriented_edges() == [(0, 1, "directed")]


def _paper_dag(seed: int, d: int = 10, p: float = 0.25) -> nx.DiGraph:
    """The paper's random model with one context variable: each ordered pair of data variables is
    an edge with probability ``p``, each unordered pair shares a hidden common cause with
    probability ``p``, and the index targets one data variable."""
    rng = np.random.default_rng(seed)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(d + 1))
    order = rng.permutation(d)
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < p:
                dag.add_edge(int(order[i]), int(order[j]))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < p:
                latent = dag.number_of_nodes()
                dag.add_edges_from([(latent, i), (latent, j)])
    dag.add_edge(d, int(rng.integers(d)))
    return dag


def _paper_sample(dag: nx.DiGraph, d: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` rows per context: weights in +-[0.5, 1.5], each equation scaled to variance 1 under
    independent standard-normal causes, and the target shifted by 1 in the second context."""
    rng = np.random.default_rng([seed, 2])
    values = np.zeros((2 * n, dag.number_of_nodes()))
    values[:, d] = np.repeat([0.0, 1.0], n)
    for node in nx.topological_sort(dag):
        if node > d:
            values[:, node] = rng.standard_normal(2 * n)
        elif node < d:
            parents = [q for q in dag.predecessors(node) if q != d]
            w = rng.uniform(0.5, 1.5, len(parents)) * rng.choice([-1.0, 1.0], len(parents))
            noise = rng.standard_normal(2 * n)
            values[:, node] = (noise + values[:, parents] @ w) / np.sqrt(1.0 + w @ w)
            values[:, node] += values[:, d] * dag.has_edge(d, node)
    return values[:, :d], values[:, d]


@pytest.mark.parametrize("confounded_index", [True, False])
@pytest.mark.parametrize("seed", range(3))
def test_fci_jci_recovers_the_pag_in_the_papers_setting(seed, confounded_index):
    """10 variables, edge and hidden-cause probabilities 0.25, two contexts of 500 rows, alpha 0.01:
    the marks over the data and the index, against the oracle's, keep nearly every adjacency true
    and stay below 0.7 times the empty graph's errors."""
    d = 10
    dag = _paper_dag(seed)
    x, index = _paper_sample(dag, d, 500, seed)
    pag, index_edges = _oracle_fit(dag, d, pytest.MonkeyPatch(), confounded_index=confounded_index)
    truth = GraphStructure.from_numpy(_with_index(pag, index_edges), kind="pag")
    out = andrey.fci_jci(x, index, alpha=0.01, confounded_index=confounded_index)
    marks = _with_index(out.structure, out.metadata["index_edges"])
    found = GraphStructure.from_numpy(marks, kind="pag")
    empty = GraphStructure.from_numpy(np.zeros((d + 1, d + 1), dtype=np.int8), kind="pag")
    assert skeleton_scores(found, truth)["precision"] >= 0.9
    assert shd(found, truth, endpoint_aware=True) < 0.7 * shd(empty, truth, endpoint_aware=True)


def test_fci_jci_matches_the_recorded_output():
    """``gauss_5v`` with its root shifted in the second domain."""
    x, index = datagen.domain_shift("gauss_5v", (0,))
    assert_recorded_output(andrey.fci_jci(x, index), "FCIJCI", case="gauss_5v")


# --- contract -------------------------------------------------------------------------------------


def _pooled(seed: int, d: int = 8) -> tuple[np.ndarray, np.ndarray]:
    return _sample(_random_dag(seed, d, 1), d, 1200, seed)


@pytest.mark.parametrize("seed", range(3))
def test_fci_jci_is_deterministic_and_well_formed(seed):
    x, index = _pooled(seed)
    first = andrey.fci_jci(x, index)
    assert first.structure.kind == "pag"
    assert first.structure.n_nodes == x.shape[1]
    first.structure.validate()
    again = andrey.fci_jci(x, index)
    assert np.array_equal(again.structure.to_numpy(), first.structure.to_numpy())
    assert again.metadata == first.metadata


@pytest.mark.parametrize("seed", range(3))
def test_fci_jci_is_invariant_to_column_order(seed):
    x, index = _pooled(seed)
    order = np.random.default_rng(seed).permutation(x.shape[1])
    back = np.argsort(order)
    first = andrey.fci_jci(x, index)
    permuted = andrey.fci_jci(x[:, order], index)
    marks = permuted.structure.to_numpy()[np.ix_(back, back)]
    assert np.array_equal(marks, first.structure.to_numpy())
    relabeled = sorted([int(order[j]), kind] for j, kind in permuted.metadata["index_edges"])
    assert relabeled == first.metadata["index_edges"]


def test_reordering_the_rows_leaves_fci_jci_unchanged():
    x, index = _pooled(0)
    labels = np.where(index == 0, "b", "a")
    first = andrey.fci_jci(x, labels)
    order = np.random.default_rng(0).permutation(len(index))
    again = andrey.fci_jci(x[order], labels[order])
    assert np.array_equal(again.structure.to_numpy(), first.structure.to_numpy())
    assert again.metadata == first.metadata


def test_fci_jci_names_the_index_edges_by_column():
    x, index = _chain()
    out = andrey.fci_jci(pd.DataFrame(x, columns=["a", "b", "c"]), index)
    assert out.structure.labels == ("a", "b", "c")
    assert out.metadata["index_edges"] == [["a", "partially_directed"]]


@pytest.mark.parametrize(
    ("index", "message"),
    [
        (np.zeros((2000, 2)), "one value per row"),
        (np.zeros(1999), "one value per row"),
        (np.r_[np.zeros(1999), np.nan], "NaN"),
        (np.r_[np.zeros(1999), np.inf], "NaN or inf"),
    ],
)
def test_fci_jci_rejects_a_bad_index(index, message):
    x, _ = _chain()
    with pytest.raises(ValueError, match=message):
        andrey.fci_jci(x, index)


def test_fci_jci_checks_the_index_shape_before_its_categories():
    x = np.random.default_rng(0).integers(0, 2, (200, 2)).astype(float)
    for index in (0.5, np.full((200, 2), 0.5)):
        with pytest.raises(ValueError, match="one value per row"):
            andrey.fci_jci(x, index, indep_test="chisq")


def test_fci_jci_rejects_a_fractional_index_with_a_categorical_test():
    x, index = _chain()
    with pytest.raises(ValueError, match="env"):
        andrey.fci_jci(x, index + 0.5, indep_test="cond_gauss", categorical=[])


def test_fci_jci_rejects_a_confounded_index_that_is_not_a_bool():
    x, index = _chain()
    with pytest.raises(TypeError, match="confounded_index"):
        andrey.fci_jci(x, index, confounded_index=1)
