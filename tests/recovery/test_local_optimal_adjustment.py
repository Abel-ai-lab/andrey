"""Checks of ``local_optimal_adjustment``, LOAD (Schubert, Claassen, and Magliacane 2026).

Under a d-separation oracle, on random graphs and the paper's examples, both records of every pair
asked equal what the true CPDAG says: the relation of the two variables (Lemma 4.1), whether the
effect is identifiable (Corollary 4.1), the optimal adjustment set (Theorem 4.3), and the locally
valid parent sets otherwise; the edges at the two variables are the CPDAG's. On linear-Gaussian
data in the paper's setting, the records beat the base-rate answers on whether the effect is
nonzero and identifiable.
"""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pytest

import andrey
from andrey.core import adjustment as adj
from andrey.core import ci
from andrey.core.ci import register_indep_test
from andrey.data.latent import DSepOracle
from andrey.metrics import effect_oracle, effect_scores
from andrey.metrics.graph import to_cpdag


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


def _adjacency(edges, d: int) -> np.ndarray:
    dag = np.zeros((d, d), dtype=int)
    for a, b in edges:
        dag[a, b] = 1
    return dag


def _random_dag(rng: np.random.Generator, d: int, degree: float) -> np.ndarray:
    """An Erdos-Renyi DAG with expected degree ``degree`` in a random order."""
    order = rng.permutation(d)
    upper = np.triu(rng.random((d, d)) < degree / (d - 1), 1)
    return upper[np.ix_(np.argsort(order), np.argsort(order))].astype(int)


def expected(dag: np.ndarray, x: int, y: int) -> dict:
    """The record the true CPDAG gives: ``effect_oracle`` with the optimal set where the effect is
    identifiable and the locally valid parent sets where not, ``nonzero`` read in the class:
    ``True`` for an explicit ancestor, ``False`` for a definite non-ancestor, ``None`` otherwise."""
    record = effect_oracle(dag, x, y, set_kind="optimal")
    if not record["identifiable"]:
        record = effect_oracle(dag, x, y, set_kind="local_parents")
    cpdag = to_cpdag(dag).to_numpy()
    if y in adj.reach(adj.directed(cpdag), [x]):
        record["nonzero"] = True
    else:
        record["nonzero"] = None if y in adj.possible_descendants(cpdag, [x]) else False
    return record


def _oracle_fit(box: dict, dag: np.ndarray, pair, seed: int = 0):
    """``local_optimal_adjustment`` under the oracle; random columns give the ranks."""
    d = len(dag)
    box["dag"] = nx.from_numpy_array(dag, create_using=nx.DiGraph)
    x = np.random.default_rng(seed).standard_normal((40, d))
    return andrey.local_optimal_adjustment(x, variables=list(pair), indep_test="dsep")


def _check_pair(box: dict, dag: np.ndarray, pair, seed: int = 0) -> list[dict]:
    out = _oracle_fit(box, dag, pair, seed)
    records = out.metadata["effects"]
    assert [(r["treatment"], r["outcome"]) for r in records] == [tuple(pair), tuple(pair[::-1])]
    for record in records:
        assert record == expected(dag, record["treatment"], record["outcome"])
    cpdag = to_cpdag(dag).to_numpy()
    star = np.zeros_like(cpdag)
    for v in pair:
        star[v, :], star[:, v] = cpdag[v, :], cpdag[:, v]
    assert (out.structure.to_numpy() == star).all()
    return records


# Schubert et al. (2026), Example A.1: only {1, 2} separates the treatment 0 and its spouse 4.
EXAMPLE_A1 = [(0, 1), (0, 3), (1, 4), (2, 1), (2, 4), (4, 3)]
# Schubert et al. (2026), Example A.2: the CPDAG directs 2 -> 0 through 3 -> 2 <- 4 and Meek's
# rules, with 3 and 4 adjacent to the treatment 0.
EXAMPLE_A2 = [(0, 1), (2, 0), (3, 0), (4, 0), (3, 2), (4, 2), (5, 2), (5, 3), (5, 4)]


@pytest.mark.parametrize(("edges", "d"), [(EXAMPLE_A1, 5), (EXAMPLE_A2, 6)])
def test_every_pair_of_the_paper_examples_gets_the_cpdag_answer(dsep, edges, d):
    dag = _adjacency(edges, d)
    for pair in ((a, b) for a in range(d) for b in range(a + 1, d)):
        _check_pair(dsep, dag, pair)


def test_a_zero_effect_that_a_set_identifies_is_identified(dsep):
    # Schubert et al. (2026), Example A.3: V1 -> X <- V2 -> Y <- V3. Neither variable is an
    # ancestor of the other, and each effect is zero and identifiable.
    dag = _adjacency([(0, 1), (2, 1), (2, 3), (4, 3)], 5)
    for record in _check_pair(dsep, dag, (1, 3)):
        assert (record["nonzero"], record["identifiable"], record["adjustment_sets"]) == (
            False,
            True,
            [],
        )


def test_an_identifiable_effect_gets_the_optimal_set_and_its_reverse_is_zero(dsep):
    # i -> x <- c, c -> y <- p, x -> y: x causes y, {c, p} is the optimal set, and the edge
    # x -> y makes the reverse effect zero with no valid set.
    c, i, p, x, y = range(5)
    dag = _adjacency([(i, x), (c, x), (c, y), (p, y), (x, y)], 5)
    forward, reverse = _check_pair(dsep, dag, (x, y))
    assert forward["adjustment_sets"] == [[c, p]] and forward["set_kind"] == "optimal"
    assert (reverse["nonzero"], reverse["identifiable"]) == (False, False)


def test_an_undirected_neighbor_adjacent_to_the_outcome_breaks_identifiability(dsep):
    # x - v undirected, v -> y and x -> y are compelled by w -> y <- v: the path x - v -> y may be
    # causal, so the effect of x on y is not identifiable.
    x, v, y, w = range(4)
    dag = _adjacency([(x, v), (x, y), (v, y), (w, y)], 4)
    forward, _ = _check_pair(dsep, dag, (x, y))
    assert forward["identifiable"] is False and forward["set_kind"] == "local_parents"


@pytest.mark.parametrize("seed", range(40))
def test_random_graphs_get_the_cpdag_answer_for_every_kind_of_pair(dsep, seed):
    rng = np.random.default_rng(seed)
    d = int(rng.integers(6, 16))
    dag = _random_dag(rng, d, float(rng.choice([1.5, 2.0, 3.0, 4.0])))
    cpdag = to_cpdag(dag).to_numpy()
    directed = adj.directed(cpdag)
    explicit = [(a, b) for a in range(d) for b in adj.reach(directed, [a]) if a != b]
    pairs = {tuple(sorted(explicit[k])) for k in rng.permutation(len(explicit))[:3]}
    pairs |= {tuple(sorted(rng.choice(d, 2, replace=False).tolist())) for _ in range(3)}
    for pair in sorted(pairs):
        _check_pair(dsep, dag, pair, seed)


def _paper_data(seed: int, d: int = 20, n: int = 10_000):
    """The paper's synthetic setting: an Erdos-Renyi DAG of expected degree 2, weights from
    [-3, -0.5] and [0.5, 3], standard Gaussian noise, and a pair with an explicit ancestor."""
    rng = np.random.default_rng(seed)
    while True:
        dag = _random_dag(rng, d, 2.0)
        cpdag = to_cpdag(dag).to_numpy()
        directed = adj.directed(cpdag)
        explicit = [(a, b) for a in range(d) for b in adj.reach(directed, [a]) if a != b]
        if explicit:
            break
    weights = dag * rng.uniform(0.5, 3, dag.shape) * rng.choice([-1, 1], dag.shape)
    noise = rng.standard_normal((n, d))
    x = noise @ np.linalg.inv(np.eye(d) - weights)
    pair = explicit[int(rng.integers(len(explicit)))]
    return dag, x, pair


def test_paper_setting_data_beat_the_base_rates_on_relation_and_identifiability():
    nonzero, identifiable = [], []
    for seed in range(8):
        dag, x, pair = _paper_data(seed)
        out = andrey.local_optimal_adjustment(x, variables=list(pair))
        oracle = [expected(dag, *pair), expected(dag, *pair[::-1])]
        for record in oracle:
            record["nonzero"] = bool(
                effect_oracle(dag, record["treatment"], record["outcome"])["nonzero"]
            )
        scores = effect_scores(out.metadata["effects"], oracle, true=dag)
        nonzero.append(scores["nonzero"])
        identifiable.append(scores["identifiable"])
    for field in (nonzero, identifiable):
        right = sum(s["correct"] for s in field)
        queries = sum(s["queries"] for s in field)
        best_constant = max(sum(s["policy_correct"][k] for s in field) for k in ("True", "False"))
        assert right > best_constant, (right, best_constant, queries)
