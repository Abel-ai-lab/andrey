"""Exercise BOSS and GRaSP adapters in one benchmark environment.

Tests cover the settings both packages run with, search reproducibility, and output conversion.
Conversion tests use shared DAGs to isolate encoding behavior from stochastic search results. The
fit-to-valid-CPDAG check for all four adapters is in :mod:`test_adapters_contract`.
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey import GraphStructure
from andrey.core.orient import dag2cpdag, to_structure
from andrey_bench.adapters.andrey_boss import AndreyBOSS
from andrey_bench.adapters.andrey_grasp import AndreyGRaSP
from andrey_bench.adapters.bic_penalty import ANDREY_LAMBDA, CAUSAL_LEARN_LAMBDA, SEARCH_SEED
from andrey_bench.adapters.causal_learn_boss import CausalLearnBOSS
from andrey_bench.adapters.causal_learn_convert import generalgraph_to_adjacency
from andrey_bench.adapters.causal_learn_grasp import CausalLearnGRaSP
from andrey_bench.contracts import structure_from_adjacency
from andrey_bench.scoring import structure_hash

#: Andrey endpoint marks at node ``i`` on edge ``i-j``.
_TAIL, _ARROW = 1, 2

ALL_ADAPTERS = (AndreyBOSS(), AndreyGRaSP(), CausalLearnBOSS(), CausalLearnGRaSP())


def _sample() -> np.ndarray:
    """Return a fixed small linear-Gaussian ER sample."""
    import andrey.data as data

    ds = data.benchmarks.scm("linear_gauss_er", 6).sample(n=200, seed=0)
    return np.asarray(ds.data, dtype=np.float64)


# --- Settings ------------------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda a: a.name)
def test_params_carry_the_campaign_penalty(adapter) -> None:
    """Declare equivalent penalties in each package's units rather than using defaults."""
    params = adapter.params()
    expected = ANDREY_LAMBDA if adapter.package == "andrey" else CAUSAL_LEARN_LAMBDA
    assert params["lambda_value"] == expected
    assert params["seed"] == SEARCH_SEED
    assert params["score_func"] == "local_score_BIC_from_cov"


def test_grasp_adapters_agree_on_depth() -> None:
    """Give both GRaSP adapters the same depth and omit it from BOSS."""
    assert AndreyGRaSP().params()["depth"] == CausalLearnGRaSP().params()["depth"] == 3
    assert "depth" not in AndreyBOSS().params()
    assert "depth" not in CausalLearnBOSS().params()


@pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda a: a.name)
def test_a_fixed_seed_makes_the_search_reproducible(adapter) -> None:
    """Return the same graph for the same data and seed.

    causal-learn uses the global ``random`` module and exposes no seed argument, so its adapters
    must seed the module for every fit.
    """
    data = _sample()
    first = adapter.to_structure(np.asarray(adapter.fit(data, adapter.params())))
    second = adapter.to_structure(np.asarray(adapter.fit(data, adapter.params())))
    assert structure_hash(first) == structure_hash(second)


# --- Shared output representation ----------------------------------------------------------------


def _causal_learn_cpdag_hash(adjacency: np.ndarray) -> str:
    """Build and hash a causal-learn CPDAG in the shared representation."""
    from causallearn.graph.GeneralGraph import GeneralGraph
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.DAG2CPDAG import dag2cpdag as causal_learn_dag2cpdag

    nodes = [GraphNode(f"X{i + 1}") for i in range(len(adjacency))]
    graph = GeneralGraph(nodes)
    for i, j in zip(*np.nonzero(adjacency)):
        graph.add_directed_edge(nodes[i], nodes[j])
    reduced = causal_learn_dag2cpdag(graph)
    return structure_hash(
        structure_from_adjacency(generalgraph_to_adjacency(reduced.graph), kind="cpdag")
    )


def _andrey_cpdag(adjacency: np.ndarray) -> GraphStructure:
    """Build an Andrey CPDAG from a DAG adjacency."""
    marks = np.zeros(adjacency.shape, dtype=np.uint8)
    marks[adjacency == 1] = _TAIL
    marks[(adjacency == 1).T] = _ARROW
    return to_structure(dag2cpdag(marks), kind="cpdag")


def _andrey_cpdag_hash(adjacency: np.ndarray) -> str:
    return structure_hash(_andrey_cpdag(adjacency))


def _edge_kinds(structure: GraphStructure) -> set[str]:
    """Return the directed and undirected edge classes present in a CPDAG."""
    kinds = set()
    for _i, _j, mark_i, mark_j in structure.to_edges():
        kinds.add("undirected" if mark_i == mark_j == _TAIL else "directed")
    return kinds


@pytest.mark.parametrize("seed", range(5))
def test_both_output_formats_reduce_to_the_same_cpdag(seed: int) -> None:
    """Reduce each package's encoding of one DAG to the same CPDAG hash."""
    rng = np.random.default_rng(seed)
    d = 9
    # Upper-triangular edges run low -> high, so the graph is acyclic.
    adjacency = np.triu((rng.random((d, d)) < 0.3).astype(np.int8), 1)

    assert _andrey_cpdag_hash(adjacency) == _causal_learn_cpdag_hash(adjacency)


def _mixed(d: int = 5) -> np.ndarray:
    """A collider and a disconnected edge: ``X0 -> X2 <- X1`` alongside ``X3 -> X4``.

    The collider remains directed. The disconnected edge is reversible and becomes undirected.
    Disconnection prevents Meek's rules from orienting the reversible edge.
    """
    adjacency = np.zeros((d, d), dtype=np.int8)
    adjacency[0, 2] = adjacency[1, 2] = 1
    adjacency[3, 4] = 1
    return adjacency


def _chain(d: int = 4) -> np.ndarray:
    """Return a chain whose edges are all reversible in its CPDAG."""
    adjacency = np.zeros((d, d), dtype=np.int8)
    adjacency[np.arange(d - 1), np.arange(1, d)] = 1
    return adjacency


@pytest.mark.parametrize(
    "adjacency, expected",
    [(_mixed(), {"directed", "undirected"}), (_chain(), {"undirected"})],
    ids=["mixed", "chain"],
)
def test_both_endpoint_classes_convert(adjacency: np.ndarray, expected: set[str]) -> None:
    """Exercise directed and undirected CPDAG edges explicitly.

    Directed edges set one adjacency entry; undirected edges set both. Random DAGs do not guarantee
    both edge types.
    """
    assert _edge_kinds(_andrey_cpdag(adjacency)) == expected
    assert _andrey_cpdag_hash(adjacency) == _causal_learn_cpdag_hash(adjacency)


def test_the_hash_still_separates_graphs_that_differ() -> None:
    """Negative control: a hash that collapsed every graph would pass the equality tests above."""
    d = 9
    empty = np.zeros((d, d), dtype=np.int8)
    chain = np.zeros((d, d), dtype=np.int8)
    chain[np.arange(d - 1), np.arange(1, d)] = 1

    assert _andrey_cpdag_hash(empty) != _andrey_cpdag_hash(chain)
    assert _causal_learn_cpdag_hash(empty) != _causal_learn_cpdag_hash(chain)
