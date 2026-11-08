"""Models pooled from two domains, for the Joint Causal Inference checks.

Nodes ``0..d-1`` of a model are the data variables, ``d`` is the domain index, and the hidden
variables follow. :func:`random_dag` draws a model, :func:`sample_pooled` its linear-Gaussian data,
and :func:`oracle_fit` runs FCI-JCI over its data variables and index with d-separation in place of
the tests.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from andrey import GraphStructure
from andrey.constraint.fci_jci import fci_jci
from andrey.core import ci
from andrey.data.latent import DSepOracle

#: The name the d-separation oracle is registered under for :func:`oracle_fit`.
ORACLE = "dsep_oracle"


def random_dag(seed: int, d: int, n_latent: int, *, confounded: bool = False) -> nx.DiGraph:
    """A random model over ``d`` data variables, the index, and ``n_latent`` hidden variables.

    Each ordered pair of data variables is an edge with probability ``2 / (d - 1)``, each hidden
    variable causes two data variables, and the index is a root that shifts a third of the data
    variables; with ``confounded`` the first hidden variable also causes the index.
    """
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


def sample_pooled(dag: nx.DiGraph, d: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
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


def oracle_fit(
    dag: nx.DiGraph, d: int, monkeypatch: pytest.MonkeyPatch, **params
) -> tuple[GraphStructure, list]:
    """FCI-JCI over the graph's data variables and index, with d-separation in place of tests."""
    monkeypatch.setitem(ci._CI_REGISTRY, ORACLE, lambda _data: DSepOracle(dag))
    placeholder = np.zeros((2, d))
    return fci_jci(
        placeholder, np.array([0.0, 1.0]), indep_test=ORACLE, possible_dsep="fast", **params
    )
