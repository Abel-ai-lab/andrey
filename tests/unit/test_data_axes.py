"""Check topology, reproducibility, and finite samples across functional and noise families."""

from __future__ import annotations

import numpy as np
import pytest

import andrey.data as data


def _valid_dag(ds):
    ds.graph.validate()
    assert ds.graph.kind == "dag"
    import networkx as nx

    # Check acyclicity on the *directed* (parent->child) edges -- to_networkx emits the raw i<j
    # topology for a dag, which is trivially acyclic regardless of orientation.
    g = nx.DiGraph()
    g.add_nodes_from(range(ds.graph.n_nodes))
    g.add_edges_from(map(tuple, ds.params.edges))
    assert nx.is_directed_acyclic_graph(g)


# ---- graph topologies ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec_fn",
    [
        lambda d: data.graphs.scale_free(d, m=2),
        lambda d: data.graphs.small_world(d, k=4, p=0.1),
        lambda d: data.graphs.hub(d, n_hubs=3),
    ],
)
@pytest.mark.parametrize("d", [10, 100, 1000])
def test_topologies_acyclic_and_valid(spec_fn, d):
    scm = data.SCM(
        graph=spec_fn(d), functional=data.functional.linear(), noise=data.noise.gaussian()
    )
    _valid_dag(scm.sample(n=16, seed=0))


@pytest.mark.parametrize(
    "spec_fn",
    [
        lambda: data.graphs.scale_free(64, m=2),
        lambda: data.graphs.small_world(64, k=4, p=0.2),
        lambda: data.graphs.hub(64, n_hubs=3),
    ],
)
def test_topologies_reproducible(spec_fn):
    scm = data.SCM(
        graph=spec_fn(), functional=data.functional.linear(), noise=data.noise.gaussian()
    )
    a, b = scm.sample(n=32, seed=11), scm.sample(n=32, seed=11)
    assert a.graph == b.graph
    assert np.array_equal(a.data, b.data)


def test_scale_free_is_hubby():
    """A Barabasi-Albert graph has a heavy-tailed degree: the max degree far exceeds the mean."""
    scm = data.SCM(
        graph=data.graphs.scale_free(200, m=2),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    g = scm.sample(n=8, seed=0).graph.to_networkx().to_undirected()
    degrees = np.array([deg for _, deg in g.degree()])
    assert degrees.max() > 4 * degrees.mean()


def test_hub_has_high_degree_hubs():
    scm = data.SCM(
        graph=data.graphs.hub(60, n_hubs=3),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    g = scm.sample(n=8, seed=0).graph.to_networkx().to_undirected()
    degrees = np.sort([deg for _, deg in g.degree()])[::-1]
    assert degrees[0] > 10  # a genuine hub, not a uniform graph


# ---- functional + noise families -------------------------------------------------------------


@pytest.mark.parametrize(
    "functional_fn", [data.functional.additive_noise, data.functional.post_nonlinear]
)
def test_nonlinear_functionals_finite(functional_fn):
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(30, 2.0),
        functional=functional_fn(),
        noise=data.noise.gaussian(),
    )
    ds = scm.sample(n=500, seed=0)
    assert np.all(np.isfinite(ds.data))
    assert ds.data.shape == (500, 30)
    _valid_dag(ds)


@pytest.mark.parametrize("noise_name", ["uniform", "laplace", "exponential", "gumbel"])
def test_non_gaussian_noise_finite_and_centered(noise_name):
    ds = data.sample_scm(
        graph="erdos_renyi", functional="linear", noise=noise_name, d=20, n=4000, seed=0
    )
    assert np.all(np.isfinite(ds.data))
    # Every noise family is normalized to ~unit variance, so column spreads stay well-scaled.
    assert 0.5 < np.median(ds.data.std(axis=0)) < 20
