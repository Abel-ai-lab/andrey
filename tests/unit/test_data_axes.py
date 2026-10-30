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
    _valid_dag(scm.sample(n=16, seed=0, scale="raw"))


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
    a, b = scm.sample(n=32, seed=11, scale="raw"), scm.sample(n=32, seed=11, scale="raw")
    assert a.graph == b.graph
    assert np.array_equal(a.data, b.data)


def test_scale_free_is_hubby():
    """A Barabasi-Albert graph has a heavy-tailed degree: the max degree far exceeds the mean."""
    scm = data.SCM(
        graph=data.graphs.scale_free(200, m=2),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    g = scm.sample(n=8, seed=0, scale="raw").graph.to_networkx().to_undirected()
    degrees = np.array([deg for _, deg in g.degree()])
    assert degrees.max() > 4 * degrees.mean()


def test_hub_has_high_degree_hubs():
    scm = data.SCM(
        graph=data.graphs.hub(60, n_hubs=3),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    g = scm.sample(n=8, seed=0, scale="raw").graph.to_networkx().to_undirected()
    degrees = np.sort([deg for _, deg in g.degree()])[::-1]
    assert degrees[0] > 10  # a genuine hub, not a uniform graph


# ---- functional + noise families -------------------------------------------------------------


@pytest.mark.parametrize(
    "functional_fn",
    [
        data.functional.additive_noise,
        data.functional.post_nonlinear,
        data.functional.nonmonotone,
        data.functional.location_scale,
    ],
)
def test_nonlinear_functionals_finite(functional_fn):
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(30, 2.0),
        functional=functional_fn(),
        noise=data.noise.gaussian(),
    )
    ds = scm.sample(n=500, seed=0, scale="raw")
    assert np.all(np.isfinite(ds.data))
    assert ds.data.shape == (500, 30)
    _valid_dag(ds)


def test_nonmonotone_child_is_uncorrelated_with_a_symmetric_parent():
    """A child of one Gaussian source is ``w cos(parent) + noise``: uncorrelated with the parent,
    exactly determined by it up to the noise."""
    ds = data.sample_scm(
        graph="erdos_renyi",
        functional="nonmonotone",
        d=12,
        n=20_000,
        seed=0,
        density=2.0,
        scale="raw",
    )
    parents = {c: [] for c in range(12)}
    for parent, child, _ in ds.graph.oriented_edges(index=True):
        parents[child].append(parent)
    pairs = [(p[0], c) for c, p in parents.items() if len(p) == 1 and not parents[p[0]]]
    assert pairs
    for parent, child in pairs:
        x, y = ds.data[:, parent], ds.data[:, child]
        weight = np.cov(np.cos(x), y)[0, 1] / np.var(np.cos(x), ddof=1)
        assert abs(np.corrcoef(x, y)[0, 1]) < 0.03
        assert np.std(y - weight * np.cos(x)) == pytest.approx(1.0, rel=0.05)


def _squared_residual_r2(ds) -> list[float]:
    """For each child, the R^2 of its squared residuals regressed on its parents.

    The residuals come from a least-squares fit of the child on an intercept, its parents, and
    their sines. Under constant noise the squared residuals do not depend on the parents and the
    R^2 is near 0.
    """
    x = ds.data.astype(np.float64)
    parents = {c: [] for c in range(x.shape[1])}
    for parent, child, _ in ds.graph.oriented_edges(index=True):
        parents[child].append(parent)
    r2 = []
    for child, pa in parents.items():
        if not pa:
            continue
        xs = x[:, pa]
        mean_basis = np.column_stack([np.ones(len(x)), xs, np.sin(xs)])
        coef = np.linalg.lstsq(mean_basis, x[:, child], rcond=None)[0]
        squared = (x[:, child] - mean_basis @ coef) ** 2
        scale_basis = np.column_stack([np.ones(len(x)), xs])
        fit = scale_basis @ np.linalg.lstsq(scale_basis, squared, rcond=None)[0]
        r2.append(1.0 - np.sum((squared - fit) ** 2) / np.sum((squared - squared.mean()) ** 2))
    return r2


@pytest.mark.parametrize("scale", ["rescale", "standardize"])
def test_location_scale_noise_variance_changes_with_the_parents(scale):
    """Every child's squared residuals depend on its parents under location-scale noise, and on
    additive noise, the same graph and weights, they do not."""
    kwargs = {"d": 10, "n": 4000, "seed": 0, "density": 2.0, "scale": scale}
    location_scale = _squared_residual_r2(data.sample_scm(functional="location_scale", **kwargs))
    additive = _squared_residual_r2(data.sample_scm(functional="additive_noise", **kwargs))
    assert len(location_scale) == len(additive) > 0
    assert min(location_scale) > 0.02
    assert max(additive) < 0.01


def test_location_scale_with_a_constant_scale_is_additive_noise():
    def sample(functional):
        scm = data.SCM(
            graph=data.graphs.erdos_renyi(15, 2.0),
            functional=functional,
            noise=data.noise.gaussian(heteroscedastic=True),
            latents=2,
        )
        return scm.sample(n=300, seed=3, scale="rescale")

    constant = sample(data.functional.location_scale(scale_range=(1.0, 1.0)))
    additive = sample(data.functional.additive_noise())
    assert constant.data.tobytes() == additive.data.tobytes()
    assert constant.graph == additive.graph
    assert np.array_equal(constant.params.weights, additive.params.weights)


def test_location_scale_is_determined_by_its_seed():
    kwargs = {"functional": "location_scale", "d": 12, "n": 200, "density": 2.0}
    first = data.sample_scm(**kwargs, seed=4)
    again = data.sample_scm(**kwargs, seed=4)
    other = data.sample_scm(**kwargs, seed=5)
    assert first.data.shape == (200, 12)
    assert first.params.weights.shape == (first.params.edges.shape[0],)
    assert first.data.tobytes() == again.data.tobytes()
    assert first.graph == again.graph
    assert not np.allclose(first.data, other.data)


@pytest.mark.parametrize(
    "scale_range", [(0.0, 1.0), (-1.0, 1.0), (2.0, 1.0), (float("nan"), 1.0), (0.2, float("inf"))]
)
def test_location_scale_rejects_a_scale_that_is_not_positive_and_bounded(scale_range):
    with pytest.raises(ValueError):
        data.functional.location_scale(scale_range=scale_range)


def test_location_scale_rejects_unitless():
    with pytest.raises(ValueError):
        data.sample_scm(functional="location_scale", d=6, n=10, seed=0, scale="unitless")


#: The first row of each functional's raw sample at ``d=5, n=20, seed=0, density=2.0``.
RECORDED_FIRST_ROWS = {
    "linear": [-1.294827865, -1.902191957, -1.754535235, -0.5607019596, -3.662228292],
    "additive_noise": [-2.728147483, -1.902191957, -3.095045662, -1.759770999, -9.021055629],
    "post_nonlinear": [-1.72502073, -1.902191957, -2.225479987, -1.341697663, -6.051786046],
    "nonmonotone": [1.095321977, -1.902191957, 0.4808494939, 1.722508962, 0.04524138155],
}


@pytest.mark.parametrize("functional", sorted(RECORDED_FIRST_ROWS))
def test_each_functional_keeps_its_recorded_draws(functional):
    """A change to a weight, noise, or column stream moves these values by far more than the
    tolerance, which only absorbs last-bit differences between CPUs."""
    ds = data.sample_scm(functional=functional, d=5, n=20, seed=0, density=2.0, scale="raw")
    np.testing.assert_allclose(ds.data[0], RECORDED_FIRST_ROWS[functional], rtol=1e-9)


@pytest.mark.parametrize("noise_name", ["uniform", "laplace", "exponential", "gumbel"])
def test_non_gaussian_noise_finite_and_centered(noise_name):
    ds = data.sample_scm(
        graph="erdos_renyi",
        functional="linear",
        noise=noise_name,
        d=20,
        n=4000,
        seed=0,
        scale="raw",
    )
    assert np.all(np.isfinite(ds.data))
    # Every noise family is normalized to ~unit variance, so column spreads stay well-scaled.
    assert 0.5 < np.median(ds.data.std(axis=0)) < 20
