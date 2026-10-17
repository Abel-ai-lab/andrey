"""Falsification tests for ``andrey.data``.

Each test refutes one claim about the generator: reproducibility, the eager-draw trap,
component-stream isolation, truth validity, the float64 dtype, no dense allocation at 10k nodes, and
the CausalDataset contract. All are deterministic (fixed seeds) -- no
statistical thresholds here.
"""

from __future__ import annotations

import tracemalloc

import networkx as nx
import numpy as np
import pytest

import andrey.data as data
from andrey.core.structure import GraphStructure


def _scm(d=12, density=2.0, heteroscedastic=False):
    return data.SCM(
        graph=data.graphs.erdos_renyi(d=d, avg_degree=density),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(heteroscedastic=heteroscedastic),
    )


def test_shape_and_kind():
    ds = _scm(d=12).sample(n=200, seed=0)
    assert ds.data.shape == (200, 12)
    assert isinstance(ds.graph, GraphStructure)
    assert ds.graph.kind == "dag"


def test_reproducible_same_object():
    """Same SCM, same seed -> byte-identical data and equal truth."""
    scm = _scm()
    a, b = scm.sample(n=300, seed=7), scm.sample(n=300, seed=7)
    assert np.array_equal(a.data, b.data)
    assert a.graph == b.graph
    assert np.array_equal(a.params.weights, b.params.weights)


def test_config_rebuild_no_eager_draw():
    """Two independently constructed SCMs with identical specs + seed -> identical bytes.

    Refutes a hidden construction-time RNG (the eager-draw trap the same-object test cannot catch):
    if the graph were drawn at SCM construction, two separate constructions would diverge.
    """
    a = _scm(d=20).sample(n=150, seed=3)
    b = _scm(d=20).sample(n=150, seed=3)
    assert np.array_equal(a.data, b.data)
    assert a.graph == b.graph


def test_component_stream_isolation():
    """Changing only the noise axis leaves the graph and weight streams byte-identical."""
    a = _scm(d=16, heteroscedastic=False).sample(n=200, seed=1)
    b = _scm(d=16, heteroscedastic=True).sample(n=200, seed=1)
    assert a.graph == b.graph  # graph stream independent of the noise axis
    assert np.array_equal(a.params.weights, b.params.weights)  # weight stream too
    assert not np.array_equal(a.data, b.data)  # but the noise differs -> data differs


def test_seed_variation_changes_data():
    a = _scm().sample(n=200, seed=0)
    b = _scm().sample(n=200, seed=1)
    assert not np.array_equal(a.data, b.data)


@pytest.mark.parametrize("d", [10, 100, 1000])
def test_truth_is_acyclic_and_valid(d):
    """Truth is an acyclic, valid GraphStructure of kind 'dag'."""
    ds = _scm(d=d, density=2.0).sample(n=32, seed=5)
    ds.graph.validate()  # raises if malformed
    assert ds.graph.kind == "dag"
    # acyclicity on the directed parent->child edges (to_networkx's i<j view is trivially acyclic)
    g = nx.DiGraph()
    g.add_nodes_from(range(ds.graph.n_nodes))
    g.add_edges_from(map(tuple, ds.params.edges))
    assert nx.is_directed_acyclic_graph(g)


def test_labels_not_causal_order():
    """The random relabeling breaks the index==topological-order correlation (else methods can
    cheat)."""
    ds = _scm(d=200, density=3.0).sample(n=16, seed=2)
    parents, children = ds.params.edges[:, 0], ds.params.edges[:, 1]
    # If index order were topological, every parent label would be < its child label.
    assert np.any(parents > children)


def test_dtype_and_nbytes():
    """Data is float64 by default (no lossy f64->f32->f64 round-trip for f64 consumers)."""
    ds = _scm(d=12).sample(n=100, seed=0)
    assert ds.data.dtype == np.float64
    assert ds.data.nbytes == 100 * 12 * 8
    assert np.all(np.isfinite(ds.data))


def test_dtype_float32_opt_in():
    """dtype=np.float32 still works (half footprint) and equals the f64 output cast to f32."""
    scm = _scm(d=12)
    f64 = scm.sample(n=200, seed=0)
    f32 = scm.sample(n=200, seed=0, dtype=np.float32)
    assert f64.data.dtype == np.float64
    assert f32.data.dtype == np.float32
    assert f32.data.nbytes == 200 * 12 * 4
    assert np.all(np.isfinite(f32.data))
    # float64 is a strict precision superset: its f32-narrowing equals the f32 sample.
    assert np.array_equal(f64.data.astype(np.float32), f32.data)


def test_truth_is_sparse_not_dense():
    """Structural no-dense proxy: edge count is O(d), never O(d^2)."""
    d = 1000
    ds = _scm(d=d, density=2.0).sample(n=8, seed=0)
    assert ds.graph.to_edges().shape[0] < 5 * d


def test_no_dense_allocation_at_10k():
    """Generating a 10k-node dataset never materializes a dense (d, d) array.

    tracemalloc only sees the allocation if numpy registers with it; self-check first and skip
    (documented, not a silent pass) if numpy allocations are invisible here.
    """
    tracemalloc.start()
    tracemalloc.reset_peak()
    probe = np.ones((3000, 3000), dtype=np.float64)  # 72 MB
    _, probe_peak = tracemalloc.get_traced_memory()
    del probe
    numpy_tracked = probe_peak > 50 * 2**20
    tracemalloc.stop()
    if not numpy_tracked:
        pytest.skip(
            "numpy allocations not visible to tracemalloc in this build; "
            "this check needs an RSS-based probe"
        )

    tracemalloc.start()
    tracemalloc.reset_peak()
    ds = _scm(d=10_000, density=2.0).sample(n=64, seed=0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Legit peak is O(n*d) data (~5 MB) + O(edges) CSR (~KB). A dense (10k,10k) would be
    # >=100 MB (int8) or 800 MB (float64). 60 MB sits well above the legit ceiling and far
    # below any dense (d,d).
    assert peak < 60 * 2**20, f"peak {peak / 2**20:.0f} MB suggests a dense (d, d) allocation"
    assert ds.data.shape == (64, 10_000)


def test_h2_dataset_contract():
    """Every CausalDataset field is present and typed, including provenance."""
    ds = _scm(d=10).sample(n=50, seed=0)
    assert isinstance(ds.data, np.ndarray)
    assert isinstance(ds.graph, GraphStructure)
    assert isinstance(ds.report, data.QAReport)
    assert isinstance(ds.params, data.SCMParams)
    assert isinstance(ds.provenance, dict)
    for key in ("root_entropy", "spawn_key", "seed_scheme", "config", "versions"):
        assert key in ds.provenance
    assert isinstance(ds.report.varsortability, float)  # computed edge-only
    assert ds.observed_graph is ds.graph  # no latents -> identity


def test_sample_scm_convenience():
    ds = data.sample_scm(d=15, n=120, seed=4)
    assert ds.data.shape == (120, 15)
    assert ds.graph.kind == "dag"
    with pytest.raises(ValueError):
        data.sample_scm(graph="nope", d=5, n=5, seed=0)


# ---- column shuffle -----------------------------------------------------------------------------


def _assert_relabeled(plain, shuffled):
    """``shuffled`` is ``plain`` with its observed nodes in ``column_order``; latents stay last."""
    order = np.asarray(shuffled.provenance["column_order"])
    n_obs, d = plain.data.shape[1], plain.graph.n_nodes
    assert np.array_equal(np.sort(order), np.arange(n_obs))
    full = np.concatenate([order, np.arange(n_obs, d)])
    assert np.array_equal(shuffled.data, plain.data[:, order])
    assert shuffled.data.flags.c_contiguous and plain.data.flags.c_contiguous
    assert shuffled.graph.kind == plain.graph.kind
    assert np.array_equal(shuffled.graph.to_numpy(), plain.graph.to_numpy()[np.ix_(full, full)])
    types = plain.graph.node_types
    if types is None:
        assert shuffled.graph.node_types is None
    else:
        assert np.array_equal(shuffled.graph.node_types, types[full])
    assert np.array_equal(shuffled.params.edges, np.argsort(full)[plain.params.edges])
    assert np.array_equal(shuffled.params.weights, plain.params.weights, equal_nan=True)
    scales = plain.params.noise_scales  # per node, or empty for a discrete network
    assert np.array_equal(
        shuffled.params.noise_scales, scales[full] if scales.size else scales, equal_nan=True
    )
    assert plain.provenance["column_order"] == list(range(n_obs))
    assert (plain.provenance["config"]["shuffle"], shuffled.provenance["config"]["shuffle"]) == (
        False,
        True,
    )


@pytest.mark.parametrize("latents", [0, 2])
@pytest.mark.parametrize("graph", ["erdos_renyi", "scale_free", "small_world", "hub"])
def test_shuffle_relabels_an_scm_sample(graph, latents):
    """Shuffling permutes the observed columns and the truth together and changes nothing else."""
    kwargs = {"graph": graph, "d": 14, "n": 200, "seed": 3, "density": 4.0, "latents": latents}
    plain = data.sample_scm(**kwargs, shuffle=False)
    shuffled = data.sample_scm(**kwargs)
    _assert_relabeled(plain, shuffled)
    assert shuffled.provenance["column_order"] != list(range(14 - latents))
    assert shuffled.report.varsortability == pytest.approx(plain.report.varsortability)
    assert shuffled.report.r2sortability == pytest.approx(plain.report.r2sortability)


@pytest.mark.parametrize("latents", [0, 2])
@pytest.mark.parametrize("scale", ["standardize", "rescale"])
def test_shuffle_relabels_noise_scales_and_scaled_or_screened_samples(scale, latents):
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=12, avg_degree=3.0),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(heteroscedastic=True),
        latents=latents,
    )
    screen = data.FaithfulnessScreen(min_abs_corr=0.01)
    plain = scm.sample(n=300, seed=1, scale=scale, screen=screen, shuffle=False)
    shuffled = scm.sample(n=300, seed=1, scale=scale, screen=screen)
    _assert_relabeled(plain, shuffled)
    assert shuffled.report.faithfulness_accept_rate == plain.report.faithfulness_accept_rate


@pytest.mark.parametrize("graph", ["erdos_renyi", "scale_free"])
def test_shuffle_relabels_a_discrete_sample(graph):
    kwargs = {"graph": graph, "d": 10, "n": 300, "seed": 2, "categories": (2, 4)}
    plain = data.sample_discrete(**kwargs, shuffle=False)
    shuffled = data.sample_discrete(**kwargs)
    _assert_relabeled(plain, shuffled)
    cards = plain.provenance["config"]["categories"]
    order = shuffled.provenance["column_order"]
    assert shuffled.provenance["config"]["categories"] == [cards[i] for i in order]


@pytest.mark.parametrize("graph", ["erdos_renyi", "scale_free"])
def test_shuffle_relabels_a_mixed_sample(graph):
    """The categorical columns and their category counts move with the data."""
    kwargs = {"graph": graph, "d": 10, "n": 300, "seed": 2, "categories": (2, 4)}
    plain = data.sample_mixed(**kwargs, shuffle=False)
    shuffled = data.sample_mixed(**kwargs)
    _assert_relabeled(plain, shuffled)
    config = plain.provenance["config"]
    cards = dict(zip(config["categorical"], config["categories"], strict=True))
    order = shuffled.provenance["column_order"]
    moved = [j for j, i in enumerate(order) if i in cards]
    assert shuffled.provenance["config"]["categorical"] == moved
    assert shuffled.provenance["config"]["categories"] == [cards[order[j]] for j in moved]


def test_the_column_order_comes_from_its_own_stream_of_the_seed():
    """The same seed gives the same order whatever the noise; another seed gives another order."""
    order = data.sample_scm(d=20, n=10, seed=4).provenance["column_order"]
    assert data.sample_scm(d=20, n=10, seed=4).provenance["column_order"] == order
    assert data.sample_scm(d=20, n=10, seed=4, noise="laplace").provenance["column_order"] == order
    assert data.sample_scm(d=20, n=10, seed=5).provenance["column_order"] != order
    discrete = data.sample_discrete(d=20, n=10, seed=4).provenance["column_order"]
    assert data.sample_discrete(d=20, n=10, seed=4).provenance["column_order"] == discrete


def test_a_column_order_free_method_finds_the_same_graph_either_way():
    """DirectLiNGAM gives the same graph on the shuffled columns, up to the permutation."""
    import andrey

    kwargs = {"noise": "uniform", "d": 8, "n": 1000, "density": 2.0}
    for seed in range(3):
        plain = data.sample_scm(**kwargs, seed=seed, shuffle=False)
        shuffled = data.sample_scm(**kwargs, seed=seed)
        back = np.argsort(shuffled.provenance["column_order"])
        expected = andrey.direct_lingam(plain.data).structure.to_numpy()
        found = andrey.direct_lingam(shuffled.data).structure.to_numpy()
        assert np.array_equal(found[np.ix_(back, back)], expected)
