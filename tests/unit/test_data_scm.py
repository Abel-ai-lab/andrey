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
