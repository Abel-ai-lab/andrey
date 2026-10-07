"""Storage round-trip tests: native safetensors + JSON manifest.

Refutations: a dataset round-trips losslessly (data, truth including node_types,
report, params); node_types None encodes as tensor absence (not zeros) so Structure.__eq__ holds; an
edgeless graph round-trips; a collection saves + reloads and its manifest is filterable without
opening a buffer.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import andrey.data as data
from andrey.data import storage

safetensors = pytest.importorskip("safetensors")


def _assert_roundtrip(a, b):
    assert np.array_equal(a.data, b.data)
    assert a.data.dtype == b.data.dtype
    assert a.graph == b.graph  # includes node_types + kind + labels
    assert a.report == b.report  # frozen dataclass; configs chosen so no nan fields
    assert np.array_equal(a.params.edges, b.params.edges)
    assert np.array_equal(a.params.weights, b.params.weights)
    assert a.params.functional == b.params.functional
    assert a.provenance == b.provenance


def test_roundtrip_no_latents(tmp_path):
    ds = data.sample_scm(d=20, n=500, seed=0, density=3.0)  # n>d -> report has no nan
    assert ds.graph.node_types is None
    storage.save(ds, tmp_path / "d.safetensors")
    _assert_roundtrip(ds, storage.load(tmp_path / "d.safetensors"))


def test_roundtrip_with_latents(tmp_path):
    ds = data.sample_scm(d=30, n=500, seed=1, density=3.0, latents=4)
    assert ds.graph.node_types is not None
    storage.save(ds, tmp_path / "d.safetensors")
    loaded = storage.load(tmp_path / "d.safetensors")
    _assert_roundtrip(ds, loaded)
    assert loaded.graph.node_types is not None  # not silently dropped to None/zeros


def test_edgeless_graph_roundtrips(tmp_path):
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(3, 0.0),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    ds = scm.sample(n=50, seed=0)
    assert ds.params.edges.shape[0] == 0
    storage.save(ds, tmp_path / "e.safetensors")
    loaded = storage.load(tmp_path / "e.safetensors")
    assert loaded.graph == ds.graph
    assert loaded.params.edges.shape == (0, 2)


def test_collection_roundtrip_and_manifest(tmp_path):
    scms = [
        data.SCM(
            graph=data.graphs.erdos_renyi(15, 2.0),
            functional=data.functional.linear(),
            noise=data.noise.gaussian(),
        ),
        data.SCM(
            graph=data.graphs.scale_free(15, 2),
            functional=data.functional.linear(),
            noise=data.noise.uniform(),
        ),
    ]
    ens = data.Ensemble(scms=scms, seeds=range(3), n=300)
    datasets = data.generate(ens, workers=1)
    storage.save_collection(datasets, tmp_path / "coll")
    loaded = storage.load_collection(tmp_path / "coll")
    assert len(loaded) == len(datasets) == 6
    for a, b in zip(datasets, loaded):
        _assert_roundtrip(a, b)
    manifest = json.loads((tmp_path / "coll" / "collection.json").read_text())
    assert len(manifest["datasets"]) == 6
    assert all("varsortability" in e and "nnz" in e for e in manifest["datasets"])


def test_header_metadata_read_without_data(tmp_path):
    """Provenance is readable from the header alone (mmap), without materializing X."""
    ds = data.sample_scm(d=20, n=2000, seed=0)
    storage.save(ds, tmp_path / "d.safetensors")
    from safetensors import safe_open

    with safe_open(str(tmp_path / "d.safetensors"), framework="numpy") as f:
        meta = f.metadata()
    assert meta["kind"] == "dag"
    assert json.loads(meta["provenance"])["config"]["graph"]["num_nodes"] == 20


def test_zero_sample_dataset_roundtrips(tmp_path):
    """A dataset with n=0 (X dropped as zero-size) rebuilds X on load (regression: KeyError)."""
    ds = data.sample_scm(d=6, n=0, seed=0)
    storage.save(ds, tmp_path / "z.safetensors")
    loaded = storage.load(tmp_path / "z.safetensors")
    assert loaded.data.shape == (0, 6)
    assert loaded.graph == ds.graph
