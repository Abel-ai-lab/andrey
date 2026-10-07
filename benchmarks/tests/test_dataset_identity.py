"""One file per dataset — and never one file for two.

Were ``density`` left out of ``DatasetKey``, a dense key and a sparse key at otherwise identical
coordinates would compute the same digest and write the same ``ds_<digest>.npz``: the second run
would read the first run's data. Same-key-same-bytes is true by construction (materialize once,
read many), so the direction under test here is the one that can fail — *different* keys
colliding.

The digest is also the ``dataset_id``, so a collision here does not merely share a file: it gives two
tasks one identity in every record that was ever measured.

The assertions run at three levels, because a fix at one of them is not a fix: the digest and the
filename (which dataset a key resolves to), the label (which :func:`materialize` dedupes on, so a
density-blind label merges the two keys before the digest is ever consulted), and the materialized
files themselves.
"""

from __future__ import annotations

import pytest
from tiny_keys import TINY_KEYS

from andrey_bench import datasets

_BASE_KEY = {
    "topology": "er",
    "functional": "linear",
    "noise": "gaussian",
    "density": 2.0,
    "d": 10,
    "n": 100,
    "seed": 0,
    "standardize": "standardized",
}


def _key(**overrides) -> datasets.DatasetKey:
    return datasets.DatasetKey(**{**_BASE_KEY, **overrides})


def test_every_generation_coordinate_moves_the_digest_and_the_filename():
    """Vary one coordinate at a time; no two variants may resolve to the same dataset."""
    variants = {
        "base": _key(),
        "topology": _key(topology="hub"),
        "functional": _key(functional="anm"),
        "noise": _key(noise="laplace"),
        "density": _key(density=6.0),
        "d": _key(d=11),
        "n": _key(n=101),
        "seed": _key(seed=1),
        "standardize": _key(standardize="raw"),
        "latents": _key(latents=1),
    }
    digests = {name: key.digest for name, key in variants.items()}
    assert len(set(digests.values())) == len(variants), digests
    assert len({key.filename for key in variants.values()}) == len(variants)
    # The label is the registry map key AND materialize's dedupe key, so it has to separate them too.
    assert len({key.label for key in variants.values()}) == len(variants)


def test_dense_and_sparse_erdos_renyi_are_two_datasets():
    sparse, dense = _key(density=2.0), _key(density=6.0)
    assert sparse.digest != dense.digest
    assert sparse.filename != dense.filename
    assert sparse.label != dense.label
    # And the density each one carries is the density its generator call gets.
    assert datasets.generation_kwargs(sparse)["density"] == 2.0
    assert datasets.generation_kwargs(dense)["density"] == 6.0


@pytest.mark.parametrize("topology", ["scale_free", "small_world", "hub"])
def test_density_travels_on_every_topology(topology):
    """``density`` is the one density axis: ``sample_scm`` maps it onto whatever knob the builder
    exposes, so two densities are two datasets on every topology, not only Erdos-Renyi."""
    sparse, dense = _key(topology=topology, density=2.0), _key(topology=topology, density=6.0)
    assert datasets.generation_kwargs(sparse)["density"] == 2.0
    assert datasets.generation_kwargs(dense)["density"] == 6.0
    assert sparse.digest != dense.digest
    assert sparse.filename != dense.filename
    assert sparse.label != dense.label


def test_two_densities_materialize_two_files(tmp_path):
    """Through :func:`materialize`, which dedupes on the label — the step a digest-only fix misses."""
    sparse, dense = _key(density=2.0), _key(density=6.0)
    registry = datasets.materialize([sparse, dense], tmp_path)

    assert registry["n_datasets"] == 2
    assert len(list(tmp_path.glob("ds_*.npz"))) == 2

    # The dense file really is the dense graph, and the registry says which density made it.
    _, sparse_marks = datasets.load(tmp_path, sparse)
    _, dense_marks = datasets.load(tmp_path, dense)
    assert (dense_marks != 0).sum() > (sparse_marks != 0).sum()
    calls = {entry["generation_call"]["density"] for entry in registry["keys"].values()}
    assert calls == {2.0, 6.0}


def test_the_tiny_keys_declare_the_canonical_density():
    for key in TINY_KEYS:
        assert key.density == datasets.CANONICAL_DENSITY
        assert datasets.generation_kwargs(key)["density"] == 2.0
    # Four distinct coordinates, so the micro-grid exercises four tasks and not one four times.
    assert len({key.digest for key in TINY_KEYS}) == len(TINY_KEYS) == 4


def test_no_hidden_nodes_leaves_the_generator_call_and_label_as_they_were():
    """Check that `latents=0` preserves dataset identity in existing stores and records."""
    assert "latents" not in datasets.generation_kwargs(_key())
    assert _key().digest == "314b39c589d4"  # Digest is unchanged by `latents=0`.
    assert datasets.generation_kwargs(_key(latents=2))["latents"] == 2
    assert _key().label.endswith("-standardized")


def test_a_latent_key_drops_the_hidden_columns_and_keeps_them_in_the_truth(tmp_path):
    key = _key(latents=1)
    registry = datasets.materialize([key], tmp_path)
    data, marks = datasets.load(tmp_path, key)
    assert data.shape == (key.n, key.d - 1)
    assert marks.shape == (key.d, key.d)
    assert registry["keys"][key.label]["latents"] == 1


def test_the_latent_regime_hides_one_node_in_ten():
    assert datasets.latents_for("latent_gauss_er", 800) == 80
    assert datasets.latents_for("latent_gauss_er", 5) == 1
    assert datasets.latents_for("linear_gauss_er", 800) == 0
    with pytest.raises(KeyError, match="unknown regime"):
        datasets.latents_for("latent_gauss_ER", 800)  # Reject a typo instead of returning 0.


def test_one_key_passed_twice_is_materialized_once(tmp_path):
    """A benchmark that runs one dataset at two dtypes hands ``materialize`` the same key twice."""
    key = _key()
    registry = datasets.materialize([key, key], tmp_path)
    assert registry["n_datasets"] == 1
    assert len(list(tmp_path.glob("ds_*.npz"))) == 1
