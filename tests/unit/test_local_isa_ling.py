"""Contract tests for ``local_isa_ling`` and its sparse ICA."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.lingam import local
from andrey.lingam.sparse_ica import sparse_ica


def fit(X, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        return andrey.local_isa_ling(X, **kwargs)


def cyclic_four(n: int = 500) -> np.ndarray:
    """x0 -> x1, x1 -> x2 -> x3 -> x1 (a cycle), in mixed units, noise U(-1, 1)^5."""
    rng = np.random.default_rng(0)
    B = np.array([[0, 0, 0, 0], [0.7, 0, 0, -0.6], [0, 0.8, 0, 0], [0, 0, 0.6, 0]])
    E = rng.uniform(-1, 1, size=(n, 4)) ** 5
    return np.linalg.solve(np.eye(4) - B, E.T).T * np.array([1, 3, 0.5, 2])


#: The demixing matrix of ``cyclic_four()``'s standardized columns, recorded from the authors'
#: code.
AUTHORS_W = np.array(
    [
        [1.0, 5.323045107703e-07, 1.662540536491e-04, 5.631875487536e-04],
        [-6.144872093131e-01, 1.0, 5.860439136444e-05, 5.431248058279e-01],
        [-2.322816223595e-06, -7.811407057403e-01, 1.0, 2.476764897560e-07],
        [-2.467197347518e-04, -1.924505929656e-04, -6.496360586267e-01, 1.0],
    ]
)


def test_sparse_ica_reproduces_the_authors_demixing_matrix():
    found = sparse_ica(cyclic_four())
    support = np.abs(AUTHORS_W) > 0.05
    assert ((np.abs(found.W) > 0.05) == support).all()
    # Other BLAS kernels round differently, and the sign-adaptive steps carry that forward.
    np.testing.assert_allclose(found.W[support], AUTHORS_W[support], atol=1e-3)
    assert (found.init_steps, found.converged) == (5000, True)


def chain(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """x -> y -> z with noise U(-1, 1)^3."""
    e = np.random.default_rng(seed).uniform(-1, 1, (n, 3)) ** 3
    x = e[:, 0]
    y = 0.8 * x + e[:, 1]
    z = -0.7 * y + e[:, 2]
    return pd.DataFrame({"x": x, "y": y, "z": z})


@pytest.fixture(scope="module")
def chain_fit():
    return fit(chain(), target="y")


def test_the_answer_is_a_weighted_digraph_of_the_edges_at_the_target(chain_fit):
    out = chain_fit
    assert out.structure.kind == "digraph"
    assert out.structure.labels == ("x", "y", "z")
    W = out.weighted_adjacency
    assert np.flatnonzero(W).tolist() == [1, 5]  # x -> y and y -> z
    assert W[0, 1] == pytest.approx(0.8, abs=0.1) and W[1, 2] == pytest.approx(-0.7, abs=0.1)
    marks = out.structure.to_numpy()
    assert (marks[0, 1], marks[1, 0], marks[1, 2], marks[2, 1]) == (1, 2, 1, 2)


def test_metadata_names_the_sets_and_every_model(chain_fit):
    meta = chain_fit.metadata
    assert meta["target"] == "y"
    assert meta["markov_blanket"] == ["x", "z"]
    assert (meta["parents"], meta["children"]) == (["x"], ["z"])
    (model,) = meta["equivalent_models"]
    assert model["stable"] is True
    W = chain_fit.weighted_adjacency
    assert model["edges"] == [["x", "y", W[0, 1]], ["y", "z", W[1, 2]]]
    assert meta["subspaces"] == [["x"], ["y"], ["z"]]


def test_a_rerun_repeats_exactly(chain_fit):
    again = fit(chain(), target="y")
    assert np.array_equal(again.weighted_adjacency, chain_fit.weighted_adjacency)
    assert again.metadata == chain_fit.metadata


def test_reordered_columns_relabel_the_answer(chain_fit):
    out = fit(chain()[["z", "x", "y"]], target="y")
    W = out.weighted_adjacency
    assert np.flatnonzero(W).tolist() == [5, 6]  # x -> y and y -> z in the new positions
    assert W[1, 2] == pytest.approx(chain_fit.weighted_adjacency[0, 1], abs=1e-3)
    assert W[2, 0] == pytest.approx(chain_fit.weighted_adjacency[1, 2], abs=1e-3)


def test_rescaled_columns_give_the_same_graph_and_weights_in_their_units(chain_fit):
    factors = np.ldexp(1.0, [-7, 12, 3])
    out = fit(chain() * factors, target="y")
    assert out.structure.to_numpy().tolist() == chain_fit.structure.to_numpy().tolist()
    want = chain_fit.weighted_adjacency * factors[None, :] / factors[:, None]
    np.testing.assert_allclose(out.weighted_adjacency, want, rtol=1e-12)


def test_a_two_cycle_at_the_target_lists_its_partner_as_parent_and_child():
    """x -> y, y -> z -> y: z is a parent and a child of y, two directed edges."""
    rng = np.random.default_rng(1)
    E = rng.uniform(-1, 1, (3000, 3)) ** 5
    B = np.array([[0, 0, 0], [0.8, 0, -0.5], [0, 0.6, 0]])
    X = np.linalg.solve(np.eye(3) - B, E.T).T
    out = fit(X, target=1)
    assert (out.metadata["parents"], out.metadata["children"]) == ([0, 2], [2])
    marks = out.structure.to_numpy()
    assert (marks[1, 2], marks[2, 1]) == (2, 2)
    W = out.weighted_adjacency
    assert (W[2, 1], W[1, 2]) == (pytest.approx(-0.5, abs=0.1), pytest.approx(0.6, abs=0.1))
    assert [m["stable"] for m in out.metadata["equivalent_models"]] == [True, False]


def test_independent_columns_have_no_edge():
    X = np.random.default_rng(2).uniform(-1, 1, (2000, 3)) ** 3
    out = fit(X, target=0)
    assert not out.weighted_adjacency.any()
    assert out.metadata["equivalent_models"] == [{"edges": [], "stable": True}]


def test_one_column_has_an_empty_blanket():
    out = fit(np.random.default_rng(2).uniform(-1, 1, (100, 1)), target=0)
    assert out.metadata["markov_blanket"] == [] and not out.weighted_adjacency.any()
    assert out.metadata["equivalent_models"] == [{"edges": [], "stable": True}]


def test_acyclic_peels_sinks_and_falls_back_on_a_cycle():
    """On the chain the sink peeling gives the default's edges; with the target on a cycle no
    column is a sink, and the fit runs Algorithm 1."""
    dag = fit(chain(), target="y", acyclic=True)
    assert np.flatnonzero(dag.weighted_adjacency).tolist() == [1, 5]
    assert "subspaces" not in dag.metadata
    X = cyclic_four()
    cyclic = fit(X, target=1, acyclic=True)
    default = fit(X, target=1)
    assert np.array_equal(cyclic.weighted_adjacency, default.weighted_adjacency)
    assert cyclic.metadata == default.metadata


def test_algorithm3_reports_no_sink():
    assert local.algorithm3(np.array([[1.0, 0.5], [0.4, 1.0]]), 0) is None


def test_the_target_is_required_and_missing_values_raise():
    with pytest.raises(ValueError, match="pass target"):
        fit(chain())
    X = chain().to_numpy(copy=True)
    X[3, 0] = np.nan
    with pytest.raises(ValueError, match="local_isa_ling"):
        fit(X, target=1)
