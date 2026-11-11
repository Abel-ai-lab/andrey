"""The fixed-network sampler with hidden nodes: its model, its rows, its targets, and its truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from andrey_bench import latent_networks as L
from andrey_bench.networks import Network, parse_network

ASIA = Path(__file__).resolve().parents[2] / "tests" / "unit" / "fixtures" / "asia.bif.gz"
_TAIL, _ARROW, _CIRCLE = 1, 2, 3


@pytest.fixture(scope="module")
def asia() -> Network:
    import gzip

    return parse_network("asia", gzip.decompress(ASIA.read_bytes()).decode())


def test_the_targets_are_the_most_connected_nodes_and_depend_on_the_network_alone(asia):
    assert L.default_targets(asia, 2) == (
        "either",
        "tub",
    )  # degree 4, then the first of degree 2 in file order
    drawn = [L.sample(asia, n=50, seed=s, hidden=2) for s in range(5)]
    assert {tuple(d.columns[t] for t in d.targets) for d in drawn} == {("either", "tub")}


def test_a_target_is_never_hidden_and_the_hidden_nodes_come_last_in_the_truth(asia):
    for seed in range(20):
        d = L.sample(asia, n=20, seed=seed, hidden=3)
        assert set(d.hidden_names).isdisjoint(d.columns) and len(d.hidden_names) == 3
        assert {"either", "tub"} <= set(d.columns)
        assert d.data.shape == (20, 5) and d.dag.shape == (8, 8) and d.d == 8
        # Node i of the truth is column i for the observed ones, then the hidden ones.
        names = d.columns + d.hidden_names
        arcs = {(names[i], names[j]) for i, j in zip(*np.nonzero(d.weights))}
        assert arcs == set(asia.arcs)


def test_the_weights_are_signed_uniform_magnitudes_on_the_range_only_on_arcs(asia):
    d = L.sample(asia, n=10, seed=0, hidden=0, weight_range=(0.5, 1.0))
    w = d.weights[d.weights != 0]
    assert len(w) == 8 and np.abs(w).min() >= 0.5 and np.abs(w).max() <= 1.0
    signs = set()
    for seed in range(30):
        w = L.sample(asia, n=10, seed=seed, hidden=0).weights
        signs |= set(np.sign(w[w != 0]))
    assert signs == {-1.0, 1.0}
    wide = L.sample(asia, n=10, seed=0, hidden=0, weight_range=(2.0, 3.0))
    assert np.abs(wide.weights[wide.weights != 0]).min() >= 2.0


def test_the_dag_marks_follow_the_weights(asia):
    d = L.sample(asia, n=10, seed=1, hidden=2)
    for i, j in zip(*np.nonzero(d.weights)):
        assert (d.dag[i, j], d.dag[j, i]) == (_TAIL, _ARROW)
    assert np.count_nonzero(d.dag == _TAIL) == 8


def test_the_model_is_a_function_of_the_network_and_seed_and_the_rows_of_n_too(asia):
    a, b = L.sample(asia, n=100, seed=7, hidden=2), L.sample(asia, n=100, seed=7, hidden=2)
    np.testing.assert_array_equal(a.data, b.data)
    longer = L.sample(asia, n=300, seed=7, hidden=2)
    for field in ("dag", "weights"):
        np.testing.assert_array_equal(getattr(a, field), getattr(longer, field))
    assert (a.columns, a.hidden_names, a.targets) == (
        longer.columns,
        longer.hidden_names,
        longer.targets,
    )
    assert not np.array_equal(a.data, longer.data[:100])  # fresh rows for another n
    other = L.sample(asia, n=100, seed=8, hidden=2)
    assert not np.array_equal(a.weights, other.weights)


def test_the_columns_are_shuffled_by_the_seed_unless_asked_not_to(asia):
    orders = {L.sample(asia, n=5, seed=s, hidden=0).columns for s in range(10)}
    assert len(orders) > 1
    plain = L.sample(asia, n=5, seed=0, hidden=0, shuffle=False)
    assert plain.columns == asia.nodes


def test_rows_have_the_covariance_the_weights_imply(asia):
    d = L.sample(asia, n=200_000, seed=2, hidden=2)
    full = np.linalg.inv(np.eye(d.d) - d.weights)
    covariance = full.T @ full
    m = d.d - d.hidden
    np.testing.assert_allclose(
        np.cov(d.data, rowvar=False), covariance[:m, :m], rtol=0.05, atol=0.05
    )
    assert d.max_variance == pytest.approx(np.diag(covariance).max(), rel=0.05)


def test_asking_to_hide_more_than_the_non_targets_hold_is_an_error(asia):
    L.sample(asia, n=5, seed=0, hidden=6)
    with pytest.raises(ValueError, match="cannot hide 7 of the 6"):
        L.sample(asia, n=5, seed=0, hidden=7)
    with pytest.raises(ValueError, match="no nodes"):
        L.sample(asia, n=5, seed=0, hidden=1, targets=("nonesuch",))


def test_named_targets_replace_the_default(asia):
    d = L.sample(asia, n=5, seed=0, hidden=1, targets=("smoke",))
    assert [d.columns[t] for t in d.targets] == ["smoke"]


def test_hiding_a_common_cause_leaves_its_children_adjacent_in_the_truth():
    net = Network("fork", ("h", "x", "y", "z"), (("h", "x"), ("h", "y"), ("x", "z")))
    d = L.sample(net, n=10, seed=0, hidden=1, targets=("x", "y", "z"), shuffle=False)
    assert d.hidden_names == ("h",) and d.columns == ("x", "y", "z")
    pag = L.truth(d)
    assert pag.shape == (3, 3)
    assert pag[0, 1] != 0 and pag[1, 0] != 0  # x and y, joined by the hidden common cause
    assert pag[0, 2] != 0 and pag[1, 2] == pag[2, 1] == 0  # x - z; y and z are separated by x


def test_hiding_a_leaf_leaves_the_pag_of_the_rest():
    arcs = (("a", "c"), ("b", "c"), ("c", "d"), ("a", "e"))
    net = Network("leaf", ("a", "b", "c", "d", "e"), arcs)
    d = L.sample(net, n=10, seed=0, hidden=1, targets=("a", "b", "c", "d"), shuffle=False)
    assert d.hidden_names == ("e",)
    pag = L.truth(d)
    assert (pag[0, 2], pag[2, 0]) == (_CIRCLE, _ARROW)  # a o-> c <-o b
    assert (pag[1, 2], pag[2, 1]) == (_CIRCLE, _ARROW)
    assert (pag[2, 3], pag[3, 2]) == (_TAIL, _ARROW)  # c -> d


def test_without_hidden_nodes_the_truth_is_the_dags_cpdag():
    net = Network("collider", ("a", "b", "c", "d"), (("a", "c"), ("b", "c"), ("c", "d")))
    d = L.sample(net, n=10, seed=0, hidden=0, shuffle=False)
    cpdag = L.truth(d)
    assert cpdag.shape == (4, 4)
    for parent in (0, 1):  # the v-structure and the edge it forces are directed
        assert (cpdag[parent, 2], cpdag[2, parent]) == (_TAIL, _ARROW)
    assert (cpdag[2, 3], cpdag[3, 2]) == (_TAIL, _ARROW)
    undirected = L.sample(
        Network("chain", ("a", "b", "c"), (("a", "b"), ("b", "c"))),
        n=10,
        seed=0,
        hidden=0,
        shuffle=False,
    )
    chain = L.truth(undirected)
    assert (chain[0, 1], chain[1, 0]) == (_TAIL, _TAIL)
