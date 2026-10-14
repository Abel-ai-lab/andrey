"""``andrey.data.discrete``: BIF tables, ancestral sampling, and random discrete networks."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

import andrey.data as data
from andrey.data.discrete import DiscreteBN, DiscreteNetwork, read_bif_network

_ASIA = Path(__file__).parent / "fixtures" / "asia.bif.gz"


def test_bif_tables_are_read_with_the_last_parent_fastest():
    network = read_bif_network(gzip.decompress(_ASIA.read_bytes()).decode("utf-8"))
    j = network.nodes.index("dysp")
    assert [network.nodes[p] for p in network.parents[j]] == ["bronc", "either"]
    # Rows (bronc, either) = (yes, yes), (yes, no), (no, yes), (no, no) in the file's state order.
    np.testing.assert_allclose(network.tables[j][:, 0], [0.9, 0.8, 0.7, 0.1])
    assert len(network.arcs) == 8 and network.states[j] == ("yes", "no")


def test_bif_without_a_table_is_refused():
    text = "variable a {\n  type discrete [ 2 ] { x, y };\n}\n"
    with pytest.raises(ValueError, match="no probability block"):
        read_bif_network(text)


def test_sampling_follows_the_tables():
    # b copies a with probability 0.9.
    network = DiscreteNetwork(
        nodes=("a", "b"),
        states=(("0", "1"), ("0", "1")),
        parents=((), (0,)),
        tables=(np.array([[0.5, 0.5]]), np.array([[0.9, 0.1], [0.1, 0.9]])),
    )
    codes = network.sample(20_000, np.random.default_rng(0))
    assert codes.dtype == np.int64 and set(np.unique(codes)) == {0, 1}
    assert 0.88 < (codes[:, 0] == codes[:, 1]).mean() < 0.92


def test_a_table_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="shape"):
        DiscreteNetwork(
            nodes=("a", "b"),
            states=(("0", "1"), ("0", "1")),
            parents=((), (0,)),
            tables=(np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]])),
        )


def test_random_network_is_reproducible_and_matches_its_graph():
    bn = DiscreteBN(graph=data.graphs.erdos_renyi(d=7, avg_degree=2), levels=(2, 3))
    first, again = bn.sample(n=300, seed=4), bn.sample(n=300, seed=4)
    np.testing.assert_array_equal(first.data, again.data)
    assert not np.array_equal(first.data, bn.sample(n=300, seed=5).data)
    network = bn.network(4)
    arcs = {(network.nodes.index(a), network.nodes.index(b)) for a, b in network.arcs}
    assert arcs == {tuple(e) for e in first.params.edges.tolist()}
    assert all(2 <= len(s) <= 3 for s in network.states)
    assert first.graph.kind == "dag" and first.params.functional == "categorical"
    assert set(np.unique(first.data)) <= {0.0, 1.0, 2.0}


@pytest.mark.parametrize(
    ("options", "named"),
    [({"levels": (1, 3)}, "levels"), ({"concentration": 0.0}, "concentration")],
)
def test_bad_settings_are_refused(options, named):
    with pytest.raises(ValueError, match=named):
        DiscreteBN(graph=data.graphs.erdos_renyi(d=3, avg_degree=1), **options)
