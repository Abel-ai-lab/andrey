"""The published networks of the target and latent lanes: the registry, the checked fetch, and the
structure and tables read from a file.

No test reaches the network: the cache holds the asia fixture, which ``andrey.data`` pins too, and
a source that cannot be reached is a refused URL.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from andrey_bench import networks

ASIA = Path(__file__).resolve().parents[2] / "tests" / "unit" / "fixtures" / "asia.bif.gz"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A network cache holding asia, and no way out to the network."""
    folder = tmp_path / "networks"
    folder.mkdir()
    shutil.copy(ASIA, folder / "asia.bif.gz")
    monkeypatch.setenv(networks.CACHE_ENV, str(folder))

    def refuse(*_args, **_kwargs):
        raise OSError("no network in tests")

    monkeypatch.setattr(networks.urllib.request, "urlopen", refuse)
    return folder


def test_the_four_networks_andrey_lacks_are_registered_with_pinned_files():
    assert set(networks.HARNESS_NETWORKS) == {"mildew", "win95pts", "hailfinder", "pigs"}
    for name, source in networks.HARNESS_NETWORKS.items():
        assert re.fullmatch(r"[0-9a-f]{64}", source.sha256), name
        assert all(url.startswith("https://") and name in url for url in source.sources), name
    sizes = {n: (s.nodes, s.arcs) for n, s in networks.HARNESS_NETWORKS.items()}
    assert sizes == {
        "mildew": (35, 46),
        "win95pts": (76, 112),
        "hailfinder": (56, 66),
        "pigs": (441, 592),
    }


def test_every_network_of_the_papers_can_be_loaded_by_name():
    names = networks.available()
    assert {"mildew", "alarm", "win95pts", "andes", "hailfinder", "pigs"} <= set(names)
    assert "sachs" not in names  # measured data, not a network
    with pytest.raises(ValueError, match="unknown network"):
        networks.load_network("nonesuch")


def test_a_network_is_read_from_its_checked_file(cache):
    net = networks.load_network("asia")
    assert (net.d, len(net.arcs)) == (8, 8)
    assert ("smoke", "lung") in net.arcs
    adjacency = net.adjacency
    assert adjacency.shape == (8, 8) and adjacency.sum() == 8
    assert adjacency[net.nodes.index("smoke"), net.nodes.index("lung")] == 1
    # Degrees count parents and children.
    assert net.degrees[net.nodes.index("either")] == 4


def test_a_cached_file_that_does_not_match_its_hash_is_never_read(cache):
    (cache / "asia.bif.gz").write_bytes(gzip.compress(b"variable x { }"))
    with pytest.raises(OSError, match="does not match"):
        networks.load_network("asia")


def test_a_file_that_is_absent_and_unreachable_names_where_to_put_it(cache):
    (cache / "asia.bif.gz").unlink()
    with pytest.raises(OSError, match="could not fetch asia.bif.gz") as error:
        networks.fetch_file("asia")
    assert str(cache) in str(error.value)


def test_a_download_with_the_wrong_hash_is_skipped(cache, monkeypatch):
    (cache / "asia.bif.gz").unlink()

    class Reply:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return self.body

    served = iter([b"not the network", ASIA.read_bytes()])
    monkeypatch.setattr(networks.urllib.request, "urlopen", lambda *_a, **_k: Reply(next(served)))
    raw = networks.fetch_file("asia")
    assert hashlib.sha256(raw).hexdigest() == networks._source_of("asia")[2]
    assert (cache / "asia.bif.gz").read_bytes() == raw  # the good one is kept


def test_a_file_with_other_counts_than_published_is_refused(cache, monkeypatch):
    monkeypatch.setitem(
        networks.HARNESS_NETWORKS,
        "asia",
        networks.Source((), networks._source_of("asia")[2], nodes=9, arcs=8),
    )
    with pytest.raises(ValueError, match="8 nodes and 8 arcs, not 9 and 8"):
        networks.load_network("asia")


def test_rows_are_drawn_from_the_tables_and_repeat_by_seed(cache):
    net = networks.load_network("asia")
    a, b = net.sample(500, seed=3), net.sample(500, seed=3)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (500, 8) and not np.array_equal(a, net.sample(500, seed=4))
    assert set(np.unique(a)) <= {0.0, 1.0}  # asia's variables have two categories
    with pytest.raises(ValueError, match="no tables"):
        networks.Network("bare", net.nodes, net.arcs).sample(5, 0)
