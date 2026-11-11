"""The published Bayesian networks of the target and latent lanes, read for the harness.

``andrey.data`` ships some of the bnlearn networks (``alarm``, ``andes``, and others);
:data:`HARNESS_NETWORKS` adds the four its papers use that it does not: ``mildew``, ``win95pts``,
``hailfinder``, and ``pigs``. Their files are fetched from the bnlearn repository, or from its
pgmpy mirror at a pinned commit, on first use; each download is checked against a SHA-256 and kept
in a git-ignored folder (``benchmarks/.networks``, or ``$ANDREY_BENCH_NETWORKS``), and a file that
does not match is never read. :func:`load_network` reads the structure with ``read_bif`` and checks
it against the published node and arc counts. Nothing here reaches ``andrey.data``'s own dataset
registry.

A :class:`Network` gives the DAG as an adjacency and as endpoint marks. :meth:`Network.sample`
draws rows of categories from the network's own tables, for the discrete lanes.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from andrey.data import real

#: The variable naming the folder that holds the network files.
CACHE_ENV = "ANDREY_BENCH_NETWORKS"

_BNLEARN = "https://www.bnlearn.com/bnrepository/{name}/{name}.bif.gz"
_PGMPY = (
    "https://raw.githubusercontent.com/pgmpy/example_models/"
    "920f4e03fce2a319af0fbc8dc6fad1ac99ff4318/discrete/{name}.bif.gz"
)


@dataclass(frozen=True)
class Source:
    """A network's file: the URLs to try in order, its SHA-256, and its published size."""

    sources: tuple[str, ...]
    sha256: str
    nodes: int
    arcs: int


def _source(name: str, sha256: str, nodes: int, arcs: int) -> Source:
    return Source((_BNLEARN.format(name=name), _PGMPY.format(name=name)), sha256, nodes, arcs)


#: The networks ``andrey.data`` does not ship, with the node and arc counts of the bnlearn
#: repository's pages.
HARNESS_NETWORKS: dict[str, Source] = {
    "mildew": _source(
        "mildew", "06f4342f1bef2eef67988b41382b327ca7bf4b308841534c9a3b3c48ff61189a", 35, 46
    ),
    "win95pts": _source(
        "win95pts", "a4987cf8c09d3aa034504b03ea3e5dc01c57ca20228e2bd79ec9a861c3aef684", 76, 112
    ),
    "hailfinder": _source(
        "hailfinder", "efa122a88c4b550569ec821c7879bed9007f2b1e012e1ea52ea26069c7f26099", 56, 66
    ),
    "pigs": _source(
        "pigs", "618c72433d8814e5197f78938e9ef75cf91dfb9a7463952682b3703c8eef68f1", 441, 592
    ),
}


def available() -> tuple[str, ...]:
    """The names :func:`load_network` accepts: the harness's four and ``andrey.data``'s."""
    return tuple(HARNESS_NETWORKS) + tuple(
        name for name, remote in real.DATASETS.items() if not remote.measured
    )


def cache_dir() -> Path:
    """``$ANDREY_BENCH_NETWORKS``, else ``benchmarks/.networks``."""
    configured = os.environ.get(CACHE_ENV, "").strip()
    return Path(configured) if configured else Path(__file__).resolve().parents[1] / ".networks"


def _source_of(name: str) -> tuple[str, tuple[str, ...], str, int | None, int | None]:
    """The file name, URLs, SHA-256, and published sizes of ``name``."""
    if name in HARNESS_NETWORKS:
        s = HARNESS_NETWORKS[name]
        return f"{name}.bif.gz", s.sources, s.sha256, s.nodes, s.arcs
    remote = real.DATASETS.get(name)
    if remote is None or remote.measured:
        raise ValueError(f"unknown network {name!r}; the networks are {available()}")
    return remote.filename, remote.sources, remote.sha256, None, None


def fetch_file(name: str, folder: Path | None = None) -> bytes:
    """The checked bytes of ``name``'s file: the cached one, else the first source that matches.

    A cached file that does not match the SHA-256 raises ``OSError`` and is left in place. A
    download that does not match is skipped.
    """
    filename, sources, sha256, _, _ = _source_of(name)
    path = (folder or cache_dir()) / filename
    if path.is_file():
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha256:
            raise OSError(f"{path} does not match its SHA-256 {sha256}; delete or replace it")
        return raw
    tried = []
    for url in sources:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read()
        except OSError as error:
            tried.append(f"  {url}: {error}")
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest != sha256:
            tried.append(f"  {url}: SHA-256 {digest}, not {sha256}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # A file of its own per download, so two jobs fetching at once do not move each other's.
        partial = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
        try:
            partial.write_bytes(raw)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
        return raw
    raise OSError(
        f"could not fetch {filename}; tried:\n" + "\n".join(tried) + f"\nPut it at {path}"
    )


@dataclass(frozen=True)
class Network:
    """A published network: its variables in file order and its arcs ``(parent, child)``."""

    name: str
    nodes: tuple[str, ...]
    arcs: tuple[tuple[str, str], ...]
    #: The file's text, kept for the tables; empty for a network built from a structure alone.
    text: str = ""

    @property
    def d(self) -> int:
        return len(self.nodes)

    @property
    def adjacency(self) -> np.ndarray:
        """``adjacency[i, j] = 1`` for the arc ``i -> j``, nodes in file order, as ``int8``."""
        index = {v: i for i, v in enumerate(self.nodes)}
        adj = np.zeros((self.d, self.d), dtype=np.int8)
        for parent, child in self.arcs:
            adj[index[parent], index[child]] = 1
        return adj

    @property
    def degrees(self) -> np.ndarray:
        """Each node's number of neighbors, parents and children."""
        adj = self.adjacency
        return adj.sum(0) + adj.sum(1)

    def sample(self, n: int, seed: int) -> np.ndarray:
        """``n`` rows of category codes, drawn from the network's own tables, as ``float64``.

        Raises ``ValueError`` for a network built without its file's text.
        """
        if not self.text:
            raise ValueError(f"{self.name} has no tables: it was built from a structure alone")
        return real.read_bif_tables(self.text).sample(n, np.random.default_rng(seed))


def parse_network(name: str, text: str) -> Network:
    """The network a BIF file's text describes, its structure read with ``read_bif``."""
    nodes, arcs = real.read_bif(text)
    return Network(name=name, nodes=nodes, arcs=arcs, text=text)


def load_network(name: str, folder: Path | None = None) -> Network:
    """The network ``name``, fetched and checked as :func:`fetch_file` says.

    Raises ``ValueError`` when the file's node or arc count is not the published one.
    """
    network = parse_network(name, gzip.decompress(fetch_file(name, folder)).decode("utf-8"))
    _, _, _, nodes, arcs = _source_of(name)
    if (nodes, arcs) != (None, None) and (network.d, len(network.arcs)) != (nodes, arcs):
        raise ValueError(
            f"{name} has {network.d} nodes and {len(network.arcs)} arcs, not {nodes} and {arcs}"
        )
    return network
