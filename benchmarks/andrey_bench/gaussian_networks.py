"""The fitted Gaussian networks of the adjustment lane: MAGIC-NIAB and SANGIOVESE.

The bnlearn repository publishes them as fitted R objects (``.rds``), which only R reads. The
harness reads each file once in the R lane's session (:mod:`andrey_bench.rsession`, bnlearn loaded)
and keeps the structure and the fitted parameters as a small JSON file next to the download, so
every later load needs no R. A download is checked against a SHA-256 before it is read, and the
node and arc counts against the published ones. Nothing here reaches ``andrey.data``.

A :class:`GaussianNetwork` is the linear-Gaussian model the fitted object describes: for each node,
an intercept, the coefficients of its parents, and the residual standard deviation. MAGIC-NIAB is
one (44 nodes, 66 arcs). SANGIOVESE is a conditional linear Gaussian network whose discrete root
``Treatment`` (16 levels) selects the coefficients of three nodes; the paper that uses it
(Maasch et al., 2025, Sect. 5) calls it a linear-Gaussian model and does not say how ``Treatment``
was handled, so the harness fixes ``Treatment`` at one level (the most probable by default) and
drops the column: the other 14 nodes are then exactly a linear-Gaussian network.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import uuid
from dataclasses import dataclass

import numpy as np

from andrey_bench.networks import cache_dir

_BNLEARN = "https://www.bnlearn.com/bnrepository/{name}/{name}.rds"


@dataclass(frozen=True)
class Source:
    """A network's file: its URL, SHA-256, and the published node and arc counts of the structure
    the harness keeps (SANGIOVESE without its discrete root)."""

    url: str
    sha256: str
    nodes: int
    arcs: int


#: The fitted networks the lane reads. SANGIOVESE has 15 nodes and 55 arcs in the repository; the
#: discrete root ``Treatment`` and its 3 arcs (to ``SproutN``, ``BunchN``, ``SPAD06``) and its
#: 1 more to ``Brix`` leave 14 nodes and 51 arcs.
NETWORKS: dict[str, Source] = {
    "magic-niab": Source(
        _BNLEARN.format(name="magic-niab"),
        "78637b0054ddb9c5b90cd7568f79a4ecd9a7b7975b6cfe8001c902a2d7b2af71",
        44,
        66,
    ),
    "sangiovese": Source(
        _BNLEARN.format(name="sangiovese"),
        "8a51ef68ad458d44c85e62810a651ce6cb2899c3eddc22fc3028d03b1fce6f39",
        14,
        51,
    ),
}

#: The R function the reader defines: structure and parameters of a ``bn.fit`` Gaussian or
#: conditional Gaussian network, discrete nodes fixed at ``level`` (ignored for a Gaussian network).
_R_READER = """
andrey_read_gaussian <- function(path, level) {
    fit <- readRDS(path)
    cls <- vapply(fit, function(n) class(n)[1], "")
    discrete <- names(fit)[cls == "bn.fit.dnode"]
    if (length(discrete) > 1L) stop("more than one discrete node")
    if (length(discrete) == 1L && !nzchar(level)) level <- names(which.max(fit[[discrete]]$prob))
    keep <- setdiff(names(fit), discrete)
    from <- character(); to <- character(); coef <- numeric()
    intercept <- numeric(); sd <- numeric()
    for (v in keep) {
        n <- fit[[v]]
        if (cls[[v]] == "bn.fit.cgnode") {
            cf <- n$coefficients[, level]; s <- n$sd[[level]]
        } else {
            cf <- n$coefficients; s <- n$sd
        }
        for (p in setdiff(n$parents, discrete)) {
            from <- c(from, p); to <- c(to, v); coef <- c(coef, unname(cf[[p]]))
        }
        intercept <- c(intercept, unname(cf[["(Intercept)"]])); sd <- c(sd, s)
    }
    list(nodes = keep, intercept = intercept, sd = sd, from = from, to = to, coef = coef,
         level = level, narcs_all = bnlearn::narcs(fit))
}
"""


@dataclass(frozen=True)
class GaussianNetwork:
    """A linear-Gaussian network: node names in file order, ``weights[i, j]`` the coefficient of
    the arc ``i -> j``, the intercepts, and the residual standard deviations."""

    name: str
    nodes: tuple[str, ...]
    weights: np.ndarray
    intercepts: np.ndarray
    noise_sd: np.ndarray
    level: str = ""

    @property
    def d(self) -> int:
        return len(self.nodes)

    def index(self, node: str) -> int:
        """The position of ``node`` in file order."""
        return self.nodes.index(node)


def _checked_file(name: str) -> bytes:
    """The bytes of ``name``'s file: the cached one, else a download that matches the SHA-256."""
    source = NETWORKS[name]
    path = cache_dir() / f"{name}.rds"
    if path.is_file():
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != source.sha256:
            raise OSError(
                f"{path} does not match its SHA-256 {source.sha256}; delete or replace it"
            )
        return raw
    with urllib.request.urlopen(source.url, timeout=120) as response:
        raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != source.sha256:
        raise OSError(f"{source.url} has SHA-256 {digest}, not {source.sha256}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
    try:
        partial.write_bytes(raw)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return raw


def read_rds(name: str, level: str = "") -> dict:
    """Read ``name``'s file in the R lane's session and return its structure and parameters.

    Needs the R lane (:func:`andrey_bench.rsession.start` with bnlearn). ``level`` fixes the
    discrete root of a conditional Gaussian network, the most probable level when empty.
    """
    from andrey_bench import rsession

    _checked_file(name)
    session = rsession.start("bnlearn")
    session.define(_R_READER)
    out = session.function("andrey_read_gaussian")(str(cache_dir() / f"{name}.rds"), level)
    keys = list(out.names)
    values = {k: out[i] for i, k in enumerate(keys)}
    return {
        "nodes": [str(v) for v in values["nodes"]],
        "intercept": [float(v) for v in values["intercept"]],
        "sd": [float(v) for v in values["sd"]],
        "arcs": [
            [str(a), str(b), float(c)]
            for a, b, c in zip(values["from"], values["to"], values["coef"], strict=True)
        ],
        "level": str(values["level"][0]),
        "rds_sha256": NETWORKS[name].sha256,
    }


def from_structure(name: str, spec: dict) -> GaussianNetwork:
    """The network a :func:`read_rds`-style dict describes."""
    nodes = tuple(spec["nodes"])
    index = {v: i for i, v in enumerate(nodes)}
    weights = np.zeros((len(nodes), len(nodes)))
    for a, b, c in spec["arcs"]:
        weights[index[a], index[b]] = c
    return GaussianNetwork(
        name=name,
        nodes=nodes,
        weights=weights,
        intercepts=np.array(spec["intercept"]),
        noise_sd=np.array(spec["sd"]),
        level=str(spec.get("level", "")),
    )


def load_gaussian_network(name: str, level: str = "") -> GaussianNetwork:
    """The network ``name``: from the JSON kept next to the download, else read once in R.

    Raises ``ValueError`` when the structure is not the published size, and ``OSError`` when the
    file does not match its SHA-256.
    """
    if name not in NETWORKS:
        raise ValueError(f"unknown network {name!r}; the networks are {sorted(NETWORKS)}")
    source = NETWORKS[name]
    kept = cache_dir() / f"{name}.structure.json"
    spec = None
    if kept.is_file():
        spec = json.loads(kept.read_text())
        if spec.get("rds_sha256") != source.sha256 or (level and spec.get("level") != level):
            spec = None
    if spec is None:
        spec = read_rds(name, level)
        kept.write_text(json.dumps(spec))
    if (len(spec["nodes"]), len(spec["arcs"])) != (source.nodes, source.arcs):
        raise ValueError(
            f"{name} has {len(spec['nodes'])} nodes and {len(spec['arcs'])} arcs, "
            f"not {source.nodes} and {source.arcs}"
        )
    return from_structure(name, spec)
