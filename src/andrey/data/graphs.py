"""Causal-graph topology specs and sparse DAG-truth construction.

A graph builder is a **lazy spec** (:class:`GraphSpec`): ``graphs.erdos_renyi(d, avg_degree)``
returns a description, and the draw happens only inside ``SCM.sample`` from an explicit RNG, so a
dataset is regenerable from its recorded seed alone. Truth is assembled straight into the
endpoint-mark CSR via :func:`dag_truth` -- never through a dense ``(d, d)`` matrix -- so generation
scales to ``d`` in the tens of thousands.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt

from andrey.core.structure import ARROW, TAIL, GraphStructure, Kind

#: Reported on their own beside ``params``, so no knob appears twice.
_GENERAL_FIELDS = frozenset({"name", "d"})

#: ``model -> (d -> (min, max))`` mean degree the builder can draw. Erdos-Renyi is continuous;
#: the rest step in twos off their sparsest legal graph, and hub is bounded by its hub count.
_DENSITY_RANGE: dict[str, Any] = {
    "erdos_renyi": lambda d: (0.0, float(max(0, d - 1))),
    "scale_free": lambda d: (2.0, 2.0 * max(1, d - 1)),
    "small_world": lambda d: (2.0, float(max(2, d - 1 - ((d - 1) % 2)))),
    "hub": lambda d: (2.0 * (d - 3) / d, 6.0 * (d - 3) / d) if d > 3 else (0.0, 2.0),
}


@dataclass(frozen=True)
class DAGDraw:
    """A materialized DAG: its directed edges (label space) and a topological node order.

    Parameters
    ----------
    d : int
        Node count.
    parents, children : np.ndarray of shape (m,)
        Directed edges ``parents[k] -> children[k]`` in node-label space.
    topo_order : np.ndarray of shape (d,)
        Node labels in a valid topological order (every parent precedes its children).
    """

    d: int
    parents: np.ndarray
    children: np.ndarray
    topo_order: np.ndarray


@dataclass(frozen=True, kw_only=True)
class GraphSpec:
    """A lazy causal-graph specification; :meth:`materialize` performs the seeded draw."""

    name: str

    def materialize(self, rng: np.random.Generator) -> tuple[GraphStructure, DAGDraw]:
        """Draw a DAG from ``rng``; return its truth ``GraphStructure`` and a :class:`DAGDraw`."""
        raise NotImplementedError

    @property
    def density(self) -> float:
        """How dense the draw this spec describes is, as **mean total degree** (``2 * |E| / d``).

        The one axis every generator answers, whatever its own knob spells it as. Nominal, because
        the realized degree does not exist until the graph is drawn: rounding and the builders' own
        clamps move it.
        """
        raise NotImplementedError

    @property
    def params(self) -> dict[str, Any]:
        """The builder's own knobs — everything :data:`_GENERAL_FIELDS` does not already carry."""
        return {
            f.name: getattr(self, f.name) for f in fields(self) if f.name not in _GENERAL_FIELDS
        }


@dataclass(frozen=True, kw_only=True)
class ErdosRenyi(GraphSpec):
    """Erdos-Renyi random DAG: each of the ``C(d, 2)`` node pairs is an edge with equal probability.

    Parameters
    ----------
    d : int
        Number of nodes.
    avg_degree : float
        Target mean total degree (``2 * n_edges / d``); the edge count is
        ``round(avg_degree * d / 2)``.
    """

    name: str = "erdos_renyi"
    d: int
    avg_degree: float

    @property
    def density(self) -> float:
        """The knob is the axis."""
        return float(self.avg_degree)

    def materialize(self, rng: np.random.Generator) -> tuple[GraphStructure, DAGDraw]:
        """Draw an Erdos-Renyi DAG from ``rng``; return its truth and a :class:`DAGDraw`."""
        r, c = _er_position_edges(self.d, self.avg_degree, rng)  # positions, r < c (topological)
        perm = rng.permutation(self.d)  # relabel so index order != causal order
        parents, children = perm[r], perm[c]
        truth = dag_truth(parents, children, self.d)
        return truth, DAGDraw(d=self.d, parents=parents, children=children, topo_order=perm)


def params_for_density(model: str, d: int, density: float) -> dict[str, Any]:
    """Translate a target mean degree into ``model``'s own knobs.

    ``d`` and ``density`` are the two axes every generator answers; this is where the second one
    becomes whatever that generator actually takes. Rough by nature — a knob that moves in steps of
    two cannot express an odd degree, so the result is the nearest density the topology can draw.

    Raises ``ValueError`` when the target is outside what the topology can draw at all, rather than
    clamping to its nearest legal graph and recording a density that was never requested.
    """
    if model not in _DENSITY_RANGE:
        raise ValueError(f"unknown graph {model!r}; supported: {sorted(_DENSITY_RANGE)}")
    lo, hi = _DENSITY_RANGE[model](d)
    if not lo <= density <= hi:
        raise ValueError(
            f"{model} on {d} nodes draws mean degree {lo:g} to {hi:g}; "
            f"{density:g} is outside that. Change the density, the node count, or the graph."
        )
    if model == "erdos_renyi":
        return {"avg_degree": float(density)}
    if model == "scale_free":
        return {"m": _half(density)}
    if model == "small_world":
        return {"k": max(2, int(density + 0.5))}
    return {"edges_per_spoke": _half(density)}


def _half(density: float) -> int:
    """``density / 2`` to the nearest integer, ties up — a knob that adds two degrees per step."""
    return max(1, int(density / 2 + 0.5))


def erdos_renyi(d: int, avg_degree: float = 4.0) -> ErdosRenyi:
    """Build an Erdos-Renyi DAG spec on ``d`` nodes with target mean degree ``avg_degree``."""
    return ErdosRenyi(d=int(d), avg_degree=float(avg_degree))


@dataclass(frozen=True, kw_only=True)
class ScaleFree(GraphSpec):
    """Barabasi-Albert scale-free DAG: preferential attachment gives a heavy-tailed degree hub.

    Parameters
    ----------
    d : int
        Number of nodes.
    m : int, default=2
        Edges added per new node (``1 <= m < d``); mean degree is ``~2 * m``.
    """

    name: str = "scale_free"
    d: int
    m: int = 2

    @property
    def density(self) -> float:
        """``2 * m``: preferential attachment adds ``m`` edges per node."""
        return 2.0 * self.m

    def materialize(self, rng: np.random.Generator) -> tuple[GraphStructure, DAGDraw]:
        """Draw a scale-free DAG from ``rng``; return its truth and a :class:`DAGDraw`."""
        if self.d < 2:  # barabasi_albert needs n >= 2
            return _dag_from_undirected(nx.empty_graph(self.d), self.d, rng)
        m = max(1, min(self.m, self.d - 1))
        g = nx.barabasi_albert_graph(self.d, m, seed=_nx_seed(rng))
        return _dag_from_undirected(g, self.d, rng)


@dataclass(frozen=True, kw_only=True)
class SmallWorld(GraphSpec):
    """Watts-Strogatz small-world DAG: a ring lattice with a fraction of rewired edges.

    Parameters
    ----------
    d : int
        Number of nodes.
    k : int, default=4
        Each node joins its ``k`` nearest ring neighbors (rounded down to even, ``< d``).
    p : float, default=0.1
        Rewiring probability.
    """

    name: str = "small_world"
    d: int
    k: int = 4
    p: float = 0.1

    @property
    def density(self) -> float:
        """``k``, rounded down to even as the ring lattice does."""
        return float(self.k - (self.k % 2))

    def materialize(self, rng: np.random.Generator) -> tuple[GraphStructure, DAGDraw]:
        """Draw a small-world DAG from ``rng``; return its truth and a :class:`DAGDraw`."""
        if self.d < 3:  # watts_strogatz needs a ring of degree k >= 2 with k < d
            return _dag_from_undirected(nx.empty_graph(self.d), self.d, rng)
        k = min(self.k - (self.k % 2), self.d - 1 - ((self.d - 1) % 2))
        k = max(2, k)
        g = nx.watts_strogatz_graph(self.d, k, self.p, seed=_nx_seed(rng))
        return _dag_from_undirected(g, self.d, rng)


@dataclass(frozen=True, kw_only=True)
class Hub(GraphSpec):
    """Hub DAG: a few high-degree hubs each fanning out to spoke nodes.

    Parameters
    ----------
    d : int
        Number of nodes.
    n_hubs : int, default=3
        Number of hub nodes.
    edges_per_spoke : int, default=1
        Hubs each spoke attaches to (``<= n_hubs``); mean degree is ``~2 * edges_per_spoke``. This
        is the density knob — ``n_hubs`` sets the shape.
    """

    name: str = "hub"
    d: int
    n_hubs: int = 3
    edges_per_spoke: int = 1

    @property
    def density(self) -> float:
        """One edge per spoke per hub it joins, over ``d`` nodes."""
        n_hubs = max(1, min(self.n_hubs, self.d - 1))
        per_spoke = max(1, min(self.edges_per_spoke, n_hubs))
        return 2.0 * per_spoke * (self.d - n_hubs) / self.d

    def materialize(self, rng: np.random.Generator) -> tuple[GraphStructure, DAGDraw]:
        """Draw a hub DAG from ``rng``; return its truth and a :class:`DAGDraw`."""
        n_hubs = max(1, min(self.n_hubs, self.d - 1))
        per_spoke = max(1, min(self.edges_per_spoke, n_hubs))
        perm = rng.permutation(self.d)
        hubs, spokes = perm[:n_hubs], perm[n_hubs:]
        if per_spoke == 1:
            u, v = rng.choice(hubs, size=spokes.shape[0]), spokes
        else:
            # argsort of a uniform row draws ``per_spoke`` *distinct* hubs per spoke at once.
            pick = rng.random((spokes.shape[0], n_hubs)).argsort(axis=1)[:, :per_spoke]
            u, v = hubs[pick].ravel(), np.repeat(spokes, per_spoke)
        parents, children, topo = _orient_undirected_as_dag(u, v, self.d, rng)
        return dag_truth(parents, children, self.d), DAGDraw(
            d=self.d, parents=parents, children=children, topo_order=topo
        )


@dataclass(frozen=True, kw_only=True)
class FixedDAG(GraphSpec):
    """A published DAG: every draw returns the same graph; the seed varies weights and noise.

    Parameters
    ----------
    nodes : tuple of str
        The variables, in column order.
    arcs : tuple of (str, str)
        Each ``(parent, child)`` arc. Their order fixes which weight each arc draws, so the same
        arcs in another order sample different data.
    """

    name: str = "fixed"
    nodes: tuple[str, ...]
    arcs: tuple[tuple[str, str], ...]

    @property
    def density(self) -> float:
        """``2 * |arcs| / |nodes|``, exact: the graph does not vary."""
        return 2 * len(self.arcs) / len(self.nodes)

    def materialize(self, rng: np.random.Generator) -> tuple[GraphStructure, DAGDraw]:
        """Return the DAG's truth and a :class:`DAGDraw`; ``rng`` is not used."""
        index = {name: i for i, name in enumerate(self.nodes)}
        parents = np.array([index[a] for a, _ in self.arcs], dtype=np.int64)
        children = np.array([index[b] for _, b in self.arcs], dtype=np.int64)
        d = len(self.nodes)
        graph = nx.DiGraph(zip(parents.tolist(), children.tolist(), strict=True))
        graph.add_nodes_from(range(d))
        order = np.array(list(nx.topological_sort(graph)))
        draw = DAGDraw(d=d, parents=parents, children=children, topo_order=order)
        return dag_truth(parents, children, d), draw


def scale_free(d: int, m: int = 2) -> ScaleFree:
    """Build a Barabasi-Albert scale-free DAG spec (``m`` attachments per node)."""
    return ScaleFree(d=int(d), m=int(m))


def small_world(d: int, k: int = 4, p: float = 0.1) -> SmallWorld:
    """Build a Watts-Strogatz small-world DAG spec (ring degree ``k``, rewiring ``p``)."""
    return SmallWorld(d=int(d), k=int(k), p=float(p))


def hub(d: int, n_hubs: int = 3, edges_per_spoke: int = 1) -> Hub:
    """Build a hub DAG spec: ``n_hubs`` hubs, each spoke joining ``edges_per_spoke`` of them."""
    return Hub(d=int(d), n_hubs=int(n_hubs), edges_per_spoke=int(edges_per_spoke))


def _nx_seed(rng: np.random.Generator) -> int:
    """Draw a 31-bit int seed for a networkx generator from the Generator (keeps one RNG root)."""
    return int(rng.integers(0, 2**31 - 1))


def _dag_from_undirected(
    g: nx.Graph, d: int, rng: np.random.Generator
) -> tuple[GraphStructure, DAGDraw]:
    """Orient an undirected networkx graph into a DAG via a random topological rank."""
    if g.number_of_edges() == 0:
        empty = np.empty(0, dtype=np.int64)
        return dag_truth(empty, empty, d), DAGDraw(
            d=d, parents=empty, children=empty, topo_order=rng.permutation(d)
        )
    edges = np.asarray(g.edges(), dtype=np.int64)
    parents, children, topo = _orient_undirected_as_dag(edges[:, 0], edges[:, 1], d, rng)
    return dag_truth(parents, children, d), DAGDraw(
        d=d, parents=parents, children=children, topo_order=topo
    )


def _orient_undirected_as_dag(
    u: np.ndarray, v: np.ndarray, d: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orient undirected edges into a DAG by a random node rank.

    Low rank -> high rank = acyclic.
    """
    rank = np.empty(d, dtype=np.int64)
    rank[rng.permutation(d)] = np.arange(d)  # rank[node] = its topological position
    u, v = np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64)
    lo_first = rank[u] < rank[v]
    parents = np.where(lo_first, u, v)
    children = np.where(lo_first, v, u)
    topo_order = np.argsort(
        rank, kind="stable"
    )  # node labels in ascending-rank (topological) order
    return parents, children, topo_order


def dag_truth(
    parents: np.ndarray, children: np.ndarray, d: int, *, node_types: np.ndarray | None = None
) -> GraphStructure:
    """Build a ``kind="dag"`` ground-truth graph straight into the symmetric endpoint-mark CSR.

    Each directed edge ``p -> c`` sets a ``TAIL`` at ``p`` and an ``ARROW`` at ``c`` (the endpoint
    convention: ``M[p, c] = TAIL``, ``M[c, p] = ARROW``). The CSR is assembled vectorized -- no
    dense ``(d, d)`` matrix -- so ``d`` in the tens of thousands stays cheap.

    Parameters
    ----------
    parents, children : np.ndarray of shape (m,)
        Directed edges ``parents[k] -> children[k]``; must be a DAG (unchecked here -- validated by
        the caller's acyclicity test).
    d : int
        Node count.
    node_types : np.ndarray of shape (d,) or None, default=None
        Per-node OBSERVED/LATENT codes; ``None`` when every node is observed.

    Returns
    -------
    GraphStructure
        The DAG truth, backed by the canonical endpoint-mark CSR.
    """
    parents = np.asarray(parents, dtype=np.int64)
    children = np.asarray(children, dtype=np.int64)
    m = parents.shape[0]
    rows = np.concatenate([parents, children])
    cols = np.concatenate([children, parents])
    marks = np.concatenate([np.full(m, TAIL, dtype=np.int8), np.full(m, ARROW, dtype=np.int8)])
    return csr_from_endpoints(rows, cols, marks, d, kind="dag", node_types=node_types)


def csr_from_endpoints(
    rows: npt.ArrayLike,
    cols: npt.ArrayLike,
    marks: npt.ArrayLike,
    n: int,
    *,
    kind: Kind,
    node_types: np.ndarray | None = None,
) -> GraphStructure:
    """Assemble a symmetric endpoint-mark CSR from parallel ``(row, col, mark)`` entries.

    ``rows``/``cols``/``marks`` are any array-likes (lists or arrays): they are coerced to the CSR
    dtypes here, so callers may accumulate plain Python lists. Each undirected pair contributes two
    entries (``i->j`` and ``j->i``) so the support is symmetric; sorts row-major (columns ascending
    within a row) and builds ``indptr`` by a bincount over rows. Never materializes a dense
    ``(n, n)`` matrix.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    marks = np.asarray(marks, dtype=np.int8)
    order = np.lexsort((cols, rows))  # row-major, columns ascending within a row
    rows, cols, marks = rows[order], cols[order], marks[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, rows + 1, 1)
    np.cumsum(indptr, out=indptr)
    return GraphStructure._from_csr(n, indptr, cols, marks, kind=kind, node_types=node_types)


def _er_position_edges(
    d: int, avg_degree: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``round(avg_degree * d / 2)`` upper-triangular position edges (``r < c``).

    No dense grid: draws unique linear indices into the ``C(d, 2)`` condensed upper triangle by
    with-replacement sampling + dedup (never ``choice(replace=False)`` on the full population,
    which is ``O(d^2)`` memory), then inverts each index to an ``(r, c)`` pair. ``r < c`` makes
    position order a valid topological order before relabeling.
    """
    empty = np.empty(0, dtype=np.int64)
    if d < 2:
        return empty, empty
    n_pairs = d * (d - 1) // 2
    target = min(int(round(avg_degree * d / 2)), n_pairs)
    if target == 0:
        return empty, empty
    lin = np.unique(rng.integers(0, n_pairs, size=int(target * 1.2) + 16))
    while (
        lin.size < target
    ):  # dedup can lose >20% in dense regimes; top up to hit the target exactly
        extra = rng.integers(0, n_pairs, size=(target - lin.size) * 2 + 16)
        lin = np.unique(np.concatenate([lin, extra]))
    if lin.size > target:
        lin = lin[rng.choice(lin.size, size=target, replace=False)]  # unbiased trim (small array)
    r, c = _condensed_to_rc(lin, d)
    return r, c


def _condensed_to_rc(m: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Invert row-major upper-triangular (``i < j``) condensed indices to ``(i, j)`` pairs.

    The closed-form inverse of the condensed-distance-matrix index, so 10k-node graphs never
    enumerate the ``O(d^2)`` pair grid.
    """
    m = np.asarray(m, dtype=np.int64)
    dd = float(d)
    ii = (dd - 2.0 - np.floor(np.sqrt(-8.0 * m + 4.0 * dd * (dd - 1.0) - 7.0) / 2.0 - 0.5)).astype(
        np.int64
    )
    jj = m + ii + 1 - (d * (d - 1)) // 2 + ((d - ii) * (d - ii - 1)) // 2
    return ii, jj
