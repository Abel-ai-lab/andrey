"""Discrete (categorical) Bayesian networks: random ones on a graph, and the published BIF tables.

A :class:`DiscreteNetwork` holds each variable's states and its conditional probability table, one
row per configuration of its parents, and samples rows ancestrally: every variable is drawn from
its table row given its parents' drawn states. :class:`DiscreteBN` draws such a network on a random
graph, with each table row from a symmetric Dirichlet, and returns a
:class:`~andrey.data.dataset.CausalDataset` whose columns hold the state codes ``0 .. r - 1``.
:func:`read_bif_network` reads a published network's own tables from its BIF file.

>>> import andrey.data as data
>>> bn = data.discrete.DiscreteBN(graph=data.graphs.erdos_renyi(d=6, avg_degree=2))
>>> ds = bn.sample(n=500, seed=0)
>>> ds.data.shape, ds.params.functional
((500, 6), 'categorical')
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import networkx as nx
import numpy as np

import andrey
from andrey.core import GraphStructure

from .dataset import CausalDataset, QAReport, SCMParams
from .graphs import DAGDraw, GraphSpec
from .scm import _GRAPH, _NOISE, _SEED_SCHEME, _WEIGHTS, _component_rng

__all__ = ["DiscreteBN", "DiscreteNetwork", "read_bif_network"]


@dataclass(frozen=True)
class DiscreteNetwork:
    """A discrete Bayesian network: states, parents, and a conditional probability table per node.

    Parameters
    ----------
    nodes : tuple of str
        The variables, in column order.
    states : tuple of tuple of str
        Each variable's state names; a sampled value ``k`` is state ``states[j][k]``.
    parents : tuple of tuple of int
        Each variable's parents, as column indices, in the order its table's rows use.
    tables : tuple of numpy.ndarray
        Each variable's table, shape ``(q, r)``: row ``i`` is the distribution of its ``r`` states
        under parent configuration ``i``, numbered in mixed radix with the last parent varying
        fastest. A root's table has one row.
    """

    nodes: tuple[str, ...]
    states: tuple[tuple[str, ...], ...]
    parents: tuple[tuple[int, ...], ...]
    tables: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        """Check every table's shape against its node's states and parents, and its rows' sums."""
        for j, (name, table) in enumerate(zip(self.nodes, self.tables, strict=True)):
            q = int(np.prod([len(self.states[p]) for p in self.parents[j]], dtype=np.int64))
            if table.shape != (q, len(self.states[j])):
                raise ValueError(
                    f"the table of {name!r} has shape {table.shape}, expected "
                    f"{(q, len(self.states[j]))}"
                )
            if (table < 0).any() or not np.allclose(table.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError(f"every row of the table of {name!r} must be a distribution")

    @property
    def arcs(self) -> tuple[tuple[str, str], ...]:
        """Every ``(parent, child)`` arc, by name."""
        return tuple(
            (self.nodes[p], self.nodes[j]) for j, ps in enumerate(self.parents) for p in ps
        )

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` rows ancestrally; return the ``int64`` state codes, shape ``(n, d)``."""
        d = len(self.nodes)
        graph = nx.DiGraph(
            (p, j) for j, ps in enumerate(self.parents) for p in ps
        )  # parent -> child
        graph.add_nodes_from(range(d))
        codes = np.zeros((n, d), dtype=np.int64)
        for j in nx.lexicographical_topological_sort(graph):
            row = np.zeros(n, dtype=np.int64)
            for p in self.parents[j]:
                row = row * len(self.states[p]) + codes[:, p]
            cumulative = np.cumsum(self.tables[j], axis=1)[row]
            cumulative[:, -1] = 1.0  # rounding can leave the last cumulative a hair below 1
            codes[:, j] = (rng.random((n, 1)) >= cumulative).sum(axis=1)
        return codes


@dataclass(frozen=True, kw_only=True)
class DiscreteBN:
    """A random discrete Bayesian network on a graph: categorical variables with Dirichlet tables.

    Each variable gets a number of states drawn uniformly from ``levels``, and each row of its
    table is drawn from a symmetric Dirichlet with parameter ``concentration``: below 1, rows put
    most of their mass on few states, so dependences are strong; above 1, rows approach uniform.

    Parameters
    ----------
    graph : GraphSpec
        A lazy graph spec (for example, ``andrey.data.graphs.erdos_renyi(...)``).
    levels : tuple of int, default=(2, 4)
        The fewest and the most states of a variable, each at least 2.
    concentration : float, default=1.0
        The Dirichlet parameter of every table row; positive.
    """

    graph: GraphSpec
    levels: tuple[int, int] = (2, 4)
    concentration: float = 1.0

    def __post_init__(self) -> None:
        """Check ``levels`` and ``concentration``."""
        low, high = self.levels
        if not 2 <= low <= high:
            raise ValueError(f"levels must be (low, high) with 2 <= low <= high, got {self.levels}")
        if not self.concentration > 0:
            raise ValueError(f"concentration must be positive, got {self.concentration}")

    def network(self, seed: int) -> DiscreteNetwork:
        """Return the network :meth:`sample` draws from with ``seed``: graph, states, and tables."""
        return self._draw(np.random.SeedSequence(seed))[2]

    def sample(self, n: int, seed: int) -> CausalDataset:
        """Draw ``n`` rows; return a :class:`~andrey.data.dataset.CausalDataset` of state codes.

        Parameters
        ----------
        n : int
            Number of samples (rows).
        seed : int
            Root seed; the graph, the tables, and the rows come from independent streams of it.

        Returns
        -------
        CausalDataset
            ``data`` holds each variable's state codes ``0 .. r - 1`` as ``float64``; ``params``
            names the ``"categorical"`` functional and ``"dirichlet"`` tables and carries no
            weights or noise scales.
        """
        root = np.random.SeedSequence(seed)
        truth, draw, network = self._draw(root)
        data = network.sample(int(n), _component_rng(root, _NOISE)).astype(np.float64)
        edges = np.stack([draw.parents, draw.children], axis=1)
        params = SCMParams(
            edges=edges,
            weights=np.zeros(0),
            noise_scales=np.zeros(0),
            functional="categorical",
            noise="dirichlet",
        )
        provenance = {
            "root_entropy": str(root.entropy),
            "spawn_key": list(root.spawn_key),
            "seed_scheme": _SEED_SCHEME,
            "config": {
                "graph": {
                    "model": self.graph.name,
                    "num_nodes": draw.d,
                    "density": self.graph.density,
                    "params": self.graph.params,
                },
                "functional": "categorical",
                "levels": list(self.levels),
                "concentration": float(self.concentration),
                "n": int(n),
            },
            "versions": {"andrey": andrey.__version__, "numpy": np.__version__},
        }
        return CausalDataset(
            data=data,
            graph=truth,
            report=QAReport(scale="raw"),
            provenance=provenance,
            params=params,
        )

    def _draw(
        self, root: np.random.SeedSequence
    ) -> tuple[GraphStructure, DAGDraw, DiscreteNetwork]:
        """Draw the graph, then the network's states and tables, from ``root``'s streams."""
        truth, draw = self.graph.materialize(_component_rng(root, _GRAPH))
        rng = _component_rng(root, _WEIGHTS)
        low, high = self.levels
        r = rng.integers(low, high + 1, size=draw.d)
        parents = tuple(
            tuple(sorted(int(p) for p in draw.parents[draw.children == j])) for j in range(draw.d)
        )
        tables = tuple(
            rng.dirichlet(
                np.full(int(r[j]), float(self.concentration)),
                size=int(np.prod(r[list(parents[j])], dtype=np.int64)),
            )
            for j in range(draw.d)
        )
        network = DiscreteNetwork(
            nodes=tuple(f"X{j}" for j in range(draw.d)),
            states=tuple(tuple(str(k) for k in range(int(r[j]))) for j in range(draw.d)),
            parents=parents,
            tables=tables,
        )
        return truth, draw, network


_VARIABLE = re.compile(
    r"variable\s+([^\s{]+)\s*\{[^}]*?discrete\s*\[\s*\d+\s*\]\s*\{([^}]*)\}", re.DOTALL
)
_PROBABILITY = re.compile(r"probability\s*\(([^)]*)\)\s*\{([^}]*)\}", re.DOTALL)
_ENTRY = re.compile(r"(\(([^)]*)\)|table)\s*([^;]*);")


def read_bif_network(text: str) -> DiscreteNetwork:
    """Return the discrete network a BIF file describes: its states, parents, and tables.

    Reads each ``variable``'s ``type discrete`` states and each ``probability`` block, in its
    ``table`` form or one ``(parent states) probabilities;`` line per parent configuration.

    Raises
    ------
    ValueError
        If a variable is not discrete, a block names an undeclared variable or state, or a table
        misses a parent configuration or does not sum to 1 in a row.
    """
    declared = re.findall(r"variable\s+([^\s{]+)\s*\{", text)
    states = {
        name: tuple(s.strip() for s in body.split(",")) for name, body in _VARIABLE.findall(text)
    }
    continuous = [name for name in declared if name not in states]
    if continuous:
        raise ValueError(f"the BIF variables {continuous} are not discrete")
    nodes = tuple(declared)
    index = {name: j for j, name in enumerate(nodes)}
    parents: list[tuple[int, ...]] = [()] * len(nodes)
    tables: list[np.ndarray | None] = [None] * len(nodes)
    for head, body in _PROBABILITY.findall(text):
        child, _, given = head.partition("|")
        child = child.strip()
        names = [p.strip() for p in given.split(",") if p.strip()]
        unknown = [v for v in [child, *names] if v not in index]
        if unknown:
            raise ValueError(f"the BIF probability block names undeclared variables {unknown}")
        sizes = [len(states[p]) for p in names]
        table = np.full((int(np.prod(sizes, dtype=np.int64)), len(states[child])), np.nan)
        for _, config, values in _ENTRY.findall(body):
            probs = [float(v) for v in values.replace(",", " ").split()]
            if not names:
                table[0] = probs
                continue
            row = 0
            for p, s in zip(names, (c.strip() for c in config.split(",")), strict=True):
                if s not in states[p]:
                    raise ValueError(
                        f"the BIF table of {child!r} names unknown state {s!r} of {p!r}"
                    )
                row = row * len(states[p]) + states[p].index(s)
            table[row] = probs
        if np.isnan(table).any():
            raise ValueError(f"the BIF table of {child!r} misses a parent configuration")
        parents[index[child]] = tuple(index[p] for p in names)
        tables[index[child]] = table
    missing = [nodes[j] for j, t in enumerate(tables) if t is None]
    if missing:
        raise ValueError(f"the BIF variables {missing} have no probability block")
    return DiscreteNetwork(
        nodes=nodes,
        states=tuple(states[name] for name in nodes),
        parents=tuple(parents),
        tables=tuple(t for t in tables if t is not None),
    )
