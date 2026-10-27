"""A d-separation oracle: the conditional-independence test that the true DAG answers exactly."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np

from andrey import GraphStructure


class DSeparation:
    """A :class:`~andrey.core.ci.CITest` that returns 1.0 when the DAG d-separates ``x`` and ``y``
    given the conditioning set, and 0.0 otherwise.

    It counts its calls in ``calls``.
    """

    def __init__(self, dag: GraphStructure) -> None:
        self.d = dag.n_nodes
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(self.d))
        self.graph.add_edges_from((i, j) for i, j, _ in dag.oriented_edges())
        self.calls = 0

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        self.calls += 1
        return float(nx.is_d_separator(self.graph, {x}, {y}, set(condition_set)))

    def parents_children(self, v: int) -> set[int]:
        """The true parents and children of ``v``."""
        return set(self.graph.predecessors(v)) | set(self.graph.successors(v))

    def markov_blanket(self, v: int) -> set[int]:
        """The true Markov blanket of ``v``: its parents, children, and its children's parents."""
        spouses = {p for c in self.graph.successors(v) for p in self.graph.predecessors(c)}
        return (self.parents_children(v) | spouses) - {v}

    def skeleton(self) -> np.ndarray:
        """The true adjacencies, a symmetric boolean matrix."""
        adj = nx.to_numpy_array(self.graph, nodelist=range(self.d)) != 0
        return adj | adj.T

    def moral_graph(self) -> np.ndarray:
        """The true moral graph: each variable joined to its Markov blanket."""
        adj = np.zeros((self.d, self.d), dtype=bool)
        for v in range(self.d):
            adj[v, sorted(self.markov_blanket(v))] = True
        return adj
