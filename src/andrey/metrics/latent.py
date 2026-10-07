"""Latent-variable metrics: how well an inferred latent clustering recovers the true one.

Latent-cluster methods (GIN) group observed variables under shared latent parents. These score that
grouping with the adjusted Rand index, plus a count of recovered latents. Latents are the nodes a
structure flags ``LATENT`` (``node_types``); each observed node's cluster is its latent parent.
"""

from __future__ import annotations

import numpy as np

from andrey.core.output import StructureOutput
from andrey.core.structure import LATENT, GraphStructure

from ._common import directed_adjacency


def _graph(x: object) -> GraphStructure:
    """Unwrap a StructureOutput to its GraphStructure (latent info is in ``node_types``)."""
    if isinstance(x, StructureOutput):
        x = x.structure
    if not isinstance(x, GraphStructure):
        raise TypeError("latent metrics need a GraphStructure with node_types")
    return x


def _clusters(g: GraphStructure) -> dict[int, object]:
    """Map each observed node to a cluster key: the frozenset of its latent parents.

    A merged cluster (an observed node under two latents) keys on the whole parent set, so it stays
    a single well-defined cluster. An observed node with no latent parent gets a unique singleton
    key, so unrelated unclustered nodes are not conflated.
    """
    marks = g.to_numpy()
    types = g.node_types
    latent = types == LATENT if types is not None else np.zeros(g.n_nodes, dtype=bool)
    parents_of = directed_adjacency(marks).T  # parents_of[o] = nodes pointing into o
    clusters: dict[int, object] = {}
    for o in np.nonzero(~latent)[0]:
        latent_parents = np.nonzero(parents_of[o] & latent)[0]
        clusters[int(o)] = (
            frozenset(int(p) for p in latent_parents)
            if latent_parents.size
            else ("singleton", int(o))
        )
    return clusters


def adjusted_rand_index(labels_a: object, labels_b: object) -> float:
    """Adjusted Rand index between two label vectors (chance-corrected; 1.0 is identical).

    Examples
    --------
    >>> adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0])
    1.0
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError(f"label-length mismatch: {a.shape} vs {b.shape}")
    n = a.shape[0]
    if n < 2:  # 0 or 1 elements: the partition is trivially perfect (matches sklearn)
        return 1.0
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    contingency = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(contingency, (ai, bi), 1)

    def comb2(counts: np.ndarray) -> np.int64:
        return (counts * (counts - 1) // 2).sum()

    sum_cells = comb2(contingency)
    sum_rows = comb2(contingency.sum(axis=1))
    sum_cols = comb2(contingency.sum(axis=0))
    total = n * (n - 1) // 2
    expected = sum_rows * sum_cols / total
    maximum = (sum_rows + sum_cols) / 2
    if maximum == expected:  # both clusterings trivial (all-together or all-singletons)
        return 1.0
    return float((sum_cells - expected) / (maximum - expected))


def latent_cluster_ari(estimated: object, true: object) -> float:
    """Adjusted Rand index of the latent clustering of the observed variables.

    Clusters each observed node by its latent parent, then scores the two partitions. The two graphs
    must describe the same observed variables (same indices and labels), though their latent counts
    may differ; a mismatch raises. Returns 1.0 when there are no observed nodes.

    Examples
    --------
    >>> from andrey.core import GraphStructure, LATENT, OBSERVED
    >>> import numpy as np
    >>> # node 2 is a latent parent of observed 0 and 1
    >>> m = np.array([[0, 0, 2], [0, 0, 2], [1, 1, 0]], dtype=np.int8)
    >>> types = np.array([OBSERVED, OBSERVED, LATENT], dtype=np.int8)
    >>> g = GraphStructure.from_numpy(m, kind="dag", node_types=types)
    >>> latent_cluster_ari(g, g)
    1.0
    """
    graph_e, graph_t = _graph(estimated), _graph(true)
    clusters_e = _clusters(graph_e)
    clusters_t = _clusters(graph_t)
    observed = sorted(clusters_e)
    if set(clusters_e) != set(clusters_t):
        raise ValueError("latent graphs describe different observed variables")
    _check_observed_labels(graph_e, graph_t, observed)
    if not observed:
        return 1.0
    return adjusted_rand_index(_int_labels(clusters_e, observed), _int_labels(clusters_t, observed))


def _check_observed_labels(ge: GraphStructure, gt: GraphStructure, observed: list[int]) -> None:
    """Raise if the two graphs label a shared observed variable differently."""
    if ge.labels is None or gt.labels is None:
        return
    if any(ge.labels[i] != gt.labels[i] for i in observed):
        raise ValueError("latent graphs label observed variables differently")


def _int_labels(clusters: dict[int, object], nodes: list[int]) -> list[int]:
    """Integer partition labels for ``nodes``: one id per distinct cluster key (ARI ignores ids)."""
    ids: dict[object, int] = {}
    return [ids.setdefault(clusters[node], len(ids)) for node in nodes]


def number_of_latents(graph: object) -> int:
    """Count of nodes flagged ``LATENT`` (0 when ``node_types`` is unset).

    Examples
    --------
    >>> import numpy as np
    >>> from andrey.core import GraphStructure, LATENT, OBSERVED
    >>> m = np.array([[0, 0, 2], [0, 0, 2], [1, 1, 0]], dtype=np.int8)
    >>> types = np.array([OBSERVED, OBSERVED, LATENT], dtype=np.int8)
    >>> number_of_latents(GraphStructure.from_numpy(m, kind="dag", node_types=types))
    1
    """
    g = _graph(graph)
    return 0 if g.node_types is None else int((g.node_types == LATENT).sum())
