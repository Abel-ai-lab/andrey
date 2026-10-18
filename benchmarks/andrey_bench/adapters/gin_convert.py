"""The native result both GIN adapters return, and the latent graph built from it.

A GIN result is a set of clusters of observed variables, one per latent, with an order of the
latents from the earliest. Both adapters return it as a ``(2, n_observed)`` int array: row 0 gives
each observed variable's cluster, ``-1`` when it is in none, and row 1 that cluster's position in
the latent order, ``-1`` when the cluster was left unordered. Clusters are numbered in a canonical
way, ordered ones by position and then unordered ones by their smallest member, so two packages
with the same clusters and order return the same array, and :func:`latent_dag` builds the same
graph from it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def encode_clusters(
    ordered: Sequence[Sequence[int]], unordered: Sequence[Sequence[int]], n_observed: int
) -> np.ndarray:
    """Return the native ``(2, n_observed)`` array of `ordered` clusters and `unordered` ones."""
    native = np.full((2, n_observed), -1, dtype=np.int64)
    clusters = [list(c) for c in ordered] + sorted((list(c) for c in unordered), key=min)
    for number, members in enumerate(clusters):
        native[0, members] = number
        if number < len(ordered):
            native[1, members] = number
    return native


def latent_dag(native: np.ndarray):
    """Return the DAG over the observed variables and one latent per cluster of `native`.

    The latents follow the observed variables, numbered as their clusters are, and are flagged
    ``LATENT``. Each points into its cluster's members, and an ordered latent points into every
    latent after it and into every unordered one, the graph Andrey's ``gin`` returns. Unordered
    latents have no edge between them.
    """
    from andrey import GraphStructure
    from andrey.core import LATENT, OBSERVED

    cluster, position = np.asarray(native)
    n_observed = len(cluster)
    m = int(cluster.max()) + 1 if n_observed and cluster.max() >= 0 else 0
    rank = np.full(m, -1)
    for c in range(m):
        rank[c] = position[cluster == c][0]
    marks = np.zeros((n_observed + m,) * 2, dtype=np.int8)

    def edge(a: int, b: int) -> None:
        marks[a, b], marks[b, a] = 1, 2  # TAIL at the cause, ARROW at the effect

    for i in np.flatnonzero(cluster >= 0):
        edge(n_observed + int(cluster[i]), int(i))
    for a in range(m):
        for b in range(m):
            if rank[a] >= 0 and (rank[b] < 0 or rank[a] < rank[b]):
                edge(n_observed + a, n_observed + b)
    types = np.array([OBSERVED] * n_observed + [LATENT] * m, dtype=np.int8)
    return GraphStructure.from_numpy(marks, kind="dag", node_types=types)
