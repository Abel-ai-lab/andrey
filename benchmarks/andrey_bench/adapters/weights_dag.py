"""The DAG of a thresholded weight matrix, how a continuous-optimization reference reports a graph.

NOTEARS and DAGMA return a weight matrix with the small weights set to zero. Its nonzero entries
are the graph, and nothing in either code makes them acyclic. :func:`dag_of_weights` drops the
weakest edges, one at a time, until no cycle is left, the repair Andrey's own fits apply. NumPy
only, so it imports in every solution's environment.
"""

from __future__ import annotations

import numpy as np


def has_cycle(adjacency: np.ndarray) -> bool:
    """Whether the directed graph ``adjacency[i, j] != 0`` for ``i -> j`` has a cycle."""
    left = np.asarray(adjacency) != 0
    alive = np.ones(len(left), dtype=bool)
    while alive.any():
        roots = alive & ~left[alive].any(axis=0)
        if not roots.any():
            return True
        alive &= ~roots
    return False


def dag_of_weights(weights: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the 0/1 adjacency of `weights` made acyclic, and the number of edges dropped.

    ``weights[i, j]`` is the coefficient of ``i -> j``. Edges leave in the order of their weight
    magnitude, weakest first, until the graph has no cycle.
    """
    w = np.asarray(weights, dtype=np.float64)
    adjacency = (w != 0).astype(np.int8)
    dropped = 0
    for flat in np.argsort(np.abs(w), axis=None, kind="stable"):
        i, j = divmod(int(flat), w.shape[1])
        if not adjacency[i, j]:
            continue
        if not has_cycle(adjacency):
            break
        adjacency[i, j] = 0
        dropped += 1
    return adjacency, dropped
