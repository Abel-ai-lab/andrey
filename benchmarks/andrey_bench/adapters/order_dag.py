"""The full DAG of a causal order, how an order-only solution reports the order it found.

An order-based method (SCORE, NoGAM, CAM, R2-SortnRegress) finds a causal order, then prunes the
DAG that joins each variable to every later one. Its order-only solution skips the pruning and
returns that full DAG, so the harness scores the order with the graph metrics it already has: the
full DAG's arrowhead recall against the truth is the share of true edges the order puts in the
right direction, one minus the order divergence ``D_top`` of Rolland et al. (2022) over the true
edge count. NumPy only, so it imports in every solution's environment.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def order_dag(order: Sequence[int]) -> np.ndarray:
    """The 0/1 adjacency with ``a[i, j] = 1`` whenever `order` lists ``i`` before ``j``."""
    order = [int(v) for v in order]
    adjacency = np.zeros((len(order), len(order)), dtype=np.int8)
    for k, v in enumerate(order):
        adjacency[v, order[k + 1 :]] = 1
    return adjacency
