"""Causal-order metrics: how well a discovered ordering respects the true ancestral relation.

A permutation method (DirectLiNGAM and kin) emits a causal order -- upstream causes first. These
score that order against a true DAG's ancestral pairs, independent of the edge estimates.
"""

from __future__ import annotations

import numpy as np

from andrey.core.output import StructureOutput

from ._common import as_marks, directed_adjacency


def _positions(order: object, n: int) -> np.ndarray:
    """Rank of each node in ``order`` (``pos[node]`` = its slot); accepts a StructureOutput."""
    if isinstance(order, StructureOutput):
        if order.ordering is None:
            raise ValueError("StructureOutput carries no ordering")
        order = order.ordering
    seq = np.asarray(order, dtype=np.intp)
    if seq.shape != (n,) or sorted(seq.tolist()) != list(range(n)):
        raise ValueError(f"order must be a permutation of range({n}), got {seq.tolist()}")
    pos = np.empty(n, dtype=np.intp)
    pos[seq] = np.arange(n)
    return pos


def _ancestor_pairs(true_dag: object) -> tuple[np.ndarray, int]:
    """``(pairs, n)`` where ``pairs`` are the ``(a, b)`` with ``a`` a strict ancestor of ``b``."""
    marks, _ = as_marks(true_dag)
    n = marks.shape[0]
    reach = directed_adjacency(marks)
    np.fill_diagonal(reach, False)
    while True:  # transitive closure by boolean reachability (fixed point in <= n steps)
        step = reach | ((reach.astype(np.int64) @ reach.astype(np.int64)) > 0)
        np.fill_diagonal(step, False)
        if np.array_equal(step, reach):
            break
        reach = step
    return np.argwhere(reach), n


def causal_order_scores(order: object, true_dag: object) -> dict[str, float]:
    """Accuracy, Kendall tau, and comparable-pair count of an order against a true DAG.

    ``accuracy`` is the fraction of true ancestral pairs the order sequences correctly;
    ``kendall_tau`` is ``2 * accuracy - 1`` over those comparable pairs; ``n_comparable_pairs`` is
    how many ancestral pairs exist (so tau and accuracy carry distinct information). A valid
    topological order scores accuracy 1.0; an order with no comparable pairs scores 1.0 / tau 1.0.

    Examples
    --------
    >>> causal_order_scores([0, 1, 2], [[0, 1, 0], [0, 0, 1], [0, 0, 0]])["accuracy"]
    1.0
    """
    pairs, n = _ancestor_pairs(true_dag)
    count = int(pairs.shape[0])
    if count == 0:
        return {"accuracy": 1.0, "kendall_tau": 1.0, "n_comparable_pairs": 0}
    pos = _positions(order, n)
    correct = int((pos[pairs[:, 0]] < pos[pairs[:, 1]]).sum())
    accuracy = correct / count
    return {
        "accuracy": accuracy,
        "kendall_tau": 2 * accuracy - 1,
        "n_comparable_pairs": count,
    }


def causal_order_accuracy(order: object, true_dag: object) -> float:
    """Fraction of true ancestral pairs the order places in the right sequence (1.0 is perfect).

    Examples
    --------
    >>> causal_order_accuracy([0, 1, 2], [[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # 0->1->2
    1.0
    """
    return causal_order_scores(order, true_dag)["accuracy"]


def causal_order_kendall_tau(order: object, true_dag: object) -> float:
    """Kendall tau over comparable (ancestral) pairs: ``(concordant - discordant) / comparable``.

    Ranges from ``-1`` (every ancestral pair reversed) to ``1``; equals ``2 * accuracy - 1``.

    Examples
    --------
    >>> causal_order_kendall_tau([0, 1, 2], [[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    1.0
    """
    return causal_order_scores(order, true_dag)["kendall_tau"]
