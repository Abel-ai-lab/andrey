"""Causal-order metrics: how well a discovered ordering respects the true ancestral relation.

A permutation method (DirectLiNGAM and kin) emits a causal order -- upstream causes first. These
score that order against a true DAG's ancestral pairs, independent of the edge estimates.
"""

from __future__ import annotations

import numpy as np

from andrey.core.output import StructureOutput
from andrey.core.structure import directed_support

from ._common import as_marks


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
    reach = directed_support(marks)
    np.fill_diagonal(reach, False)
    while True:  # transitive closure by boolean reachability (fixed point in <= n steps)
        step = reach | ((reach.astype(np.int64) @ reach.astype(np.int64)) > 0)
        np.fill_diagonal(step, False)
        if np.array_equal(step, reach):
            break
        reach = step
    return np.argwhere(reach), n


def causal_order_scores(order: object, true_dag: object) -> dict[str, float]:
    """Return the accuracy, Kendall tau, and comparable-pair count of an order against a true DAG.

    Parameters
    ----------
    order : StructureOutput or array-like of int, shape (n,)
        The causal order, causes first: a permutation of ``range(n)``, or a result whose
        ``ordering`` holds one.
    true_dag : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The true DAG. A raw array is a directed adjacency: a nonzero ``A[i, j]`` is the edge
        ``i -> j``.

    Returns
    -------
    dict
        ``"accuracy"``, the fraction of the true ancestral pairs the order sequences correctly;
        ``"kendall_tau"``, ``2 * accuracy - 1`` over those pairs; and ``"n_comparable_pairs"``,
        the number of ancestral pairs, which tells how much the other two rest on. A topological
        order of the true DAG scores 1.0, and so does any order when the DAG has no edge.

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
    """Return the fraction of the true ancestral pairs the order places in the right sequence.

    Parameters
    ----------
    order : StructureOutput or array-like of int, shape (n,)
        The causal order, causes first: a permutation of ``range(n)``, or a result whose
        ``ordering`` holds one.
    true_dag : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The true DAG. A raw array is a directed adjacency: a nonzero ``A[i, j]`` is the edge
        ``i -> j``.

    Returns
    -------
    float
        The fraction, 1.0 for a topological order of the true DAG.

    Examples
    --------
    >>> causal_order_accuracy([0, 1, 2], [[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # 0->1->2
    1.0
    """
    return causal_order_scores(order, true_dag)["accuracy"]


def causal_order_kendall_tau(order: object, true_dag: object) -> float:
    """Return the Kendall tau of an order over the true DAG's ancestral pairs.

    Parameters
    ----------
    order : StructureOutput or array-like of int, shape (n,)
        The causal order, causes first: a permutation of ``range(n)``, or a result whose
        ``ordering`` holds one.
    true_dag : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The true DAG. A raw array is a directed adjacency: a nonzero ``A[i, j]`` is the edge
        ``i -> j``.

    Returns
    -------
    float
        ``(concordant - discordant) / comparable`` over the ancestral pairs, from ``-1``, every
        pair reversed, to ``1``; it equals ``2 * accuracy - 1``.

    Examples
    --------
    >>> causal_order_kendall_tau([0, 1, 2], [[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    1.0
    """
    return causal_order_scores(order, true_dag)["kendall_tau"]
