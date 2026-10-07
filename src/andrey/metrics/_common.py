"""Shared helpers for :mod:`andrey.metrics`: input normalization and endpoint-mark arithmetic.

Every graph metric operates on the dense endpoint-mark matrix ``M`` -- unsigned codes
(``NULL``/``TAIL``/``ARROW``/``CIRCLE``), where ``M[i, j]`` is the mark at node ``i`` on edge
``i-j`` (a directed edge ``i -> j`` is ``M[i, j] == TAIL`` and ``M[j, i] == ARROW``). These helpers
turn any accepted input into that matrix and carry the precision/recall/f1 arithmetic the metrics
share.

Accepted graph inputs, all reduced to ``(M, kind)``:

- a :class:`~andrey.core.GraphStructure`, or a :class:`~andrey.core.StructureOutput` wrapping one;
- a raw directed adjacency ``ndarray`` -- a nonzero ``A[i, j]`` reads as a directed edge ``i -> j``,
  a symmetric nonzero pair as one undirected edge. To avoid a silent misread, an *integer* array
  carrying a ``2`` or ``3`` (the ``ARROW`` / ``CIRCLE`` codes) is rejected: pass endpoint marks
  through ``GraphStructure.from_numpy(..., kind=...)``, not as a bare array.
"""

from __future__ import annotations

import numpy as np

from andrey.core.output import StructureOutput
from andrey.core.structure import ARROW, TAIL, GraphStructure, Kind, Structure, _directed_support

__all__ = ["as_marks", "two_marks", "prf", "directed_adjacency"]


def as_marks(graph: object) -> tuple[np.ndarray, Kind]:
    """Return ``(M, kind)`` -- the ``(n, n)`` int8 endpoint-mark matrix and kind of ``graph``.

    Raises ``TypeError`` for a temporal structure (use :func:`~andrey.metrics.temporal_scores`).

    Examples
    --------
    >>> import numpy as np
    >>> _, kind = as_marks(np.array([[0, 1], [0, 0]]))  # 0 -> 1
    >>> kind
    'dag'
    """
    if isinstance(graph, StructureOutput):
        graph = graph.structure
    if isinstance(graph, GraphStructure):
        return graph.to_numpy(dtype=np.int8), graph.kind
    if isinstance(graph, Structure):  # TemporalStructure
        raise TypeError(
            "graph metrics need a GraphStructure; use temporal_scores for temporal output"
        )
    return _marks_from_adjacency(np.asarray(graph))


def _marks_from_adjacency(a: np.ndarray) -> tuple[np.ndarray, Kind]:
    """Endpoint marks + kind from a raw directed adjacency (nonzero ``A[i, j]`` is edge i -> j)."""
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"expected a square adjacency matrix, got shape {a.shape}")
    if np.issubdtype(a.dtype, np.integer) and np.isin(a, (ARROW, 3)).any():
        raise ValueError(
            "adjacency carries endpoint-mark codes (2=ARROW, 3=CIRCLE); pass marks through "
            "GraphStructure.from_numpy(..., kind=...) rather than as a raw array"
        )
    n = a.shape[0]
    present = a != 0
    np.fill_diagonal(present, False)  # a static adjacency carries no self-loops
    directed = present & ~present.T  # one-way support -> a directed edge
    undirected = present & present.T  # mutual support -> one undirected edge
    m = np.zeros((n, n), dtype=np.int8)
    fi, fj = np.nonzero(directed)
    m[fi, fj] = TAIL
    m[fj, fi] = ARROW
    m[undirected] = TAIL
    return m, ("cpdag" if undirected.any() else "dag")


def _labels(x: object) -> tuple[str, ...] | None:
    """Node labels of an input, or ``None`` for a raw array / unlabeled structure."""
    if isinstance(x, StructureOutput):
        x = x.structure
    return x.labels if isinstance(x, Structure) else None


def two_marks(estimated: object, true: object) -> tuple[np.ndarray, Kind, np.ndarray, Kind]:
    """Normalize a pair of graphs to ``(M_est, kind_est, M_true, kind_true)``.

    Fails loudly on a node-count mismatch, and on differing labels when both inputs carry them
    (a positional comparison across differently-ordered variables is a silent wrong-number bug).
    """
    m_est, k_est = as_marks(estimated)
    m_true, k_true = as_marks(true)
    if m_est.shape != m_true.shape:
        raise ValueError(
            f"node-count mismatch: estimated is {m_est.shape[0]}, true is {m_true.shape[0]}"
        )
    lab_est, lab_true = _labels(estimated), _labels(true)
    if lab_est is not None and lab_true is not None and lab_est != lab_true:
        raise ValueError("label mismatch: estimated and true index different variables")
    return m_est, k_est, m_true, k_true


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision/recall/f1 from counts; a zero denominator scores 1.0 (vacuously satisfied).

    So a perfect match and the empty-vs-empty case both give ``1.0`` (predicting nothing against
    nothing is precision and recall 1.0); ``f1`` is ``0.0`` only when both are ``0.0``.
    """
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _offdiag(n: int) -> np.ndarray:
    """Boolean ``(n, n)`` mask that is ``True`` off the diagonal."""
    return ~np.eye(n, dtype=bool)


def directed_adjacency(m: np.ndarray) -> np.ndarray:
    """Boolean ``(n, n)`` directed support of a mark matrix: ``D[i, j]`` is the edge ``i -> j``.

    A plain directed edge (arrowhead at ``j``) sets ``D[i, j]``; a bidirected / 2-cycle pair sets
    both directions; an autoregressive self-loop (``m[i, i] == ARROW``) sets ``D[i, i]``. Undirected
    and circle marks carry no direction and are ignored. Delegates to the core projection so the
    temporal-scoring collapse and ``TemporalStructure.summary_graph()`` share one definition.
    """
    return _directed_support(m)
