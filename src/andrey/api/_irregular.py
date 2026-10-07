"""Private adapters from the pairwise and latent-confounder engines to a ``StructureOutput``.

Fitted models are read by attribute name.

* **ANM, PNL** (``(pval_forward, pval_backward)``): a two-node ``dag`` with the edge
  :func:`_direction` picks, or none; the p-values go in metadata as ``float``.
* **CAM-UV** (``(P, U)``): ``P[i]`` holds the parents of ``i``; ``U`` holds pairs sharing an
  unobserved cause. A ``pag`` with the directed edges and a ``<->`` per pair; the pairs also go
  in metadata.
* **RCD** (``adjacency_matrix_``): a finite nonzero ``B[i, j]`` is ``j -> i`` and a ``NaN`` cell
  is ``<->``, in a ``pag``; ``weighted_adjacency`` holds ``B`` with ``NaN`` as ``0``.
* **BottomUpParceLiNGAM** (``adjacency_matrix_``, ``causal_order_``): a ``NaN`` pair is
  order-unknown, ``o-o`` in a ``pag``. The causal order nests order-unknown variables as
  sub-lists, so it goes in metadata rather than ``ordering``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from andrey.core import ARROW, TAIL, GraphStructure, StructureOutput

from ._adapt import structure_output
from ._marks import directed_marks, marks_from_signed, set_bidirected

# ---- shared helpers -----------------------------------------------------------------------------


def _coerce_float(value: Any) -> float:
    """Coerce a p-value to a plain ``float`` (PNL returns each as a shape-``(1,)`` array).

    ``_check_json_safe`` rejects numpy scalars / arrays, so metadata must hold Python ``float``.
    """
    return float(np.asarray(value, dtype=np.float64).ravel()[0])


def _adjacency(model: Any) -> np.ndarray:
    """Return the fitted ``adjacency_matrix_`` as a ``float64`` array (duck-typed; ``NaN`` kept)."""
    B = getattr(model, "adjacency_matrix_", None)
    if B is None:
        raise ValueError("model has no adjacency_matrix_ attribute")
    return np.asarray(B, dtype=np.float64)


def _nan_to_zero(B: np.ndarray) -> np.ndarray:
    """Return a copy of ``B`` with ``NaN`` sentinels replaced by ``0.0`` (finite signs kept)."""
    W = np.array(B, dtype=np.float64)
    W[np.isnan(W)] = 0.0
    return W


# ---- ANM / PNL (pairwise direction test) --------------------------------------------------------


def _direction(pval_forward: float, pval_backward: float, alpha: float | None) -> int:
    """Return ``1`` for ``x -> y``, ``-1`` for ``y -> x``, or ``0`` for no edge.

    With ``alpha=None``, choose the direction with the larger p-value; a tie gives no edge.
    Otherwise, choose a direction only when its p-value exceeds ``alpha`` and the other does not.
    """
    if alpha is None:
        forward, backward = pval_forward > pval_backward, pval_backward > pval_forward
    else:
        forward, backward = pval_forward > alpha, pval_backward > alpha
    return 1 if forward and not backward else -1 if backward and not forward else 0


def _pairwise_output(
    pval_forward: Any,
    pval_backward: Any,
    *,
    algorithm: str,
    alpha: float | None = None,
    labels: tuple[str, ...] | None = None,
) -> StructureOutput:
    """Return a two-node ``dag`` with the chosen edge and ``float`` p-value metadata."""
    forward, backward = _coerce_float(pval_forward), _coerce_float(pval_backward)
    M = np.zeros((2, 2), dtype=np.int8)
    direction = _direction(forward, backward, alpha)
    if direction:
        source, target = (0, 1) if direction > 0 else (1, 0)
        M[source, target], M[target, source] = TAIL, ARROW
    return structure_output(
        GraphStructure.from_numpy(M, kind="dag"),
        labels=labels,
        metadata={"algorithm": algorithm, "pval_forward": forward, "pval_backward": backward},
    )


def _adapt_anm(
    pval_forward: Any, pval_backward: Any, *, alpha: float | None = None
) -> StructureOutput:
    """Convert ANM p-values to a two-node ``dag`` and metadata."""
    return _pairwise_output(pval_forward, pval_backward, algorithm="ANM", alpha=alpha)


def _adapt_pnl(
    pval_forward: Any,
    pval_backward: Any,
    *,
    alpha: float | None = None,
    labels: tuple[str, ...] | None = None,
) -> StructureOutput:
    """Convert PNL p-values to a two-node ``dag`` and metadata.

    PNL returns each p-value as a shape-``(1,)`` array; :func:`_coerce_float` converts it to
    a Python ``float``.
    """
    return _pairwise_output(
        pval_forward, pval_backward, algorithm="PNL", alpha=alpha, labels=labels
    )


# ---- CAMUV (parents + confounded pairs) ---------------------------------------------------------


def _adapt_camuv(P: Any, U: Any, n_vars: int) -> StructureOutput:
    """CAMUV ``execute`` -> directed edges from ``P`` + bidirected ``<->`` from ``U`` in a ``pag``.

    ``P[i]`` is the parent set of variable ``i`` (edge ``j -> i`` for ``j in P[i]``); ``U`` is a
    list of index sets sharing an unobserved confounder. The normalized confounded pairs (sorted
    ``[i, j]`` lists) ride in ``metadata["confounded_pairs"]``. No ``weighted_adjacency``.
    """
    B = np.zeros((n_vars, n_vars), dtype=np.float64)
    items = P.items() if isinstance(P, dict) else enumerate(P)
    for i, parents in items:
        for j in parents:
            B[int(i), int(j)] = 1.0  # B[i, j] != 0 => edge j -> i
    M = directed_marks(B)
    confounded_pairs: list[list[int]] = []
    for group in U:
        idx = sorted(int(x) for x in group)
        confounded_pairs.append(idx)
        for a, b in combinations(idx, 2):
            set_bidirected(M, a, b)
    structure = GraphStructure.from_numpy(M, kind="pag")
    return structure_output(structure, metadata={"confounded_pairs": confounded_pairs})


# ---- RCD (latent-confounder adjacency) ----------------------------------------------------------


def _adapt_rcd(model: Any) -> StructureOutput:
    """RCD ``adjacency_matrix_`` -> directed edges + bidirected ``<->`` for ``NaN`` cells.

    Finite nonzero ``B[i, j]`` is edge ``j -> i``; a ``NaN`` cell marks a latent confounder and
    becomes a bidirected ``pag`` edge. ``weighted_adjacency`` is the canonical ``B.T`` with ``NaN``
    replaced by ``0.0``, so ``weighted_adjacency[i, j] != 0`` iff there is an arrowhead into ``j``
    (one orientation with the marks -- the frozen contract).
    """
    B = _adjacency(model)
    M = marks_from_signed(B, nan_mark="bidirected")
    structure = GraphStructure.from_numpy(M, kind="pag")
    return structure_output(structure, weighted_adjacency=_nan_to_zero(B).T)


# ---- BottomUpParceLiNGAM (order-unknown adjacency) ----------------------------------------------


def _adapt_bottom_up_parce(model: Any) -> StructureOutput:
    """BottomUpParceLiNGAM adjacency -> directed edges + circle ``o-o`` at ``NaN`` cells.

    A symmetric ``NaN`` block is an order-unknown pair and becomes a circle ``pag`` edge. The
    ``causal_order_`` is nested (order-unknown vars nest as sub-lists) so it cannot be a flat
    ``ordering`` permutation -- it rides in ``metadata["causal_order"]`` as a JSON-safe nested list.
    ``weighted_adjacency`` is the canonical ``B.T`` with ``NaN`` replaced by ``0.0``.
    """
    B = _adjacency(model)
    M = marks_from_signed(B, nan_mark="circle")
    structure = GraphStructure.from_numpy(M, kind="pag")
    metadata: dict[str, Any] = {}
    order = getattr(model, "causal_order_", None)
    if order is not None:
        metadata["causal_order"] = order  # json_safe (via structure_output) coerces the nesting
    return structure_output(
        structure, weighted_adjacency=_nan_to_zero(B).T, metadata=metadata or None
    )
