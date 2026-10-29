"""Private adapters from the pairwise and latent-confounder engines to a ``StructureOutput``.

Fitted models are read by attribute name.

* **ANM, PNL** (``(pval_forward, pval_backward, stat_forward, stat_backward)``): a two-node
  ``dag`` with the edge :func:`_direction` picks, or none; the p-values and statistics go in
  metadata as ``float``.
* **RCD** (``adjacency_matrix_``): a finite nonzero ``B[i, j]`` is ``j -> i`` and a ``NaN`` cell
  is ``<->``, in a ``pag``; ``weighted_adjacency`` holds ``B`` with ``NaN`` as ``0``.
* **BottomUpParceLiNGAM** (``adjacency_matrix_``, ``causal_order_``): a ``NaN`` pair is
  order-unknown, ``o-o`` in a ``pag``. The causal order nests order-unknown variables as
  sub-lists, so it goes in metadata rather than ``ordering``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from andrey.core import ARROW, CIRCLE, TAIL, GraphStructure, StructureOutput

from ._adapt import structure_output
from ._marks import marks_from_signed

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


def _nan_pairs(B: np.ndarray) -> list[list[int]]:
    """Return the sorted ``[i, j]`` pairs, ``i < j``, whose cell is ``NaN`` in either direction."""
    rows, cols = np.nonzero(np.isnan(B) | np.isnan(B.T))
    return [[int(i), int(j)] for i, j in zip(rows, cols, strict=True) if i < j]


def _direction(
    pval_forward: float,
    pval_backward: float,
    alpha: float | None,
    *,
    stat_forward: float | None = None,
    stat_backward: float | None = None,
) -> int:
    """Return ``1`` for ``x -> y``, ``-1`` for ``y -> x``, or ``0`` for no edge.

    With ``alpha=None``, choose the direction with the larger p-value. Equal p-values, given both
    test statistics, choose the direction with the smaller statistic, the weaker dependence; equal
    or missing statistics give no edge. Otherwise, choose a direction only when its p-value exceeds
    ``alpha`` and the other does not.
    """
    if alpha is None:
        forward, backward = pval_forward > pval_backward, pval_backward > pval_forward
        if pval_forward == pval_backward and stat_forward is not None and stat_backward is not None:
            forward, backward = stat_forward < stat_backward, stat_backward < stat_forward
    else:
        forward, backward = pval_forward > alpha, pval_backward > alpha
    return 1 if forward and not backward else -1 if backward and not forward else 0


def _pairwise_output(
    pval_forward: Any,
    pval_backward: Any,
    *,
    stat_forward: Any = None,
    stat_backward: Any = None,
    alpha: float | None = None,
    labels: tuple[str, ...] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> StructureOutput:
    """Return a two-node ``dag`` with the chosen edge and ``float`` p-value metadata.

    ``metadata["p_values"]`` maps ``"forward"`` (``x -> y``) and ``"backward"`` to the p-values.
    Test statistics, when given, decide a p-value tie and go in ``metadata["statistics"]`` under
    the same keys; ``extra`` adds its entries after them.
    """
    forward, backward = _coerce_float(pval_forward), _coerce_float(pval_backward)
    stats: dict[str, float] = {}
    if stat_forward is not None and stat_backward is not None:
        stats = {
            "stat_forward": _coerce_float(stat_forward),
            "stat_backward": _coerce_float(stat_backward),
        }
    M = np.zeros((2, 2), dtype=np.int8)
    direction = _direction(forward, backward, alpha, **stats)
    if direction:
        source, target = (0, 1) if direction > 0 else (1, 0)
        M[source, target], M[target, source] = TAIL, ARROW
    return structure_output(
        GraphStructure.from_numpy(M, kind="dag"),
        labels=labels,
        metadata={
            "p_values": {"forward": forward, "backward": backward},
            **(
                {
                    "statistics": {
                        "forward": stats["stat_forward"],
                        "backward": stats["stat_backward"],
                    }
                }
                if stats
                else {}
            ),
            **(extra or {}),
        },
    )


def _adapt_anm(
    pval_forward: Any,
    pval_backward: Any,
    *,
    stat_forward: Any = None,
    stat_backward: Any = None,
    alpha: float | None = None,
    labels: tuple[str, ...] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> StructureOutput:
    """Convert ANM p-values, and the statistics when given, to a two-node ``dag`` and metadata.

    Each statistic is an HSIC statistic divided by its mean under independence. With
    ``alpha=None``, the statistics decide a p-value tie: the smaller one wins. Without them, a tie
    gives no edge. ``extra`` adds metadata entries, such as ``rows_tested``.
    """
    return _pairwise_output(
        pval_forward,
        pval_backward,
        stat_forward=stat_forward,
        stat_backward=stat_backward,
        alpha=alpha,
        labels=labels,
        extra=extra,
    )


def _adapt_pnl(
    pval_forward: Any,
    pval_backward: Any,
    *,
    stat_forward: Any = None,
    stat_backward: Any = None,
    alpha: float | None = None,
    labels: tuple[str, ...] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> StructureOutput:
    """Convert PNL p-values, and the statistics when given, to a two-node ``dag`` and metadata.

    Each statistic is an HSIC statistic divided by its mean under independence. With
    ``alpha=None``, the statistics decide a p-value tie: the smaller one wins. Without them, a tie
    gives no edge. A p-value or statistic may be a shape-``(1,)`` array; :func:`_coerce_float`
    converts it to a Python ``float``. ``extra`` adds metadata entries, such as ``rows_tested``.
    """
    return _pairwise_output(
        pval_forward,
        pval_backward,
        stat_forward=stat_forward,
        stat_backward=stat_backward,
        alpha=alpha,
        labels=labels,
        extra=extra,
    )


# ---- RCD (latent-confounder adjacency) ----------------------------------------------------------


def _adapt_rcd(model: Any) -> StructureOutput:
    """RCD ``adjacency_matrix_`` -> directed edges + bidirected ``<->`` for ``NaN`` cells.

    Finite nonzero ``B[i, j]`` is edge ``j -> i``; a ``NaN`` cell marks a latent confounder and
    becomes a bidirected ``pag`` edge. ``weighted_adjacency`` is the canonical ``B.T`` with ``NaN``
    replaced by ``0.0``, so ``weighted_adjacency[i, j] != 0`` iff there is an arrowhead into ``j``
    (one orientation with the marks -- the frozen contract). The ``NaN`` pairs, as sorted
    ``[i, j]`` lists, ride in ``metadata["confounded_pairs"]``.
    """
    B = _adjacency(model)
    M = marks_from_signed(B, nan_mark="bidirected")
    structure = GraphStructure.from_numpy(M, kind="pag")
    return structure_output(
        structure,
        weighted_adjacency=_nan_to_zero(B).T,
        metadata={"confounded_pairs": _nan_pairs(B)},
    )


# ---- BottomUpParceLiNGAM (order-unknown adjacency) ----------------------------------------------


def _adapt_bottom_up_parce(
    model: Any, arrowheads: np.ndarray | None = None, *, required: np.ndarray | None = None
) -> StructureOutput:
    """BottomUpParceLiNGAM adjacency -> directed edges + circle ``o-o`` at ``NaN`` cells.

    A symmetric ``NaN`` block is an order-unknown pair and becomes a circle ``pag`` edge. The
    ``causal_order_`` is nested (order-unknown vars nest as sub-lists) so it cannot be a flat
    ``ordering`` permutation -- it rides in ``metadata["partial_order"]`` as a JSON-safe nested
    list.
    ``weighted_adjacency`` is the canonical ``B.T`` with ``NaN`` replaced by ``0.0``. The unordered
    variables form one block, which also holds the confounded variables' observed ancestors, so an
    ``o-o`` pair need not share a hidden cause, and no ``confounded_pairs`` are listed.
    ``arrowheads[i, j]``, from background knowledge, puts an arrowhead at ``i`` on an ``o-o``
    pair ``i - j``, and ``required[i, j]`` makes ``i -> j`` an edge even where its weight is ``0``.
    """
    B = _adjacency(model)
    M = marks_from_signed(B, nan_mark="circle")
    if arrowheads is not None:
        M[arrowheads & (M == CIRCLE)] = ARROW
    if required is not None:
        M[required], M[required.T] = TAIL, ARROW
    structure = GraphStructure.from_numpy(M, kind="pag")
    metadata: dict[str, Any] = {}
    order = getattr(model, "causal_order_", None)
    if order is not None:
        metadata["partial_order"] = order  # json_safe (via structure_output) coerces the nesting
    return structure_output(structure, weighted_adjacency=_nan_to_zero(B).T, metadata=metadata)
