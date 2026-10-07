"""Private adapters from fitted temporal models (VAR, VARMA, Granger) to a ``TemporalStructure``.

Fitted models are read by attribute name. The lag coefficients are stored as ``lag_weights`` in
Andrey's orientation: ``W[i, j]`` weights ``i -> j``, the transpose of the coefficient matrix ``B``,
as in :func:`andrey.api.lingam._canonical_weights`. Each lag's :class:`GraphStructure` is derived
from that block by :mod:`andrey.api._marks`. A lagged window allows 2-cycles and self-loops, so it
is a ``digraph``, whose ``ARROW``/``ARROW`` pair means both directions, not a latent confounder; the
contemporaneous lag 0 (VAR and VARMA) is a ``dag``. Nothing here is re-exported from
``andrey.api``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from andrey.api._adapt import structure_output
from andrey.api._marks import directed_marks
from andrey.core import GraphStructure, StructureOutput, TemporalStructure


def _lag_graphs(ar_stack: np.ndarray, *, instantaneous: bool) -> tuple[GraphStructure, ...]:
    """One derived :class:`GraphStructure` per lag from an AR coefficient stack.

    ``ar_stack`` has shape ``(n_lags, n, n)`` with ``block[i, j] != 0`` meaning an edge ``j -> i``.
    When ``instantaneous`` is true ``ar_stack[0]`` is the lag-0 contemporaneous block and its view
    is a self-loop-free DAG (``kind="dag"``); every lag>=1 keeps its autoregressive self-loop
    ``X(t-k) -> X(t)`` as ``M[i][i]=ARROW`` and is tagged ``kind="digraph"`` (a directed graph
    allowing 2-cycles and self-loops), so a both-directions lag pair reads as a 2-cycle, not a PAG.
    """
    graphs: list[GraphStructure] = []
    for k in range(ar_stack.shape[0]):
        contemporaneous = instantaneous and k == 0
        allow_self_loops = not contemporaneous  # lag>=1 keeps autoregressive self-loops
        marks = directed_marks(ar_stack[k], allow_self_loops=allow_self_loops)
        kind = "dag" if contemporaneous else "digraph"
        graphs.append(
            GraphStructure.from_numpy(marks, kind=kind, allow_self_loops=allow_self_loops)
        )
    return tuple(graphs)


def _adapt_var_lingam(model: Any) -> StructureOutput:
    """VARLiNGAM fitted model -> ``StructureOutput`` over a :class:`TemporalStructure`.

    Reads ``model.adjacency_matrices_`` (shape ``(1 + p, n, n)``; index 0 instantaneous, indices
    ``1..p`` the lag-1..lag-p blocks) and ``model.causal_order_`` (a permutation). The full stack is
    stored losslessly as ``lag_weights`` and the causal order becomes the envelope ``ordering``.
    """
    ar = np.asarray(getattr(model, "adjacency_matrices_"), dtype=np.float64)
    order = getattr(model, "causal_order_")
    graphs = _lag_graphs(ar, instantaneous=True)
    # topology is built from the raw B above; store the canonical (row-is-source) B.T as weights
    temporal = TemporalStructure.from_lag_graphs(graphs, lag_weights=ar.transpose(0, 2, 1))
    return structure_output(temporal, ordering=order, metadata={"algorithm": "varlingam"})


def _adapt_varma_lingam(model: Any) -> StructureOutput:
    """VARMALiNGAM fitted model -> ``StructureOutput`` over a :class:`TemporalStructure`.

    ``model.adjacency_matrices_`` is a ``(psis, omegas)`` tuple: ``psis`` (shape ``(1 + p, n, n)``,
    index 0 instantaneous) is the autoregressive stack and ``omegas`` (shape ``(q, n, n)``, no
    instantaneous entry) is the moving-average stack. The lag topology derives from ``psis`` only;
    ``psis`` is stored as ``lag_weights`` and ``omegas`` as ``lag_weights_ma``. Ordering is the
    causal order.
    """
    psis_raw, omegas_raw = getattr(model, "adjacency_matrices_")
    psis = np.asarray(psis_raw, dtype=np.float64)
    omegas = np.asarray(omegas_raw, dtype=np.float64)
    order = getattr(model, "causal_order_")
    graphs = _lag_graphs(psis, instantaneous=True)
    temporal = TemporalStructure.from_lag_graphs(
        graphs, lag_weights=psis.transpose(0, 2, 1), lag_weights_ma=omegas.transpose(0, 2, 1)
    )
    return structure_output(temporal, ordering=order, metadata={"algorithm": "varmalingam"})


def _adapt_longitudinal_lingam(model: Any) -> StructureOutput:
    """LongitudinalLiNGAM fitted model -> ``StructureOutput`` over a :class:`TemporalStructure`.

    ``model.adjacency_matrices_`` has shape ``(T, 1 + n_lags, n, n)``: entry ``[t, 0]`` is the
    instantaneous ``B(t,t)`` and ``[t, k]`` (``k >= 1``) the lag-``k`` block, both with
    ``block[i, j] != 0`` meaning edge ``j -> i``; uncomputable entries (time point 0, or a lag-``k``
    block reaching before the panel start, ``k > t``) are ``NaN``. Andrey's native model already
    NaNs them at the source; this mask also normalizes a duck-typed producer that leaves
    wrap-around coefficients there. The full panel is stored losslessly on a
    ``(time, lag)`` axis: occasion 0 (all ``NaN``) is dropped, occasions ``1..T-1`` become the
    explicit ``times``, and their coefficient blocks ride in ``time_weights``, ``NaN`` kept.
    The lag-only surface projects the final occasion, the same ``(1 + n_lags, n, n)`` view a vector
    autoregression exposes. The per-time causal orders ride in ``metadata`` (they need not agree
    across time, so there is no single envelope ``ordering``).
    """
    adjacency = np.asarray(getattr(model, "adjacency_matrices_"), dtype=np.float64).copy()
    # A lag-k block reaching before the panel start (block k > occasion t) is uncomputable. Andrey's
    # native model already NaNs it; masking here also normalizes a duck-typed producer that
    # leaves wrap-around coefficients there (empty graph + NaN).
    for t in range(adjacency.shape[0]):
        adjacency[t, t + 1 :] = np.nan
    times = range(1, adjacency.shape[0])  # occasion 0 is all-NaN (the fit starts at t=1)
    grid = tuple(_lag_graphs(adjacency[t], instantaneous=True) for t in times)
    temporal = TemporalStructure.from_time_lag_graphs(
        grid, times=times, time_weights=adjacency[1:].transpose(0, 1, 3, 2)
    )
    metadata: dict[str, Any] = {"algorithm": "longitudinallingam"}
    orders = getattr(model, "causal_orders_", None)
    if orders is not None:
        metadata["causal_orders"] = orders
    return structure_output(temporal, metadata=metadata)


def _adapt_granger(coeff: np.ndarray, n_vars: int, maxlag: int) -> StructureOutput:
    """Granger-lasso coefficients -> ``StructureOutput`` over a :class:`TemporalStructure`.

    ``coeff`` has shape ``(d, d * maxlag)``; the lag-``k`` block (``k`` in ``1..maxlag``) is
    ``coeff[:, d * (k - 1) : d * k]`` with ``block[i, j]`` the influence ``j -> i`` at lag ``k``.
    Granger has no lag-0 (contemporaneous) block and no causal order, so every lag view is tagged
    ``kind="digraph"`` and the envelope carries no ``ordering``. The re-stacked coefficients are
    stored as ``lag_weights`` (shape ``(maxlag, d, d)``), so ``n_lags == maxlag``.
    """
    coeff = np.asarray(coeff, dtype=np.float64)
    d = int(n_vars)
    if coeff.shape != (d, d * maxlag):
        raise ValueError(
            f"granger coeff must have shape ({d}, {d * maxlag}) for n_vars={d}, "
            f"maxlag={maxlag}, got {coeff.shape}"
        )
    ar = np.stack([coeff[:, d * k : d * (k + 1)] for k in range(maxlag)])
    graphs = _lag_graphs(ar, instantaneous=False)
    temporal = TemporalStructure.from_lag_graphs(
        graphs, lags=list(range(1, maxlag + 1)), lag_weights=ar.transpose(0, 2, 1)
    )
    return structure_output(temporal, metadata={"algorithm": "granger"})
