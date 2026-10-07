"""Temporal (VAR / VARMA) data generation and ``TemporalStructure`` truth.

Unlike the sparse static DGP, temporal truth is a **dense** ``(n_lags, d, d)`` weight stack
(Andrey's ``TemporalStructure``) and the instantaneous reduced form needs a dense ``(I - B0)^-1``.
So temporal generation is a modest-``d`` subsystem -- gated at :data:`MAX_TEMPORAL_D` rather than
scaling to the tens of thousands the static generators reach.

The process is ``x_t = B0 x_t + sum_k A_k x_{t-k} + e_t (+ sum_q M_q e_{t-q})``: instantaneous
lower-triangular LiNGAM mixing ``B0`` (the sampled DAG), autoregressive matrices ``A_k``, and for
VARMA a moving-average stack ``M_q``. Lag matrices are scaled so the reduced-form spectral
radius is below one (stationary); a burn-in prefix is discarded.
"""

from __future__ import annotations

import numpy as np

from andrey.core.structure import ARROW, NULL, TAIL, GraphStructure, TemporalStructure

from .graphs import DAGDraw, dag_truth

MAX_TEMPORAL_D = 1000
_STABLE_TARGET = 0.6  # scale lag matrices so the reduced-form spectral radius sits at this


def check_size(d: int) -> None:
    """Reject a temporal request whose node count exceeds the dense-truth budget."""
    if d > MAX_TEMPORAL_D:
        raise ValueError(
            f"temporal DGP is a modest-d subsystem (dense (n_lags, d, d) weight stacks and a dense "
            f"(I-B0)^-1); d={d} exceeds MAX_TEMPORAL_D={MAX_TEMPORAL_D}. Use the static generators "
            f"for large sparse graphs."
        )


def build_b0(draw: DAGDraw, weights: np.ndarray) -> np.ndarray:
    """Assemble the dense instantaneous mixing ``B0[effect, cause] = weight`` from the DAG."""
    b0 = np.zeros((draw.d, draw.d), dtype=np.float64)
    if draw.parents.shape[0]:
        b0[draw.children, draw.parents] = weights  # edge cause -> effect
    return b0


def draw_lag_stack(
    d: int, n_lags: int, density: float, weight_range: tuple[float, float], rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n_lags`` dense random lag matrices ``(n_lags, d, d)``, each with a full diagonal.

    The diagonal (a node's own lagged value) is always present so every series has memory rather
    than reducing to white noise; off-diagonal entries appear with probability ``density``.
    """
    lo, hi = weight_range
    stack = np.zeros((n_lags, d, d), dtype=np.float64)
    for k in range(n_lags):
        mask = rng.random((d, d)) < density
        np.fill_diagonal(mask, True)  # every node keeps its own lagged (self-AR) term
        vals = rng.uniform(lo, hi, size=(d, d)) * rng.choice(np.array([-1.0, 1.0]), size=(d, d))
        stack[k] = mask * vals
    return stack


def stabilize(b0: np.ndarray, a_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale the lag stack for a stationary reduced-form VAR; return ``(a_stack, (I-B0)^-1)``.

    Scaling lag ``k`` (1-based) by ``s**k`` maps every companion eigenvalue ``λ`` to ``sλ``, so a
    single geometric rescale lands the reduced-form spectral radius on the target (exact for any lag
    order, unlike a uniform scale which only converges as ``s**(1/p)``).
    """
    ib_inv = np.linalg.inv(np.eye(b0.shape[0]) - b0)
    rho = _spectral_radius(ib_inv, a_stack)
    if rho > _STABLE_TARGET:
        s = _STABLE_TARGET / rho
        a_stack = a_stack * (s ** np.arange(1, a_stack.shape[0] + 1))[:, None, None]
    return a_stack, ib_inv


def _spectral_radius(ib_inv: np.ndarray, a_stack: np.ndarray) -> float:
    """Spectral radius of the reduced-form VAR companion matrix ``[(I-B0)^-1 A_k]``."""
    d, p = ib_inv.shape[0], a_stack.shape[0]
    reduced = np.concatenate([ib_inv @ a_stack[k] for k in range(p)], axis=1)  # (d, p*d)
    companion = np.zeros((p * d, p * d), dtype=np.float64)
    companion[:d] = reduced
    if p > 1:
        companion[d:, : (p - 1) * d] = np.eye((p - 1) * d)
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def generate(
    ib_inv: np.ndarray,
    a_stack: np.ndarray,
    m_stack: np.ndarray | None,
    innovations: np.ndarray,
    burn_in: int,
) -> np.ndarray:
    """Run the VAR/VARMA recurrence over ``innovations``; drop the ``burn_in`` prefix."""
    total, d = innovations.shape
    p = a_stack.shape[0]
    q = 0 if m_stack is None else m_stack.shape[0]
    x = np.zeros((total, d), dtype=np.float64)
    for t in range(total):
        rhs = innovations[t].copy()
        for k in range(1, p + 1):
            if t >= k:
                rhs += a_stack[k - 1] @ x[t - k]
        for k in range(1, q + 1):
            if t >= k:
                rhs += m_stack[k - 1] @ innovations[t - k]  # ty: ignore[not-subscriptable]  # m_stack non-None when q>0; loop empty if None
        x[t] = ib_inv @ rhs
    return x[burn_in:]


def build_truth(
    draw: DAGDraw, b0: np.ndarray, a_stack: np.ndarray, m_stack: np.ndarray | None
) -> TemporalStructure:
    """Build the ``TemporalStructure`` truth: lag-0 DAG + per-lag digraphs + weight stacks."""
    graphs = [dag_truth(draw.parents, draw.children, draw.d)]  # lag-0 instantaneous DAG
    lags = [0]
    for k in range(a_stack.shape[0]):
        graphs.append(_lag_digraph(a_stack[k], draw.d))
        lags.append(k + 1)
    # The process matrices are [effect, cause]; a structure stores weights as [cause, effect].
    lag_weights = np.concatenate([b0[None], a_stack], axis=0).transpose(0, 2, 1)  # [B0, A_1..A_p]
    ma_weights = None if m_stack is None else m_stack.transpose(0, 2, 1)
    return TemporalStructure.from_lag_graphs(
        graphs, lags=lags, lag_weights=lag_weights, lag_weights_ma=ma_weights
    )


def _lag_digraph(a_k: np.ndarray, d: int) -> GraphStructure:
    """Build the digraph of a lag matrix's support (``a_k[i, j] != 0`` is the edge ``j -> i``)."""
    adj = a_k.T != 0  # adj[cause, effect]
    # mark at the row endpoint of pair (row, col)
    marks = np.where(adj.T, ARROW, np.where(adj, TAIL, NULL)).astype(np.int8)
    return GraphStructure.from_numpy(marks, kind="digraph", allow_self_loops=True)
