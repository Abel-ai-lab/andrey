"""Temporal metrics: score a per-lag structure against the truth, lag by lag.

Splits the contemporaneous (lag-0) block from the lagged ones, scores each lag matrix as a directed
graph (precision/recall/f1 + Hamming SHD, where a reversal costs 2 -- time has no equivalence
class), collapses all lags into a summary graph, and reports the per-lag coefficient error where lag
weights are present. Lags align by value, so a gappy or Granger-style (no lag-0) stack scores
correctly.

Lag-0 is scored only when the estimate carries it (``0 in est.lags``), a clean output-class signal:
Granger emits no contemporaneous block, so its missing lag-0 is a class limit, not an error, and is
left out of both the per-lag scores and the summary graph. A truth-only lag >= 1 stays in the union
as false negatives, since a too-small maximum lag is a real miss.
"""

from __future__ import annotations

import numpy as np

from andrey.core.output import StructureOutput
from andrey.core.structure import TemporalStructure

from ._common import _offdiag, directed_adjacency, prf


def _temporal(x: object) -> TemporalStructure:
    """Unwrap a StructureOutput and require a TemporalStructure."""
    if isinstance(x, StructureOutput):
        x = x.structure
    if not isinstance(x, TemporalStructure):
        raise TypeError("temporal_scores needs a TemporalStructure")
    return x


def _lag_directed(t: TemporalStructure, lag: int, n: int) -> np.ndarray:
    """Directed support of lag ``lag`` (all-``False`` when that lag is absent)."""
    if lag in t.lags:
        return directed_adjacency(t.lag(lag).to_numpy())
    return np.zeros((n, n), dtype=bool)


def _masks_scores(est: np.ndarray, true_: np.ndarray) -> dict[str, float]:
    """Precision/recall/f1 between two boolean supports."""
    tp = int((est & true_).sum())
    return prf(tp, int((est & ~true_).sum()), int((~est & true_).sum()))


def _edge_scores(est: np.ndarray, true_: np.ndarray) -> dict[str, float]:
    """Directed-edge precision/recall/f1 and Hamming SHD (reversal costs 2) over two supports."""
    scores = _masks_scores(est, true_)
    scores["shd"] = int((est != true_).sum())
    return scores


def temporal_scores(estimated: object, true: object) -> dict[str, object]:
    """Per-lag and summary scores for a temporal structure.

    Returns ``per_lag`` (each lag's precision/recall/f1/shd), a ``contemporaneous`` block when
    the estimate carries lag-0, an aggregate ``lagged`` block over lags >= 1, a ``summary`` graph
    collapsing every scored lag (self-loops excluded so diagonals do not inflate it),
    and ``coefficient_mae_per_lag`` when both structures carry lag weights.

    Raises ``NotImplementedError`` for a time-carrying structure (``n_times > 0``): the ``(time,
    lag)`` axis has no metric yet, and scoring only the final-occasion projection would silently
    miss per-occasion differences.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey.core import GraphStructure, TemporalStructure
    >>> g = GraphStructure.from_numpy(np.array([[0, 1], [2, 0]], dtype=np.int8), kind="dag")
    >>> t = TemporalStructure.from_lag_graphs([g])
    >>> temporal_scores(t, t)["summary"]["f1"]
    1.0
    """
    est = _temporal(estimated)
    true_ = _temporal(true)
    if est.n_nodes != true_.n_nodes:
        raise ValueError(f"node-count mismatch: estimated {est.n_nodes}, true {true_.n_nodes}")
    if est.labels is not None and true_.labels is not None and est.labels != true_.labels:
        raise ValueError("label mismatch: estimated and true index different variables")
    if est.n_times > 0 or true_.n_times > 0:
        raise NotImplementedError(
            "temporal scoring does not yet cover the (time, lag) axis: it would compare only the "
            "final occasion and silently miss per-occasion differences. Score a specific "
            "occasion's lag stack via at(time, lag), or await a time-aware metric schema."
        )
    n = est.n_nodes
    lags_ge1 = sorted({lag for lag in set(est.lags) | set(true_.lags) if lag >= 1})
    scored_lags = ([0] if 0 in est.lags else []) + lags_ge1

    per_lag: dict[int, dict[str, float]] = {}
    summary_est = np.zeros((n, n), dtype=bool)
    summary_true = np.zeros((n, n), dtype=bool)
    lagged_est = np.zeros((n, n), dtype=bool)
    lagged_true = np.zeros((n, n), dtype=bool)
    result: dict[str, object] = {"per_lag": per_lag}

    for lag in scored_lags:
        d_est = _lag_directed(est, lag, n)
        d_true = _lag_directed(true_, lag, n)
        per_lag[lag] = _edge_scores(d_est, d_true)
        summary_est |= d_est
        summary_true |= d_true
        if lag == 0:
            result["contemporaneous"] = per_lag[lag]
        else:
            lagged_est |= d_est
            lagged_true |= d_true

    if lags_ge1:
        result["lagged"] = _edge_scores(lagged_est, lagged_true)
    off = _offdiag(n)
    result["summary"] = _masks_scores(summary_est & off, summary_true & off)

    est_w, true_w = est.lag_weights, true_.lag_weights
    if est_w is not None and true_w is not None:
        result["coefficient_mae_per_lag"] = _coefficient_mae_per_lag(
            est_w, est.lags, true_w, true_.lags, scored_lags
        )
    return result


def _coefficient_mae_per_lag(
    est_w: np.ndarray,
    est_lags: tuple[int, ...],
    true_w: np.ndarray,
    true_lags: tuple[int, ...],
    lags: list[int],
) -> dict[int, float]:
    """Mean absolute lag-weight error over the true edges, per lag present in both stacks."""
    out: dict[int, float] = {}
    for lag in lags:
        if lag not in est_lags or lag not in true_lags:
            continue
        w_est = est_w[est_lags.index(lag)]
        w_true = true_w[true_lags.index(lag)]
        mask = w_true != 0
        out[lag] = float(np.abs(w_est[mask] - w_true[mask]).mean()) if mask.any() else 0.0
    return out
