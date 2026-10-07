"""Quality-assurance diagnostics and honesty checks for a sampled dataset.

:func:`report` computes the sortability diagnostics into a :class:`~andrey.data.dataset.QAReport`
(varsortability via the scalable edge-only estimator so it runs at any ``d``; R^2-sortability only
where it is defined and affordable -- ``n > d`` and small ``d``). :func:`faithfulness_margin` and
:class:`FaithfulnessScreen` support rejecting near-unfaithful draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import QAReport
from .sortability import directed_adjacency, r2sortability, standardize, varsortability_edges

# R^2-sortability uses a dense (d, d) correlation matrix and an all-paths loop; only affordable and
# non-singular for small d with n > d. Above this it is reported as skipped (nan), never guessed.
_R2_D_MAX = 100


def report(data: np.ndarray, edges: np.ndarray, d: int, *, scale: str = "raw") -> QAReport:
    """Compute the honesty diagnostics for a sampled dataset.

    Parameters
    ----------
    data : np.ndarray of shape (n, d)
        Observational data.
    edges : np.ndarray of shape (m, 2)
        Directed truth edges ``(parent, child)``.
    d : int
        Node count.
    scale : str, default="raw"
        The scale/normalization mode that produced ``data`` (recorded, not applied here).

    Returns
    -------
    QAReport
        ``varsortability`` (edge-only, scalable), ``r2sortability`` (or ``nan`` when ``n <= d`` or
        ``d`` is large), and the ``scale`` label.
    """
    x = np.asarray(data, dtype=np.float64)
    n = x.shape[0]
    # tol=1e-5 treats numerically-equal variances (standardize/rescale -> unit variance, error
    # ~1e-15 in float64) as ties, so honesty modes report exactly 0.5; genuine raw ratios differ
    # by factors, far above this.
    vs = varsortability_edges(x, edges, tol=1e-5)
    r2 = float("nan")
    if d <= _R2_D_MAX and n > d:
        r2 = r2sortability(x, directed_adjacency(edges, d))
    return QAReport(varsortability=vs, r2sortability=r2, scale=scale)


def faithfulness_margin(data: np.ndarray, edges: np.ndarray) -> float:
    """Smallest ``|correlation|`` over true adjacent pairs -- small values flag near-cancellation.

    Returns ``nan`` for an edgeless graph.
    """
    if edges.shape[0] == 0:
        return float("nan")
    x = np.asarray(data, dtype=np.float64)
    centered = x - x.mean(axis=0)
    sd = x.std(axis=0)
    p, c = edges[:, 0], edges[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (centered[:, p] * centered[:, c]).mean(axis=0) / (sd[p] * sd[c])
    return float(np.nanmin(np.abs(corr)))


@dataclass(frozen=True)
class FaithfulnessScreen:
    """A reject-resample screen for near-unfaithful draws.

    Bounded: raises rather than looping forever.

    Parameters
    ----------
    min_abs_corr : float, default=0.05
        Minimum acceptable ``|correlation|`` on every true edge.
    max_attempts : int, default=20
        Bound on resamples before the screen raises.
    """

    min_abs_corr: float = 0.05
    max_attempts: int = 20

    def accepts(self, margin: float) -> bool:
        """Whether a draw with faithfulness ``margin`` passes (``nan`` -> vacuously true)."""
        return bool(np.isnan(margin) or margin >= self.min_abs_corr)


__all__ = ["report", "faithfulness_margin", "FaithfulnessScreen", "standardize"]
