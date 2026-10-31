"""DYNOTEARS: the structural vector autoregression of one time series by continuous optimization.

The model is ``x_t = x_t W + x_{t-1} A_1 + ... + x_{t-p} A_p + z_t`` for row vectors ``x_t``:
``W[i, j]`` weights the same-step edge ``i -> j`` and ``A_k[i, j]`` the edge from ``i`` at ``t - k``
to ``j`` at ``t``. The fit minimizes the least-squares loss of every step ``t >= p`` on the same
step and the ``p`` before it, with an ``l1`` penalty on ``W`` and the ``A_k``, under the NOTEARS
acyclicity constraint on ``W`` alone, since a lagged edge points forward in time and closes no
cycle. It runs :func:`andrey.search.continuous.augmented_lagrangian` on the stacked weights
``[W; A_1; ...; A_p]``, then keeps the same-step weights that reach ``w_threshold`` and the lagged
ones that reach ``lag_threshold``, and drops the weakest same-step edges until no cycle is left.

References
----------
Pamfil, R., Sriwattanaworachai, N., Desai, S., Pilgerstorfer, P., Beaumont, P., Georgatzis, K.,
and Aragam, B. (2020). DYNOTEARS: Structure learning from time-series data. AISTATS, PMLR 108,
1595-1605.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from andrey.search.continuous import (
    Canonical,
    augmented_lagrangian,
    threshold_dag,
    trace_exponential,
)
from andrey.search.hc import tie_ranks

if TYPE_CHECKING:
    from andrey.core.knowledge import LaggedConstraints


@dataclass(frozen=True)
class DynotearsResult:
    """A thresholded fit in the data's column order.

    ``weights[k, i, j]`` is the coefficient of the edge from ``i`` at ``t - k`` to ``j`` at ``t``,
    lag 0 being the same step, and ``support[k, i, j]`` marks the edges kept; a required edge is
    kept whatever its weight. ``h_final`` is the acyclicity function of the same-step weights
    before the threshold, and ``edges_removed`` counts the same-step edges the threshold kept that
    the cycle repair then dropped.
    """

    weights: np.ndarray
    support: np.ndarray
    h_final: float
    edges_removed: int


def lagged_design(series: np.ndarray, tau_max: int) -> np.ndarray:
    """``[X | X_1 | ... | X_p]`` for ``p = tau_max``: row ``s`` holds steps ``t, t-1, ..., t-p``.

    ``t = p + s`` runs over the steps that have ``p`` steps before them, so the design has
    ``len(series) - p`` rows and ``d * (p + 1)`` columns.
    """
    steps = series.shape[0]
    return np.hstack([series[tau_max - k : steps - k] for k in range(tau_max + 1)])


def lagged_covariance(series: np.ndarray, tau_max: int) -> np.ndarray:
    """The covariance of :func:`lagged_design`'s columns, each centered on its own mean."""
    design = lagged_design(series, tau_max)
    centered = design - design.mean(axis=0, keepdims=True)
    return centered.T @ centered / design.shape[0]


def dynotears(
    series: np.ndarray,
    *,
    tau_max: int,
    lambda1: float,
    w_threshold: float,
    lag_threshold: float,
    max_iter: int,
    h_tol: float,
    rho_max: float,
    constraints: LaggedConstraints | None = None,
) -> DynotearsResult:
    """Fit DYNOTEARS (Pamfil et al. 2020) to ``series``, one row per time step.

    The columns are fit in the order of their :func:`~andrey.search.hc.tie_ranks`, so the result
    does not depend on the order they come in. ``constraints`` holds the knowledge over lags ``0``
    to ``tau_max``: a forbidden link stays at zero, and a required one leaves the penalty and the
    threshold and is never dropped by the cycle repair.
    """
    samples = np.asarray(series, dtype=np.float64)
    d = samples.shape[1]
    lags = tau_max + 1
    order = np.argsort(tie_ranks(samples))
    back = np.argsort(order)
    cov = lagged_covariance(samples[:, order], tau_max)
    if constraints is None:
        forbidden = required = np.zeros((lags, d, d), dtype=bool)
    else:
        permuted = constraints.permuted(order)
        forbidden, required = permuted.forbidden, permuted.required
    stacked = augmented_lagrangian(
        cov,
        forbidden.reshape(lags * d, d),
        required.reshape(lags * d, d),
        lambda1=lambda1,
        max_iter=max_iter,
        h_tol=h_tol,
        rho_max=rho_max,
        method="dynotears",
    ).reshape(lags, d, d)
    same_step = Canonical(cov[:d, :d], order, back, forbidden[0], required[0])
    h_final, _ = trace_exponential(stacked[0])
    contemporaneous, removed = threshold_dag(stacked[0], same_step, w_threshold)
    lagged = required[1:] | ((np.abs(stacked[1:]) >= lag_threshold) & ~forbidden[1:])
    support = np.concatenate((contemporaneous.astype(bool)[None], lagged))
    weights = np.where(support, stacked, 0.0)
    return DynotearsResult(
        weights=weights[:, back][:, :, back],
        support=support[:, back][:, :, back],
        h_final=float(h_final),
        edges_removed=removed,
    )
