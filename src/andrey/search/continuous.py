"""The engine shared by the continuous-optimization DAG learners.

``W[i, j]`` weights edge ``i -> j``. Each learner minimizes a loss over ``W`` with a penalty on the
sum of absolute weights and an acyclicity function that is zero exactly when ``W`` is a DAG. This
module holds the pieces they share: the least-squares loss written through the sample covariance,
the acyclicity functions (the trace of a matrix exponential and the log-determinant of an M-matrix),
the column canonicalization that makes a fit follow the data and not the column order, and the
threshold and cycle repair that turn weights into a DAG. The solvers are in
:func:`~andrey.search.continuous.augmented_lagrangian`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
import scipy.optimize

from andrey.core import ARROW, TAIL, GraphStructure
from andrey.core.orient import to_structure
from andrey.search.hc import tie_ranks

if TYPE_CHECKING:
    from andrey.core.knowledge import Constraints

# --------------------------------------------------------------------------------------------
# Loss and acyclicity
# --------------------------------------------------------------------------------------------


def least_squares(cov: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Value and gradient of ``0.5 * tr((I - W)^T S (I - W))`` for the sample covariance ``S``.

    This is half the mean squared residual summed over the variables, so it equals
    ``0.5 / n * ||X - X W||^2`` for centered data ``X``.
    """
    residual = np.eye(cov.shape[0]) - weights
    product = cov @ residual
    return 0.5 * float(np.trace(residual.T @ product)), -product


def trace_exponential(weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Value and gradient of ``h(W) = tr(exp(W * W)) - d`` (Zheng et al. 2018).

    ``h`` is zero exactly when ``W`` has no directed cycle, and positive otherwise.
    """
    expm = scipy.linalg.expm(weights * weights)
    return float(np.trace(expm)) - weights.shape[0], expm.T * weights * 2.0


# --------------------------------------------------------------------------------------------
# Column canonicalization
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Canonical:
    """The data in the order the solvers see it.

    ``cov`` is the covariance of the columns taken in order of their
    :func:`~andrey.search.hc.tie_ranks`: fit column ``k`` is the data's column ``order[k]``. A
    matrix ``fit`` in that order maps back to the data's order with ``fit[np.ix_(back, back)]``.
    ``forbidden[i, j]`` fixes ``W[i, j]`` at zero and ``required[i, j]`` keeps edge ``i -> j``
    without a penalty or a threshold; both are in the fit's column order.
    """

    cov: np.ndarray
    order: np.ndarray
    back: np.ndarray
    forbidden: np.ndarray
    required: np.ndarray


def canonicalize(data: np.ndarray, constraints: Constraints | None = None) -> Canonical:
    """Order the columns of ``data`` by their ranks, center them, and form their covariance.

    ``constraints`` (background knowledge in the data's column order) is carried into the fit's
    column order; ``None`` forbids and requires nothing.
    """
    samples = np.asarray(data, dtype=np.float64)
    d = samples.shape[1]
    order = np.argsort(tie_ranks(samples))
    samples = samples[:, order]
    centered = samples - samples.mean(axis=0, keepdims=True)
    if constraints is None:
        forbidden = required = np.zeros((d, d), dtype=bool)
    else:
        forbidden = constraints.forbidden[np.ix_(order, order)]
        required = constraints.required[np.ix_(order, order)]
    return Canonical(
        centered.T @ centered / samples.shape[0], order, np.argsort(order), forbidden, required
    )


# --------------------------------------------------------------------------------------------
# Augmented Lagrangian with L-BFGS-B (NOTEARS)
# --------------------------------------------------------------------------------------------


def augmented_lagrangian(
    canonical: Canonical, *, lambda1: float, max_iter: int, h_tol: float, rho_max: float
) -> np.ndarray:
    """Minimize ``loss(W) + lambda1 * |W|_1`` subject to ``h(W) = 0``; return the weights.

    The loss is over ``canonical.cov``. Entries the knowledge forbids stay at zero, and entries it
    requires leave the ``l1`` term. Writes ``W = W+ - W-`` with both parts non-negative and zero
    on the diagonal and on the forbidden entries. The inner problem
    ``loss + 0.5 * rho * h**2 + alpha * h + lambda1 * sum(W+ + W-)`` is solved by L-BFGS-B from the
    previous solution. After each solve, ``rho`` grows tenfold while ``h`` has not fallen to a
    quarter of its previous value; then ``alpha`` takes a dual step ``alpha += rho * h``. The loop
    stops when ``h <= h_tol``, ``rho >= rho_max``, or after ``max_iter`` dual steps.
    """
    cov = canonical.cov
    d = cov.shape[0]
    penalty = lambda1 * (~canonical.required).astype(np.float64)

    def split_to_weights(w: np.ndarray) -> np.ndarray:
        return (w[: d * d] - w[d * d :]).reshape(d, d)

    def objective(w: np.ndarray, rho: float, alpha: float) -> tuple[float, np.ndarray]:
        weights = split_to_weights(w)
        loss, grad_loss = least_squares(cov, weights)
        h, grad_h = trace_exponential(weights)
        value = loss + 0.5 * rho * h * h + alpha * h + penalty.ravel() @ (w[: d * d] + w[d * d :])
        smooth = grad_loss + (rho * h + alpha) * grad_h
        return value, np.concatenate((smooth + penalty, -smooth + penalty), axis=None)

    fixed = canonical.forbidden | np.eye(d, dtype=bool)
    bounds = [
        (0, 0) if fixed[i, j] else (0, None) for _ in range(2) for i in range(d) for j in range(d)
    ]
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    for _ in range(max_iter):
        w_new, h_new = w_est, h
        while rho < rho_max:
            solution = scipy.optimize.minimize(
                objective, w_est, args=(rho, alpha), method="L-BFGS-B", jac=True, bounds=bounds
            )
            w_new = solution.x
            h_new, _ = trace_exponential(split_to_weights(w_new))
            if h_new > 0.25 * h:
                rho *= 10.0
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    return split_to_weights(w_est)


# --------------------------------------------------------------------------------------------
# Threshold, acyclicity repair, and endpoint marks
# --------------------------------------------------------------------------------------------


def has_cycle(support: np.ndarray) -> bool:
    """True if the directed graph ``support`` (``[i, j]`` is ``i -> j``) has a directed cycle."""
    d = support.shape[0]
    indeg = support.sum(axis=0).astype(int)
    stack = [k for k in range(d) if indeg[k] == 0]
    visited = 0
    while stack:
        node = stack.pop()
        visited += 1
        for nxt in np.nonzero(support[node])[0]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                stack.append(int(nxt))
    return visited < d


def break_cycles(
    weighted: np.ndarray, support: np.ndarray, keep: np.ndarray | None = None
) -> np.ndarray:
    """Drop the weakest edges until ``support`` is acyclic; return the pruned DAG mask.

    An edge marked in ``keep`` is never dropped, so ``keep`` must itself be acyclic.
    """
    mask = support.copy()
    edges = sorted(
        (
            (abs(weighted[i, j]), i, j)
            for i in range(mask.shape[0])
            for j in range(mask.shape[1])
            if mask[i, j] and (keep is None or not keep[i, j])
        ),
        key=lambda e: e[0],
    )
    for _w, i, j in edges:
        if not has_cycle(mask):
            break
        mask[i, j] = 0
    return mask


def endpoint_marks(support: np.ndarray) -> np.ndarray:
    """Endpoint-mark adjacency for a DAG mask (``support[i, j]`` set means ``i -> j``)."""
    marks = np.zeros(support.shape, dtype=np.int8)
    src, dst = np.nonzero(support)
    marks[dst, src] = ARROW
    marks[src, dst] = TAIL
    return marks


def residual_variances(weighted: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Per-node residual variances ``diag((I - B)^T S (I - B))`` for the fitted weights."""
    resid = np.eye(weighted.shape[0]) - weighted
    return np.diagonal(resid.T @ cov @ resid)


@dataclass(frozen=True)
class ContinuousResult:
    """A thresholded, acyclic weight matrix and the bookkeeping of getting there.

    ``weighted_adjacency[i, j]`` is the coefficient of edge ``i -> j`` in the data's column order.
    ``dag`` is the graph of its nonzero entries. ``h_final`` is the acyclicity function of the
    solver's weights before the threshold, and ``edges_removed`` counts the edges the threshold
    kept that the cycle repair then dropped.
    """

    weighted_adjacency: np.ndarray
    dag: GraphStructure
    h_final: float
    edges_removed: int


def finish(
    weights: np.ndarray,
    canonical: Canonical,
    *,
    threshold: float,
    labels: tuple[str, ...] | None,
) -> ContinuousResult:
    """Threshold ``weights`` (fit in the canonical order), repair cycles, and map back.

    An entry is kept when its magnitude reaches ``threshold``, and a required entry is kept at any
    magnitude. Edges are then dropped in order of increasing magnitude until no cycle remains,
    never a required one, with ties in the canonical column order, so the result does not depend
    on the order the data came in.
    """
    h_final, _ = trace_exponential(weights)
    kept = canonical.required | ((np.abs(weights) >= threshold) & ~canonical.forbidden)
    np.fill_diagonal(kept, False)
    support = kept.astype(np.int8)
    repaired = break_cycles(weights, support, canonical.required)
    back = canonical.back
    weighted = (weights * repaired)[np.ix_(back, back)]
    dag = to_structure(endpoint_marks(repaired[np.ix_(back, back)]), kind="dag", labels=labels)
    return ContinuousResult(weighted, dag, h_final, int(support.sum() - repaired.sum()))


def notears(
    data: np.ndarray,
    *,
    lambda1: float,
    w_threshold: float,
    max_iter: int,
    h_tol: float,
    rho_max: float,
    labels: tuple[str, ...] | None = None,
    constraints: Constraints | None = None,
) -> ContinuousResult:
    """Linear NOTEARS (Zheng et al. 2018) with the least-squares loss, on the canonical columns."""
    canonical = canonicalize(data, constraints)
    weights = augmented_lagrangian(
        canonical, lambda1=lambda1, max_iter=max_iter, h_tol=h_tol, rho_max=rho_max
    )
    return finish(weights, canonical, threshold=w_threshold, labels=labels)
