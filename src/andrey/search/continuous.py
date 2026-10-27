"""The engine shared by the continuous-optimization DAG learners.

``W[i, j]`` weights edge ``i -> j``. Each learner minimizes a loss over ``W`` with a sparsity
penalty and an acyclicity function that is zero exactly when ``W`` is a DAG. This module holds the
pieces they share: the losses written through the sample covariance (least squares and the Gaussian
negative log-likelihood with one noise variance per variable), the penalties (the sum of absolute
weights and the quasi-MCP), the acyclicity functions (the trace of a matrix exponential and the
log-determinant of an M-matrix), the column canonicalization that makes a fit follow the data and
not the column order, the two solvers (an augmented Lagrangian, and the central path of DAGMA), the
likelihood path that refits a solution under a shrinking penalty, and the threshold and cycle
repair that turn weights into a DAG.
"""

from __future__ import annotations

from collections.abc import Callable
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


#: A loss: ``(sample covariance, weights) -> (value, gradient)``.
Loss = Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]]
#: A penalty: ``weights -> (value of each entry, gradient of each entry)``.
Penalty = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]

# A residual variance below this fraction of the mean column variance counts as that fraction, so
# the logarithm stays finite for a cyclic fit that explains a column exactly.
_VARIANCE_FLOOR = 1e-12


def gaussian_likelihood(cov: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Value and gradient of ``0.5 * sum_j log(((I - W)^T S (I - W))[j, j]) - log|det(I - W)|``.

    This is the Gaussian negative log-likelihood of the linear model with one noise variance per
    variable (the variances at their maximum-likelihood values), up to a constant; the determinant
    term is zero for a DAG. Rescaling a column adds a constant to the value (Deng et al. 2024).
    A singular ``I - W`` gives an infinite value and a zero gradient.
    """
    d = cov.shape[0]
    residual = np.eye(d) - weights
    sign, logdet = np.linalg.slogdet(residual)
    if sign == 0 or not np.isfinite(logdet):
        return np.inf, np.zeros_like(weights)
    product = cov @ residual
    variances = np.einsum("ij,ij->j", residual, product)
    variances = np.maximum(variances, _VARIANCE_FLOOR * float(np.trace(cov)) / d)
    value = 0.5 * float(np.log(variances).sum()) - logdet
    return value, -product / variances + np.linalg.inv(residual).T


def l1_penalty(lambda1: float) -> Penalty:
    """The penalty ``lambda1 * |W|`` on each entry (the subgradient is zero at zero)."""

    def penalty(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return lambda1 * np.abs(weights), lambda1 * np.sign(weights)

    return penalty


def quasi_mcp(lam: float, delta: float) -> Penalty:
    """The quasi-MCP penalty of each entry (Deng et al. 2024).

    It is ``lam * (|t| - t**2 / (2 * delta))`` for ``|t| <= delta`` and the constant
    ``lam * delta / 2`` above, so a weight larger than ``delta`` costs the same however large it is.
    """

    def penalty(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inside = np.abs(weights) <= delta
        value = lam * np.where(inside, np.abs(weights) - weights**2 / (2.0 * delta), delta / 2.0)
        grad = lam * np.where(inside, np.sign(weights) - weights / delta, 0.0)
        return value, grad

    return penalty


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
        return float(value), np.concatenate((smooth + penalty, -smooth + penalty), axis=None)

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
# Augmented Lagrangian with Adam (NOTEARS with a smooth penalty)
# --------------------------------------------------------------------------------------------


def adam_augmented_lagrangian(
    canonical: Canonical,
    *,
    loss: Loss,
    penalty: Penalty,
    start: np.ndarray,
    max_iter: int,
    h_tol: float,
    rho_max: float,
    lr: float,
    num_steps: int,
) -> np.ndarray:
    """The augmented Lagrangian of :func:`augmented_lagrangian`, its inner problem solved by Adam.

    The inner problem ``loss + 0.5 * rho * h**2 + alpha * h + penalty`` is over ``W`` itself, from
    ``start``, with ``h`` the trace of a matrix exponential. Adam (betas 0.99 and 0.999) takes up to
    ``num_steps`` steps of size ``lr`` and stops early when, every 1000 steps, the objective changes
    by a relative amount below ``1e-6``. The ``rho`` and ``alpha`` updates and the stopping rule are
    those of :func:`augmented_lagrangian`. Forbidden and diagonal entries stay at zero, and the
    penalty leaves the required entries.
    """
    cov = canonical.cov
    d = cov.shape[0]
    fixed = canonical.forbidden | np.eye(d, dtype=bool)
    free = ~fixed
    penalized = ~canonical.required

    def objective(weights: np.ndarray, rho: float, alpha: float) -> tuple[float, np.ndarray]:
        value, grad = loss(cov, weights)
        h, grad_h = trace_exponential(weights)
        pen_value, pen_grad = penalty(weights)
        value += 0.5 * rho * h * h + alpha * h + float(pen_value[penalized].sum())
        grad = grad + (rho * h + alpha) * grad_h + pen_grad * penalized
        return value, grad * free

    def solve(weights: np.ndarray, rho: float, alpha: float) -> np.ndarray:
        weights = np.where(free, weights, 0.0)
        first = np.zeros_like(weights)
        second = np.zeros_like(weights)
        previous = 1e16
        for step in range(num_steps):
            value, grad = objective(weights, rho, alpha)
            first = 0.01 * grad + 0.99 * first
            second = 0.001 * grad * grad + 0.999 * second
            direction = (first / (1.0 - 0.99 ** (step + 1))) / (
                np.sqrt(second / (1.0 - 0.999 ** (step + 1))) + 1e-8
            )
            weights = (weights - lr * direction) * free
            if step % 1000 == 0 or step == num_steps - 1:
                value, _ = objective(weights, rho, alpha)
                if abs(previous - value) / max(abs(previous), abs(value), 1.0) < 1e-6:
                    break
                previous = value
        return weights

    w_est, rho, alpha, h = np.where(free, start, 0.0), 1.0, 0.0, np.inf
    for _ in range(max_iter):
        w_new, h_new = w_est, h
        while rho < rho_max:
            w_new = solve(w_est, rho, alpha)
            h_new, _ = trace_exponential(w_new)
            if h_new > 0.25 * h:
                rho *= 10.0
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    return w_est


# --------------------------------------------------------------------------------------------
# The central path of DAGMA
# --------------------------------------------------------------------------------------------

#: The domain parameters ``s`` of the central path; a longer path repeats the last one.
DAGMA_S = (1.0, 0.9, 0.8, 0.7, 0.6)


def log_det_acyclicity(weights: np.ndarray, s: float = 1.0) -> float:
    """``h(W) = -log det(s * I - W * W) + d * log(s)`` (Bello et al. 2022).

    It is zero exactly when ``W`` has no directed cycle and is finite while ``s * I - W * W`` is an
    M-matrix.
    """
    sign, logdet = np.linalg.slogdet(s * np.eye(weights.shape[0]) - weights * weights)
    return float(weights.shape[0] * np.log(s) - logdet) if sign > 0 else np.inf


def central_path(
    canonical: Canonical,
    *,
    loss: Loss,
    penalty: Penalty,
    start: np.ndarray,
    stages: int,
    warm_iter: int,
    max_iter: int,
    lr: float,
) -> np.ndarray:
    """Minimize ``mu * (loss + penalty) + h`` along DAGMA's decreasing ``mu``; return the weights.

    ``h`` is :func:`log_det_acyclicity`. Stage ``t`` starts at the previous stage's weights
    (``start`` for the first) with ``mu = 0.1**t`` and ``s = DAGMA_S[t]`` and takes Adam steps
    (betas 0.99 and 0.999, step size ``lr``), at most ``warm_iter`` of them, or ``max_iter`` in the
    last stage. Every 1000 steps the stage ends when the objective has changed by a relative amount
    of at most ``1e-6``. A step that leaves the M-matrix domain is undone and the step size halved.
    A stage that leaves the domain at its first step, or after ``s`` has fallen to 0.9, restarts
    with half the step size and ``s`` raised by 0.1. Forbidden and diagonal entries stay at zero,
    and the penalty leaves the required entries.
    """
    cov = canonical.cov
    d = cov.shape[0]
    identity = np.eye(d)
    free = ~(canonical.forbidden | np.eye(d, dtype=bool))
    penalized = ~canonical.required
    schedule = [DAGMA_S[min(t, len(DAGMA_S) - 1)] for t in range(stages)]

    def objective(weights: np.ndarray, mu: float, s: float) -> float:
        value, _ = loss(cov, weights)
        pen_value, _ = penalty(weights)
        return mu * (value + float(pen_value[penalized].sum())) + log_det_acyclicity(weights, s)

    def stage(weights: np.ndarray, mu: float, s: float, rate: float, steps: int):
        previous = 1e16
        first = np.zeros_like(weights)
        second = np.zeros_like(weights)
        direction = np.zeros_like(weights)
        for step in range(1, steps + 1):
            inverse = np.linalg.inv(s * identity - weights * weights) + 1e-16
            while (inverse < 0).any():
                if step == 1 or s <= 0.9:
                    return weights, False
                weights = weights + rate * direction
                rate *= 0.5
                if rate <= 1e-16:
                    return weights, True
                weights = weights - rate * direction
                inverse = np.linalg.inv(s * identity - weights * weights) + 1e-16
            _, grad_loss = loss(cov, weights)
            _, grad_pen = penalty(weights)
            grad = mu * (grad_loss + grad_pen * penalized) + 2.0 * weights * inverse.T
            first = 0.01 * grad + 0.99 * first
            second = 0.001 * grad * grad + 0.999 * second
            direction = (first / (1.0 - 0.99**step)) / (
                np.sqrt(second / (1.0 - 0.999**step)) + 1e-8
            )
            weights = (weights - rate * direction) * free
            if step % 1000 == 0 or step == steps:
                value = objective(weights, mu, s)
                if abs(previous - value) <= 1e-6 * max(abs(previous), 1e-300):
                    break
                previous = value
        return weights, True

    weights, mu = np.where(free, start, 0.0), 1.0
    for t in range(stages):
        rate, ok = lr, False
        while not ok:
            candidate, ok = stage(
                weights.copy(), mu, schedule[t], rate, max_iter if t == stages - 1 else warm_iter
            )
            if not ok:
                rate *= 0.5
                schedule[t] += 0.1
        weights, mu = candidate, mu * 0.1
    return weights


# --------------------------------------------------------------------------------------------
# The likelihood path
# --------------------------------------------------------------------------------------------


#: A path step lowers the likelihood when it does so by more than this fraction of its value, the
#: tolerance of every other convergence check here; a smaller fall is the solver's noise.
_PATH_TOL = 1e-6


def likelihood_path(
    canonical: Canonical,
    start: np.ndarray,
    solve: Callable[[np.ndarray, Penalty], np.ndarray],
    prune: Callable[[np.ndarray], np.ndarray],
    *,
    lam: float,
    delta: float,
    decay: float,
    max_steps: int,
) -> tuple[np.ndarray, int]:
    """Refit ``start`` under the likelihood and a shrinking quasi-MCP; return the weights and steps.

    Each step calls ``solve(weights, penalty)`` from the last accepted weights with the quasi-MCP of
    the current ``lam`` and ``delta`` and ``prune``s its result to an acyclic, thresholded graph,
    the weights the step returns. The step is accepted when the likelihood of those weights,
    :func:`gaussian_likelihood`, is lower than that of the last accepted weights by more than a
    ``_PATH_TOL`` fraction (the likelihood of ``start`` counts as infinite); ``lam`` and ``delta``
    are then multiplied by ``decay``. The path ends at the first step that does not lower the
    likelihood, or after ``max_steps`` accepted steps, and returns the last accepted weights, or
    ``start`` when none was accepted (Deng et al. 2024).
    """
    best, best_value, steps = start, np.inf, 0
    while steps < max_steps:
        candidate = prune(solve(best, quasi_mcp(lam, delta)))
        value, _ = gaussian_likelihood(canonical.cov, candidate)
        if np.isfinite(best_value) and not value < best_value - _PATH_TOL * abs(best_value):
            break
        best, best_value, steps = candidate, value, steps + 1
        lam, delta = lam * decay, delta * decay
    return best, steps


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
    ``dag`` is the graph of its nonzero entries. ``h_final`` is the learner's acyclicity function of
    the solver's weights before the threshold, ``edges_removed`` counts the edges the threshold kept
    that the cycle repair then dropped, and ``path_steps`` counts the accepted steps of the
    likelihood path (zero without it).
    """

    weighted_adjacency: np.ndarray
    dag: GraphStructure
    h_final: float
    edges_removed: int
    path_steps: int = 0


def threshold_dag(
    weights: np.ndarray, canonical: Canonical, threshold: float
) -> tuple[np.ndarray, int]:
    """The acyclic support of ``weights`` (canonical order) and how many edges the repair dropped.

    An entry is kept when its magnitude reaches ``threshold``, and a required entry is kept at any
    magnitude. Edges are then dropped in order of increasing magnitude until no cycle remains,
    never a required one, with ties in the canonical column order, so the result does not depend
    on the order the data came in.
    """
    kept = canonical.required | ((np.abs(weights) >= threshold) & ~canonical.forbidden)
    np.fill_diagonal(kept, False)
    support = kept.astype(np.int8)
    repaired = break_cycles(weights, support, canonical.required)
    return repaired, int(support.sum() - repaired.sum())


def finish(
    weights: np.ndarray,
    canonical: Canonical,
    *,
    threshold: float,
    labels: tuple[str, ...] | None,
    h_final: float,
    path_steps: int = 0,
) -> ContinuousResult:
    """Threshold ``weights`` (fit in the canonical order), repair cycles, and map back."""
    repaired, removed = threshold_dag(weights, canonical, threshold)
    back = canonical.back
    weighted = (weights * repaired)[np.ix_(back, back)]
    dag = to_structure(endpoint_marks(repaired[np.ix_(back, back)]), kind="dag", labels=labels)
    return ContinuousResult(weighted, dag, h_final, removed, path_steps)


def notears(
    data: np.ndarray,
    *,
    lambda1: float,
    w_threshold: float,
    max_iter: int,
    h_tol: float,
    rho_max: float,
    robust: bool,
    mcp_lambda: float,
    mcp_delta: float,
    mcp_decay: float,
    max_path: int,
    labels: tuple[str, ...] | None = None,
    constraints: Constraints | None = None,
) -> ContinuousResult:
    """NOTEARS (Zheng et al. 2018), and with ``robust`` the likelihood path of Deng et al. (2024).

    The first fit is the least-squares loss with an ``l1`` penalty by :func:`augmented_lagrangian`.
    With ``robust`` its thresholded, acyclic weights start :func:`likelihood_path`, whose steps
    (each thresholded again) solve :func:`adam_augmented_lagrangian` with
    :func:`gaussian_likelihood` and the quasi-MCP.
    """
    canonical = canonicalize(data, constraints)
    weights = augmented_lagrangian(
        canonical, lambda1=lambda1, max_iter=max_iter, h_tol=h_tol, rho_max=rho_max
    )
    steps = 0
    if robust:

        def prune(raw: np.ndarray) -> np.ndarray:
            return raw * threshold_dag(raw, canonical, w_threshold)[0]

        def solve(initial: np.ndarray, penalty: Penalty) -> np.ndarray:
            return adam_augmented_lagrangian(
                canonical,
                loss=gaussian_likelihood,
                penalty=penalty,
                start=initial,
                max_iter=max_iter,
                h_tol=h_tol,
                rho_max=rho_max,
                lr=_ADAM_RATE,
                num_steps=_ADAM_STEPS,
            )

        weights, steps = likelihood_path(
            canonical,
            prune(weights),
            solve,
            prune,
            lam=mcp_lambda,
            delta=mcp_delta,
            decay=mcp_decay,
            max_steps=max_path,
        )
    h_final, _ = trace_exponential(weights)
    return finish(
        weights, canonical, threshold=w_threshold, labels=labels, h_final=h_final, path_steps=steps
    )


def dagma(
    data: np.ndarray,
    *,
    lambda1: float,
    w_threshold: float,
    stages: int,
    warm_iter: int,
    max_iter: int,
    lr: float,
    robust: bool,
    mcp_lambda: float,
    mcp_delta: float,
    mcp_decay: float,
    max_path: int,
    labels: tuple[str, ...] | None = None,
    constraints: Constraints | None = None,
) -> ContinuousResult:
    """DAGMA (Bello et al. 2022), and with ``robust`` the likelihood path of Deng et al. (2024).

    The first fit is the least-squares loss with an ``l1`` penalty along :func:`central_path`. With
    ``robust`` its thresholded, acyclic weights start :func:`likelihood_path`, whose steps run the
    central path again with :func:`gaussian_likelihood` and the quasi-MCP.
    """
    canonical = canonicalize(data, constraints)

    def solve(initial: np.ndarray, loss: Loss, penalty: Penalty) -> np.ndarray:
        return central_path(
            canonical,
            loss=loss,
            penalty=penalty,
            start=initial,
            stages=stages,
            warm_iter=warm_iter,
            max_iter=max_iter,
            lr=lr,
        )

    weights = solve(np.zeros_like(canonical.cov), least_squares, l1_penalty(lambda1))
    steps = 0
    if robust:

        def prune(raw: np.ndarray) -> np.ndarray:
            return raw * threshold_dag(raw, canonical, w_threshold)[0]

        weights, steps = likelihood_path(
            canonical,
            prune(weights),
            lambda initial, penalty: solve(initial, gaussian_likelihood, penalty),
            prune,
            lam=mcp_lambda,
            delta=mcp_delta,
            decay=mcp_decay,
            max_steps=max_path,
        )
    return finish(
        weights,
        canonical,
        threshold=w_threshold,
        labels=labels,
        h_final=log_det_acyclicity(weights),
        path_steps=steps,
    )


#: The step size and the most steps of each inner problem of the robust NOTEARS path.
_ADAM_RATE = 3e-4
_ADAM_STEPS = 15000
