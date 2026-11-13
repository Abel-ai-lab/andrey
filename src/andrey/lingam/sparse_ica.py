# The iteration below follows the code that accompanies Dai et al. (2024), which carries this
# notice:
#
# Copyright (c) 2024 Haoyue Dai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""ICA with sparse connections: a demixing matrix with an adaptive-Lasso penalty (Zhang and Chan).

The demixing matrix ``W`` has ones on its diagonal and is learned on the standardized columns by
natural-gradient steps on the likelihood of independent components, each component's score
function estimated from the data (Pham's kernel estimator). Two stages run in turn:

1. Initialization: each variable is regressed on the others, ``W`` starts from half the symmetrized
   negative coefficients, and the steps run with an L2 penalty ``regu * ||W||^2`` until the mean
   absolute gradient falls below ``1e-6`` or 5,000 steps are made.
2. Sparsification: the steps add an adaptive Lasso penalty ``lam * sum(omega * |W|)`` with weights
   ``omega = 1 / |W_init|`` (capped at three times their mean) and ``|w|`` smoothed as
   ``tanh(60 w) / 60``, and the L2 penalty halved. When the mean absolute gradient falls below
   ``1e-6``, the entries at most ``0.01`` in magnitude are fixed at their values, the penalty is
   dropped, and the steps continue on the others to the same tolerance; 15,000 steps at most.

Every step size adapts per entry: times 1.05 when its gradient keeps its sign, times 0.8 when it
flips, at most 0.03. The score functions are estimated afresh every 12th step of the first stage
and every 8th of the second, and in between each component's estimates are reassigned to its new
values by rank. Nothing is drawn at random, so a fit repeats exactly.

References
----------
Zhang, K., and Chan, L.-W. (2006). ICA with sparse connections. Intelligent Data Engineering and
Automated Learning (IDEAL 2006), LNCS 4224, 530-537.

Pham, D. T. (1996). Blind separation of instantaneous mixture of sources via an independent
component analysis. IEEE Transactions on Signal Processing 44(11), 2768-2779.

Dai, H., Ng, I., Zheng, Y., Gao, Z., and Zhang, K. (2024). Local causal discovery with linear
non-Gaussian cyclic models. AISTATS, PMLR 238.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Mean absolute gradient below which a stage stops.
TOLERANCE = 1e-6
#: Most steps of the initialization and of the sparsification.
INIT_STEPS, SPARSE_STEPS = 5_000, 15_000
#: Entries of at most this magnitude are fixed at their values once the penalized steps settle.
REFIT_ZERO = 0.01


@dataclass(frozen=True)
class SparseICA:
    """A fitted demixing matrix of the standardized columns, and how each stage stopped.

    ``W @ ((X - mean) / std).T`` gives the components; ``W`` has ones on its diagonal.
    ``init_steps`` and ``sparse_steps`` are the steps each stage made, and ``converged`` whether
    the sparsification stopped at the tolerance rather than at its step cap. The initialization
    often runs to its cap, which the schedule expects: it only sets the penalty's weights.
    """

    W: np.ndarray
    init_steps: int
    sparse_steps: int
    converged: bool


def pham_scores(y: np.ndarray) -> np.ndarray:
    """Return the estimated score function ``-(log p)'`` of each row of ``y`` at its values.

    ``y`` is ``(p, n)``, one component per row. Each row is centered and scaled to unit root mean
    square, its density is estimated on a grid by a kernel that is the density of a sum of three
    uniforms on ``[-1/2, 1/2]`` at the bandwidth that is optimal for a normal density, and the
    score is the derivative of the log density, corrected to have mean 0 and slope 1 against the
    data, then scaled back.
    """
    data = np.ascontiguousarray(y.T)
    n, p = data.shape
    width = 2 * (11 * np.sqrt(np.pi) / 20) ** (1 / 5) * (4 / (3 * n)) ** (1 / 5)
    data = data - data.mean(axis=0)
    scale = np.sqrt((data * data).mean(axis=0))
    data = data / scale
    r = data / width
    cell = np.floor(r).astype(np.int64)
    r = r - cell
    cell = cell - cell.min(axis=0) + 1
    cells = int(cell.max()) + 2
    column = np.arange(p)
    # Each datum adds (1-r)^2/2, 1/2 + r(1-r), r^2/2 to cells i-1, i, i+1 of its column, in the
    # order of the rows; one bincount over the interleaved triples keeps that order of summation.
    where = np.stack([(cell - 1) * p + column, cell * p + column, (cell + 1) * p + column], axis=-1)
    mass = np.stack([(1 - r) ** 2 / 2, 0.5 + r * (1 - r), r**2 / 2], axis=-1)
    prob = np.bincount(where.ravel(), weights=mass.ravel(), minlength=cells * p)
    prob = prob.reshape(cells, p) / n
    logp = np.log(prob, out=np.zeros_like(prob), where=prob != 0)
    psi = (
        np.ascontiguousarray(
            (
                logp[cell - 1, column] * (1 - r)
                + logp[cell, column] * (2 * r - 1)
                - logp[cell + 1, column] * r
            ).T
        )
        / width
    )
    psi = psi - psi.mean(axis=1)[:, None]
    slope = (psi.T * data).sum(axis=0) / n - 1
    return ((psi.T - data * slope) / scale).T


def _step_sizes(
    grad: np.ndarray, previous: np.ndarray, eta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the new per-entry step sizes and the step direction (the gradient)."""
    same = (grad * previous) >= 0
    eta = eta * (1.05 * same + 0.8 * (1 - same))
    eta[eta >= 0.03] = 0.03
    return eta, grad


def _score(
    y: np.ndarray, step: int, every: int, psi: np.ndarray, by_rank: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the scores at ``y`` and their values by rank: afresh every ``every`` steps, else the
    last estimates reassigned to the new values of each component by rank."""
    order = np.argsort(y, axis=1)
    if step % every == 0:
        psi = -pham_scores(y)
        by_rank = np.take_along_axis(psi, order, axis=1)
    else:
        np.put_along_axis(psi, order, by_rank, axis=1)
    return psi, by_rank


def _initial(X: np.ndarray, mask: np.ndarray, regu: float) -> tuple[np.ndarray, int]:
    """Stage 1: natural-gradient ICA with an L2 penalty from the regression start."""
    N, T = X.shape
    start = np.eye(N)
    for i in range(N):
        others = np.flatnonzero(mask[i])
        Xo = X[others]
        start[i, others] = -0.5 * (X[i] @ Xo.T) @ np.linalg.pinv(Xo @ Xo.T)
    W = 0.5 * (start + start.T)
    W[np.diag_indices(N)] = 1
    eta = 3e-3 * np.ones_like(W)
    psi, by_rank = np.zeros_like(X), np.zeros_like(X)
    edges = mask.sum()
    previous = None
    for step in range(INIT_STEPS):
        y = W @ X
        psi, by_rank = _score(y, step, 12, psi, by_rank)
        grad = psi @ X.T / float(T) + np.linalg.inv(W.T) - 2 * regu * W
        if previous is None:
            previous = grad.copy()
        eta, direction = _step_sizes(grad, previous, eta)
        W = W + eta * direction * mask
        if np.abs(grad * mask).sum() / edges < TOLERANCE:
            return W, step + 1
        previous = grad.copy()
    return W, INIT_STEPS


def sparse_ica(X: np.ndarray, *, lam: float | None = None, regu: float = 0.05) -> SparseICA:
    """Fit the sparse demixing matrix of ``X`` (``n`` rows by ``m`` columns), as described above.

    ``lam`` is the adaptive Lasso's weight, ``4 log n`` by default, and ``regu`` the L2 penalty's.
    """
    X = np.asarray(X, dtype=np.float64)
    T, N = X.shape
    lam = 4 * np.log(T) if lam is None else lam
    XX = (X - X.mean(axis=0)).T
    XX = np.ascontiguousarray(XX * (1.0 / XX.std(axis=1, ddof=1))[:, None])
    mask = np.ones((N, N)) - np.eye(N)
    edges = mask.sum()
    W_init, init_steps = _initial(XX, mask, regu)
    weight = 1.0 / np.abs(W_init[mask != 0])
    upper = 3 * weight.mean()
    weight[weight > upper] = upper
    omega = np.zeros((N, N))
    omega[mask != 0] = weight
    W = W_init.copy()
    eta = 1e-3 * np.ones_like(W)
    grad = W + np.eye(N)
    psi, by_rank = np.zeros_like(XX), np.zeros_like(XX)
    previous = None
    refit = True
    done = False
    steps = SPARSE_STEPS
    for step in range(SPARSE_STEPS):
        y = W @ XX
        if np.abs(grad * mask).sum() / edges < TOLERANCE:
            if not refit:
                done, steps = True, step
                break
            mask = np.abs(W) > REFIT_ZERO
            mask[np.diag_indices(N)] = False
            lam, refit = 0.0, False
        psi, by_rank = _score(y, step, 8, psi, by_rank)
        grad = (
            psi @ XX.T / T
            + np.linalg.inv(W.T)
            - omega * np.tanh(60 * W) * lam / T
            - 2 * (regu / 2.0) * W
        )
        if previous is None:
            previous = grad.copy()
        eta, direction = _step_sizes(grad, previous, eta)
        W = W + 0.9 * (eta * direction) * mask
        previous = grad.copy()
    return SparseICA(W=W, init_steps=init_steps, sparse_steps=steps, converged=done)
