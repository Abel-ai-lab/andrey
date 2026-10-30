"""CALM, the engine behind :func:`andrey.calm`.

The algorithm, its settings, and its references are described on :func:`andrey.calm`.

``B[i, j]`` weights edge ``i -> j``. IAMB with a Fisher-Z test estimates each variable's Markov
blanket, and only edges inside that moral graph are candidates. Each candidate has a gate, a logit
passed through a Gumbel-sigmoid, so the penalty counts edges (an approximate ``l0``). A
quadratic-penalty loop raises ``rho`` across rounds while Adam minimizes the Gaussian
non-equal-variance negative log-likelihood ``0.5 * sum_j log(residual_var_j) - log|det(I - B)|``,
the gate penalty, and ``0.5 * rho * h**2`` for the acyclicity term ``h(M) = tr((I + M/d)^d) - d``
(Yu et al. 2019) on the gate mask, until ``h`` is within tolerance. The gates are then
thresholded, and edges are removed by increasing weight magnitude until no cycle remains.

Background knowledge, as :class:`~andrey.core.knowledge.Constraints`, edits the candidate edges:
the moral graph's pairs and the pairs of the required edges, less every forbidden direction. A
required edge's gate is held at one, so the acyclicity term counts it, the penalty and the
thresholds leave it, and the cycle repair keeps it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from andrey.constraint.blanket import blanket_graph
from andrey.core import GraphStructure, StructureOutput
from andrey.core.ci import FisherZ
from andrey.core.knowledge import Constraints
from andrey.core.orient import dag2cpdag, to_structure
from andrey.core.warning_policy import warn_search_limit
from andrey.search.continuous import (
    break_cycles,
    endpoint_marks,
    residual_variances,
)
from andrey.search.hc import tie_ranks


def _require_torch():
    """Import torch or raise a clear error naming the optional extra (torch is not a core dep)."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "andrey CALM's continuous optimizer is built on torch; install the '[torch]' extra"
        ) from exc
    return torch


@dataclass(frozen=True)
class CalmResult:
    """Recovered structure from the continuous search.

    ``weighted_adjacency`` holds the thresholded real coefficients (``[i, j]`` is edge ``i -> j``).
    ``dag`` is the acyclic endpoint-mark graph of the surviving edges and ``cpdag`` its essential
    graph (the Markov-equivalence-class representative). ``output`` is the uniform run envelope
    wrapping the DAG plus the weighted adjacency and run metadata. ``h_final`` is the acyclicity
    residual of the returned support and ``objective`` its likelihood-plus-sparsity value.
    """

    weighted_adjacency: np.ndarray
    dag: GraphStructure
    cpdag: GraphStructure
    output: StructureOutput
    h_final: float
    objective: float


# --------------------------------------------------------------------------------------------
# Moral graph (IAMB Markov network over a Fisher-Z test)
# --------------------------------------------------------------------------------------------


def moral_graph(data: np.ndarray, *, alpha: float = 0.01) -> np.ndarray:
    """Estimate the undirected moral graph (symmetric ``0/1`` mask) by an IAMB Markov network.

    Runs IAMB with a Fisher-Z test for every variable and keeps edge ``(i, j)`` only when each lies
    in the other's estimated Markov blanket (the AND rule,
    :func:`~andrey.constraint.blanket.blanket_graph`), yielding a symmetric adjacency with a zero
    diagonal.
    """
    arr = np.asarray(data, dtype=np.float64)
    d = arr.shape[1]
    test = FisherZ(arr)
    ranks = tie_ranks(arr).tolist()
    mask = blanket_graph(test, d, alpha, ranks).astype(np.float64)
    np.fill_diagonal(mask, 0.0)
    return mask


# --------------------------------------------------------------------------------------------
# Continuous optimization of the gated linear-Gaussian objective
# --------------------------------------------------------------------------------------------


_NOT_FINITE = (
    "CALM's loss or weights are not finite; rescale the data, for example with standardize=True"
)


def _optimize_gated_dag(
    cov: np.ndarray,
    moral_mask: np.ndarray,
    *,
    lambda1: float,
    tau: float,
    rho_init: float,
    rho_mult: float,
    rho_max: float,
    h_tol: float,
    max_outer: int,
    subproblem_iter: int,
    lr: float,
    seed: int,
    required: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit gate logits and edge weights; return ``(gate_prob, weight)`` numpy arrays.

    Runs the quadratic-penalty schedule with a Gumbel-sigmoid gate over the moral-graph support.
    ``gate_prob`` is the noiseless edge probability ``sigmoid(logit / tau)`` and ``weight`` the real
    coefficient matrix; the caller thresholds them into a binary support. A ``required`` edge's
    gate is held at one and left out of the penalty; its weight is fitted. Raises ``ValueError``
    when the loss or the fitted parameters are not finite.
    """
    torch = _require_torch()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d = cov.shape[0]
    cov_t = torch.from_numpy(cov).double()
    mask_t = torch.from_numpy(moral_mask).double()
    on = None if required is None else torch.from_numpy(required)
    eye = torch.eye(d, dtype=torch.double)

    weight = torch.empty(d, d, dtype=torch.double).uniform_(-1e-3, 1e-3, generator=gen)
    weight.requires_grad_(True)
    logit = torch.zeros(d, d, dtype=torch.double, requires_grad=True)

    def gate(sample: bool) -> torch.Tensor:
        if sample:
            u = torch.rand(d, d, generator=gen, dtype=torch.double).clamp_(1e-9, 1 - 1e-9)
            noise = torch.log(u) - torch.log1p(-u)
            g = torch.sigmoid((logit + noise) / tau)
        else:
            g = torch.sigmoid(logit / tau)
        g = g * mask_t
        return g if on is None else torch.where(on, 1.0, g)

    def penalty(g: torch.Tensor) -> torch.Tensor:
        return g.sum() if on is None else g.masked_fill(on, 0.0).sum()

    def acyclicity(gate_mask: torch.Tensor) -> torch.Tensor:
        poly = torch.matrix_power(eye + gate_mask / d, d)
        return torch.trace(poly) - d

    def hard_h() -> float:
        with torch.no_grad():
            return float(acyclicity(gate(sample=False)))

    rho, h_val = rho_init, hard_h()
    for _outer in range(max_outer):
        opt = torch.optim.Adam([weight, logit], lr=lr)
        prev = float("inf")
        for step in range(subproblem_iter):
            opt.zero_grad()
            g = gate(sample=True)
            B = g * weight
            resid = torch.diagonal((eye - B).T @ cov_t @ (eye - B)).clamp_min(1e-12)
            likelihood = 0.5 * torch.sum(torch.log(resid)) - torch.linalg.slogdet(eye - B)[1]
            h = acyclicity(g)
            loss = likelihood + lambda1 * penalty(g) + 0.5 * rho * h * h
            cur = loss.item()
            if not math.isfinite(cur):
                raise ValueError(_NOT_FINITE)
            loss.backward()
            opt.step()
            if step > 50 and abs(prev - cur) < 1e-7 * (abs(prev) + 1e-9):
                break
            prev = cur
        h_val = hard_h()
        if h_val <= h_tol or rho >= rho_max:
            break
        rho *= rho_mult
    else:
        warn_search_limit("calm", "max_iter", max_outer, "penalty rounds")

    with torch.no_grad():
        gate_prob = (torch.sigmoid(logit / tau) * mask_t).cpu().numpy()
        weight_np = weight.detach().cpu().numpy()
    if not (np.isfinite(gate_prob).all() and np.isfinite(weight_np).all()):
        raise ValueError(_NOT_FINITE)
    return gate_prob, weight_np


# --------------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------------


def calm(
    data: np.ndarray,
    *,
    lambda1: float = 0.005,
    alpha: float = 0.01,
    tau: float = 0.5,
    rho_init: float = 1e-3,
    rho_mult: float = 3.0,
    rho_max: float = 1e16,
    h_tol: float = 1e-8,
    max_outer: int = 25,
    subproblem_iter: int = 3000,
    lr: float = 1e-2,
    w_threshold: float = 0.3,
    gate_threshold: float = 0.5,
    standardize: bool = False,
    use_moral_graph: bool = True,
    seed: int = 0,
    labels: tuple[str, ...] | None = None,
    constraints: Constraints | None = None,
) -> CalmResult:
    """Recover a linear-Gaussian DAG by continuous acyclicity-constrained optimization.

    ``data`` is an ``(n, d)`` sample matrix (rows are observations). An estimated moral graph
    (Fisher-Z IAMB at level ``alpha``) prunes the candidate edges unless ``use_moral_graph`` is
    false. ``lambda1`` prices the gated ``l0`` edge count and ``tau`` sets the Gumbel-sigmoid
    temperature. The quadratic-penalty outer loop runs up to ``max_outer`` rounds, each an Adam
    subproblem of at most ``subproblem_iter`` steps at learning rate ``lr``; the penalty ``rho``
    grows from ``rho_init`` by ``rho_mult`` until the acyclicity residual clears ``h_tol`` or
    ``rho`` reaches ``rho_max``. A gate is kept when its probability clears ``gate_threshold`` and
    its weight magnitude clears ``w_threshold``. Set ``standardize`` to z-score the columns first;
    ``seed`` makes the run reproducible. The fit runs on the columns in order of their
    :func:`~andrey.search.hc.tie_ranks`, so each column keeps its initial weights and gate noise
    when the columns are reordered. ``constraints``, in the data's column order, edit the
    candidate edges and hold the required gates on, as the module describes. Returns a
    :class:`CalmResult`.
    """
    samples = np.asarray(data, dtype=np.float64)
    n, d = samples.shape
    order = np.argsort(tie_ranks(samples))  # the fit's column k is the data's column order[k]
    back = np.argsort(order)
    samples = samples[:, order]
    centered = samples - samples.mean(axis=0, keepdims=True)
    if standardize:
        scale = centered.std(axis=0, keepdims=True)
        scale[scale == 0.0] = 1.0
        centered = centered / scale
    cov = centered.T @ centered / n

    mask = moral_graph(centered, alpha=alpha) if use_moral_graph else (1.0 - np.eye(d))
    moral_edges = int(mask.sum() // 2)
    required = None
    if constraints is not None:
        fit = constraints.permuted(order)
        required = fit.required
        candidates = ((mask > 0) | required | required.T) & ~fit.forbidden
        np.fill_diagonal(candidates, False)
        mask = candidates.astype(np.float64)

    if mask.sum() == 0:
        gate_prob = np.zeros((d, d))
        weight = np.zeros((d, d))
    else:
        gate_prob, weight = _optimize_gated_dag(
            cov,
            mask,
            lambda1=lambda1,
            tau=tau,
            rho_init=rho_init,
            rho_mult=rho_mult,
            rho_max=rho_max,
            h_tol=h_tol,
            max_outer=max_outer,
            subproblem_iter=subproblem_iter,
            lr=lr,
            seed=seed,
            required=required,
        )

    kept = (gate_prob >= gate_threshold) & (np.abs(weight) >= w_threshold)
    if required is not None:
        kept = (kept & (mask > 0)) | required
    support = break_cycles(weight, kept.astype(np.int8), required)
    weighted = weight * support
    resid_var = residual_variances(weighted, cov)
    support, weighted = (m[np.ix_(back, back)] for m in (support, weighted))

    marks = endpoint_marks(support)
    dag = to_structure(marks, kind="dag", labels=labels)
    cpdag = to_structure(dag2cpdag(marks), kind="cpdag", labels=labels)

    objective = 0.5 * float(np.sum(np.log(np.clip(resid_var, 1e-12, None))))
    objective += lambda1 * float(support.sum())
    poly = np.linalg.matrix_power(np.eye(d) + support.astype(float) / d, d)
    h_final = float(np.trace(poly) - d)

    output = StructureOutput.new(
        dag,
        weighted_adjacency=weighted,
        metadata={
            "n_edges": int(support.sum()),
            "h_final": h_final,
            "objective": objective,
            "used_moral_graph": bool(use_moral_graph),
            "moral_edges": moral_edges,
        },
    )
    return CalmResult(
        weighted_adjacency=weighted,
        dag=dag,
        cpdag=cpdag,
        output=output,
        h_final=h_final,
        objective=objective,
    )
