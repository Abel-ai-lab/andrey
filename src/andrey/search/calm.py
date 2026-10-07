"""CALM, the engine behind :func:`andrey.calm`.

The algorithm, its settings, and its references are described on :func:`andrey.calm`.

``B[i, j]`` weights edge ``i -> j``. IAMB with a Fisher-Z test estimates each variable's Markov
blanket, and only edges inside that moral graph are candidates. Each candidate has a gate, a logit
passed through a Gumbel-sigmoid, so the penalty counts edges (an approximate ``l0``). An
augmented-Lagrangian loop raises ``rho`` across rounds while Adam minimizes the Gaussian
non-equal-variance negative log-likelihood ``0.5 * sum_j log(residual_var_j) - log|det(I - B)|``,
the gate penalty, and the acyclicity term ``h(M) = tr((I + M/d)^d) - d`` (Yu et al. 2019) on the
gate mask, until ``h`` is within tolerance. The gates are then thresholded, and any remaining cycle
loses its weakest edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from andrey.core import ARROW, TAIL, GraphStructure, StructureOutput
from andrey.core.ci import FisherZ
from andrey.core.orient import dag2cpdag, to_structure
from andrey.core.warning_policy import PerformanceWarning, warn_once


def _require_torch():
    """Import torch or raise a clear error naming the optional extra (torch is not a core dep).

    Warns when torch has no CUDA device: the continuous optimizer then runs on CPU, which is correct
    but can be slow on larger graphs.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "andrey CALM's continuous optimiser is built on torch (it runs on CPU or GPU); "
            "install the '[torch]' extra"
        ) from exc
    if not torch.cuda.is_available():
        warn_once(
            "CALM's torch optimiser is running on CPU; this is correct but can be slow on larger "
            "graphs, where a CUDA GPU is much faster.",
            PerformanceWarning,
        )
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


def _markov_blanket(test: FisherZ, target: int, d: int, alpha: float) -> list[int]:
    """Estimate ``target``'s Markov blanket by the interleaved IAMB grow/shrink search.

    Grow: repeatedly add the candidate most dependent on ``target`` given the current blanket
    (smallest p-value at or below ``alpha``) until none qualifies. Shrink: drop any member that has
    become conditionally independent of ``target`` given the rest. Returns the surviving indices.
    """
    blanket: list[int] = []
    while True:
        best_p, best_x = alpha, None
        for x in range(d):
            if x == target or x in blanket:
                continue
            p = test(target, x, blanket)
            if p <= best_p:
                best_p, best_x = p, x
        if best_x is None:
            break
        blanket.append(best_x)
    for x in list(blanket):
        rest = [z for z in blanket if z != x]
        if test(target, x, rest) > alpha:
            blanket.remove(x)
    return blanket


def moral_graph(data: np.ndarray, *, alpha: float = 0.01) -> np.ndarray:
    """Estimate the undirected moral graph (symmetric ``0/1`` mask) by an IAMB Markov network.

    Runs IAMB for every variable and keeps edge ``(i, j)`` only when each lies in the other's
    estimated Markov blanket (the AND rule), yielding a symmetric adjacency with a zero diagonal.
    """
    arr = np.asarray(data, dtype=np.float64)
    d = arr.shape[1]
    test = FisherZ(arr)
    raw = np.zeros((d, d), dtype=bool)
    for i in range(d):
        for j in _markov_blanket(test, i, d, alpha):
            raw[i, j] = True
    mask = (raw & raw.T).astype(np.float64)
    np.fill_diagonal(mask, 0.0)
    return mask


# --------------------------------------------------------------------------------------------
# Continuous optimization of the gated linear-Gaussian objective
# --------------------------------------------------------------------------------------------


def _optimise_gated_dag(
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
) -> tuple[np.ndarray, np.ndarray]:
    """Fit gate logits and edge weights; return ``(gate_prob, weight)`` numpy arrays.

    Runs the augmented-Lagrangian schedule with a Gumbel-sigmoid gate over the moral-graph support.
    ``gate_prob`` is the noiseless edge probability ``sigmoid(logit / tau)`` and ``weight`` the real
    coefficient matrix; the caller thresholds them into a binary support.
    """
    torch = _require_torch()
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d = cov.shape[0]
    cov_t = torch.from_numpy(cov).double()
    mask_t = torch.from_numpy(moral_mask).double()
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
        return g * mask_t

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
            loss = likelihood + lambda1 * g.sum() + 0.5 * rho * h * h
            loss.backward()
            opt.step()
            cur = loss.item()
            if step > 50 and abs(prev - cur) < 1e-7 * (abs(prev) + 1e-9):
                break
            prev = cur
        h_val = hard_h()
        if h_val <= h_tol or rho >= rho_max:
            break
        rho *= rho_mult

    with torch.no_grad():
        gate_prob = (torch.sigmoid(logit / tau) * mask_t).cpu().numpy()
        weight_np = weight.detach().cpu().numpy()
    return gate_prob, weight_np


# --------------------------------------------------------------------------------------------
# Acyclicity repair and endpoint marks
# --------------------------------------------------------------------------------------------


def _has_cycle(support: np.ndarray) -> bool:
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


def _break_cycles(weighted: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Drop the weakest edges until ``support`` is acyclic; return the pruned DAG mask."""
    mask = support.copy()
    edges = sorted(
        (
            (abs(weighted[i, j]), i, j)
            for i in range(mask.shape[0])
            for j in range(mask.shape[1])
            if mask[i, j]
        ),
        key=lambda e: e[0],
    )
    for _w, i, j in edges:
        if not _has_cycle(mask):
            break
        mask[i, j] = 0
    return mask


def _endpoint_marks(support: np.ndarray) -> np.ndarray:
    """Endpoint-mark adjacency for a DAG mask (``support[i, j]`` set means ``i -> j``)."""
    marks = np.zeros(support.shape, dtype=np.int8)
    src, dst = np.nonzero(support)
    marks[dst, src] = ARROW
    marks[src, dst] = TAIL
    return marks


def _residual_variances(weighted: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Per-node residual variances ``diag((I - B)^T S (I - B))`` for the fitted weights."""
    resid = np.eye(weighted.shape[0]) - weighted
    return np.diagonal(resid.T @ cov @ resid)


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
) -> CalmResult:
    """Recover a linear-Gaussian DAG by continuous acyclicity-constrained optimization.

    ``data`` is an ``(n, d)`` sample matrix (rows are observations). An estimated moral graph
    (Fisher-Z IAMB at level ``alpha``) prunes the candidate edges unless ``use_moral_graph`` is
    false. ``lambda1`` prices the gated ``l0`` edge count and ``tau`` sets the Gumbel-sigmoid
    temperature. The augmented-Lagrangian outer loop runs up to ``max_outer`` rounds, each an Adam
    subproblem of at most ``subproblem_iter`` steps at learning rate ``lr``; the penalty ``rho``
    grows from ``rho_init`` by ``rho_mult`` until the acyclicity residual clears ``h_tol`` or
    ``rho`` exceeds ``rho_max``. A gate is kept when its probability clears ``gate_threshold`` and
    its weight magnitude clears ``w_threshold``. Set ``standardize`` to z-score the columns first;
    ``seed`` makes the run reproducible. Returns a :class:`CalmResult`.
    """
    samples = np.asarray(data, dtype=np.float64)
    n, d = samples.shape
    centered = samples - samples.mean(axis=0, keepdims=True)
    if standardize:
        scale = centered.std(axis=0, keepdims=True)
        scale[scale == 0.0] = 1.0
        centered = centered / scale
    cov = centered.T @ centered / n

    mask = moral_graph(centered, alpha=alpha) if use_moral_graph else (1.0 - np.eye(d))

    if mask.sum() == 0:
        gate_prob = np.zeros((d, d))
        weight = np.zeros((d, d))
    else:
        gate_prob, weight = _optimise_gated_dag(
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
        )

    support = ((gate_prob >= gate_threshold) & (np.abs(weight) >= w_threshold)).astype(np.int8)
    support = _break_cycles(weight, support)
    weighted = weight * support

    marks = _endpoint_marks(support)
    dag = to_structure(marks, kind="dag", labels=labels)
    cpdag = to_structure(dag2cpdag(marks), kind="cpdag", labels=labels)

    resid_var = _residual_variances(weighted, cov)
    objective = 0.5 * float(np.sum(np.log(np.clip(resid_var, 1e-12, None))))
    objective += lambda1 * float(support.sum())
    poly = np.linalg.matrix_power(np.eye(d) + support.astype(float) / d, d)
    h_final = float(np.trace(poly) - d)

    output = StructureOutput.new(
        dag,
        weighted_adjacency=weighted,
        metadata={
            "algorithm": "calm",
            "n_edges": int(support.sum()),
            "h_final": h_final,
            "objective": objective,
            "used_moral_graph": bool(use_moral_graph),
            "moral_edges": int(mask.sum() // 2),
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
