"""Structural expectation-maximization (EM) for searches whose fit reads only a covariance.

On data with missing values (``NaN``), :func:`structural_em` runs the alternating structural EM of
Friedman (1997) over a linear-Gaussian model of the columns: a mean ``mu``, coefficients ``B``
(``B[i, j]`` the weight of edge ``i -> j``), and noise variances ``omega``. It is made of three
steps:

- **E-step** (:func:`expected_moments`): under the current model, each row's missing values have a
  Gaussian distribution given its observed ones. Their conditional means fill the row, and their
  conditional covariances add to the second moments, which give the expected mean and the expected
  covariance ``S`` about it.
- **Parameter step**: the graph stays, and ``B`` and ``omega`` are the least-squares fit on ``S``
  of a DAG of it (:func:`fit_dag`), the parameters that maximize the expected log-likelihood of
  that graph. Every DAG of a CPDAG's class gives the same fitted covariance.
- **Structure step**: the method's own search runs on ``(S, n)``, with ``n`` every row, and a
  parameter step fits the graph it returns on the same ``S``.

The model starts from ``B = 0`` and each column's mean and variance over its observed values (the
available cases; Little and Rubin 2019), which are the maximum-likelihood parameters of the graph
with no edge, so the first round is an E-step and a structure step. Each later round runs an
E-step, then :data:`PARAMETER_STEPS` times a parameter step and an E-step, then a structure step.
:data:`ITERATIONS` rounds run, the number Gao et al. (2022) run.

Friedman (1997) repeats the parameter steps "until convergence or" a largest count and the rounds
"until convergence", and leaves the count open. The first search reads the expected covariance
under the model with no edge, where every covariance is shrunk toward zero by the share of rows
that observe both columns. Measured with ``ges``, ``boss``, and ``grasp`` at 20 and 50 variables
and 2,000 rows with 30% of the values missing, that search returned many extra edges; with one
parameter fit per round the later rounds removed them slowly and the graph was still changing
after 10 rounds, and with :data:`PARAMETER_STEPS` parameter steps before each search the last
round changed the graph in about 1 fit in 10. At 100 rows it did in about 4 in 10. The record
tells whether each round changed the graph, and a fit whose last round did logs a warning.

The rows are grouped by which columns they observe, so each group takes one solve per E-step, and
the E-step runs on the columns in order of their :func:`~andrey.core.ties.tie_ranks`, with BLAS on
one thread, so a new column order or more threads move no bit of ``S``.

The missingness must be ignorable: missing completely at random or at random given observed
values. Under values missing not at random the expected covariance is biased, and nothing in the
data can show it.

References
----------
- Friedman, N. (1997). Learning belief networks in the presence of missing values and hidden
  variables. Proceedings of the 14th International Conference on Machine Learning, 125-133.
- Gao, E., Ng, I., Gong, M., Shen, L., Huang, W., Liu, T., Zhang, K., and Bondell, H. (2022).
  MissDAG: Causal discovery in the presence of missing data with continuous additive noise models.
  Advances in Neural Information Processing Systems 35.
- Little, R. J. A., and Rubin, D. B. (2019). Statistical Analysis with Missing Data, 3rd edition.
  Wiley.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from andrey.core import backend
from andrey.core.orient import pdag2dag
from andrey.core.score import copy_groups
from andrey.core.structure import ARROW, TAIL
from andrey.core.ties import tie_ranks

if TYPE_CHECKING:
    from collections.abc import Callable

    from andrey.core.structure import GraphStructure

__all__ = [
    "ITERATIONS",
    "PARAMETER_STEPS",
    "expected_moments",
    "fit_dag",
    "observed_copies",
    "structural_em",
]

_log = logging.getLogger(__name__)

#: The rounds, each with one structure step, as Gao et al. (2022) run them.
ITERATIONS = 10
#: The parameter steps on the current graph before each structure step after the first. Friedman
#: (1997) leaves the count open ("until convergence or" a largest count, or "some specified number
#: of steps"). With 5, the graph had settled within the 10 rounds in about 9 of 10 fits measured at
#: 2,000 rows; with none it was still changing in most. Stopping the steps earlier, once a step
#: gains less than 0.01 or 0.0001 in log-likelihood, returned another graph in 8 and 2 of 360 fits.
PARAMETER_STEPS = 5

# A node's noise variance is at least this share of its expected variance, so the precision stays
# finite when a fitted node is a near copy of its parents.
_VARIANCE_FLOOR = 1e-12

R = TypeVar("R")


def expected_moments(
    X: np.ndarray, mean: np.ndarray, precision: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """The expected mean, the expected covariance, and the observed-data log-likelihood.

    ``X`` holds ``NaN`` where a value is missing; ``mean`` and ``precision`` (the inverse
    covariance) are the model's. A row that observes the columns ``o`` and misses ``m`` has
    ``x_m`` given ``x_o`` Gaussian with mean ``mean_m - K_mm^-1 K_mo (x_o - mean_o)`` and covariance
    ``K_mm^-1``, ``K`` the precision. The covariance is about the expected mean and divided by the
    number of rows. The log-likelihood is the sum over rows of the log density of each row's
    observed values under the model; a row that observes nothing adds zero.
    """
    n, d = X.shape
    observed = ~np.isnan(X)
    patterns, group = np.unique(observed, axis=0, return_inverse=True)
    filled = np.where(observed, X, 0.0)
    extra = np.zeros((d, d))
    log_det = np.linalg.slogdet(precision)[1]
    loglik = 0.0
    for k, pattern in enumerate(patterns):
        rows = np.flatnonzero(group.ravel() == k)
        o, m = np.flatnonzero(pattern), np.flatnonzero(~pattern)
        resid = X[np.ix_(rows, o)] - mean[o]
        if m.size == 0:
            inner = precision
            log_det_o = log_det
        else:
            k_mm = precision[np.ix_(m, m)]
            k_mo = precision[np.ix_(m, o)]
            factor = np.linalg.cholesky(k_mm)
            solved = np.linalg.solve(factor.T, np.linalg.solve(factor, k_mo))  # K_mm^-1 K_mo
            filled[np.ix_(rows, m)] = mean[m] - resid @ solved.T
            inverse = np.linalg.inv(factor)
            extra[np.ix_(m, m)] += rows.size * (inverse.T @ inverse)  # rows * K_mm^-1
            inner = precision[np.ix_(o, o)] - k_mo.T @ solved
            log_det_o = log_det - 2.0 * float(np.log(np.diagonal(factor)).sum())
        if o.size:
            quadratic = float(np.einsum("ij,jk,ik->", resid, inner, resid))
            loglik -= 0.5 * (quadratic + rows.size * (o.size * np.log(2 * np.pi) - log_det_o))
    expected_mean = filled.mean(axis=0)
    centered = filled - expected_mean
    return expected_mean, (centered.T @ centered + extra) / n, loglik


def observed_copies(X: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """The groups of columns of ``X`` that repeat one another up to scale and shift where observed.

    Two columns are compared on the rows that observe both, and are copies when their correlation
    there passes :func:`~andrey.core.score.copy_groups`; a pair observed together on at most two
    rows, or constant on them, is no copy. The expected covariance blurs a copy whose values go
    missing on other rows than the original's, so the linear-Gaussian BIC checks the observed
    values for copies before EM, as it checks complete data.
    """
    observed = ~np.isnan(X)
    centered = np.where(observed, X - np.nanmean(X, axis=0), 0.0)
    both = observed.astype(np.float64)
    count = both.T @ both  # count[i, j]: the rows that observe columns i and j
    sums = centered.T @ both  # sums[i, j]: column i summed over those rows
    squares = (centered**2).T @ both
    rows = np.maximum(count, 1.0)
    cross = centered.T @ centered / rows - (sums / rows) * (sums.T / rows)
    spread = (squares / rows - (sums / rows) ** 2) * (squares.T / rows - (sums.T / rows) ** 2)
    usable = (count > 2) & (spread > 0.0)
    corr = np.where(usable, cross / np.sqrt(np.where(usable, spread, 1.0)), 0.0)
    return copy_groups(corr, X.shape[0])


def fit_dag(adjacency: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The coefficients ``B`` and noise variances ``omega`` of the DAG ``adjacency`` fit on ``cov``.

    ``adjacency[i, j]`` marks edge ``i -> j``. Each node is regressed on its parents by least
    squares on ``cov``, and its noise variance is the residual variance, at least
    ``1e-12`` of its own variance.
    """
    d = cov.shape[0]
    B = np.zeros((d, d))
    omega = np.diagonal(cov).copy()
    for j in range(d):
        parents = np.flatnonzero(adjacency[:, j])
        if parents.size:
            weights = np.linalg.lstsq(cov[np.ix_(parents, parents)], cov[parents, j], rcond=None)[0]
            B[parents, j] = weights
            omega[j] = cov[j, j] - cov[j, parents] @ weights
    return B, np.maximum(omega, _VARIANCE_FLOOR * np.diagonal(cov))


def _precision(B: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """The inverse covariance ``(I - B) diag(1 / omega) (I - B)^T`` of the model ``(B, omega)``."""
    lower = np.eye(B.shape[0]) - B
    return (lower / omega) @ lower.T


def _dag_adjacency(structure: GraphStructure) -> np.ndarray:
    """A DAG of ``structure``'s class as a boolean ``i -> j`` matrix (:func:`pdag2dag`)."""
    marks = pdag2dag(structure.to_numpy())
    return (marks == TAIL) & (marks.T == ARROW)


def structural_em(
    X: np.ndarray,
    m_step: Callable[[np.ndarray], tuple[R, GraphStructure]],
    *,
    iterations: int = ITERATIONS,
) -> tuple[R, dict[str, object]]:
    """Run alternating structural EM on ``X``, which holds ``NaN`` where a value is missing.

    ``m_step(cov)`` runs the method's search on the expected covariance ``cov`` (in the columns of
    ``X``, divided by the number of rows) and returns its result and the graph it learned, a DAG,
    CPDAG, or a CPDAG narrowed by knowledge. ``iterations`` is the number of rounds, each with one
    such structure step; every round after the first runs :data:`PARAMETER_STEPS` parameter steps
    on its graph before it. Returns the last structure step's result and the record:

    - ``"iterations"``: the rounds;
    - ``"parameter_steps"``: the parameter steps before each structure step after the first;
    - ``"log_likelihood"``: the observed-data log-likelihood of the model after each round's
      structure step. A parameter step never lowers it; a structure step can, as the search
      weighs the expected log-likelihood against the graph's size;
    - ``"graph_changed"``: for each round, whether its structure step returned another graph than
      the round before, the first round against the graph with no edge.

    A last round that changed the graph logs a warning: the rounds ended before EM settled on a
    graph. A row that observes no value counts in the number of rows and adds the model's
    covariance to the expected covariance, so the facades drop such rows first
    (:func:`~andrey.api._adapt.data_matrix`).
    """
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations}")
    order = np.argsort(tie_ranks(X))  # the E-step's column k is the data's column order[k]
    back = np.argsort(order)
    ranked = np.ascontiguousarray(X[:, order])
    mean = np.nanmean(ranked, axis=0)
    precision = np.diag(1.0 / np.nanvar(ranked, axis=0))
    adjacency = np.zeros((X.shape[1], X.shape[1]), dtype=bool)
    marks = np.zeros_like(adjacency, dtype=np.int8)
    loglik: list[float] = []
    changed: list[bool] = []
    for k in range(iterations):
        with backend.one_blas_thread():
            mean, cov, value = expected_moments(ranked, mean, precision)
            if k:
                loglik.append(value)  # the model after the previous structure step
                for _ in range(PARAMETER_STEPS):
                    precision = _precision(*fit_dag(adjacency, cov))
                    mean, cov, _ = expected_moments(ranked, mean, precision)
        result, graph = m_step(cov[np.ix_(back, back)])
        found = np.asarray(graph.to_numpy())
        changed.append(not np.array_equal(found, marks))
        marks = found
        adjacency = _dag_adjacency(graph)[np.ix_(order, order)]
        precision = _precision(*fit_dag(adjacency, cov))
    with backend.one_blas_thread():
        loglik.append(expected_moments(ranked, mean, precision)[2])
    if changed[-1]:
        _log.warning(
            "structural EM: the graph still changed in the last of %d rounds, so EM has not "
            "settled on a graph; the result is the last round's",
            iterations,
        )
    return result, {
        "iterations": iterations,
        "parameter_steps": PARAMETER_STEPS,
        "log_likelihood": loglik,
        "graph_changed": changed,
    }
