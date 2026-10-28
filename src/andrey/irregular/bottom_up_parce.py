"""BottomUpParceLiNGAM: a causal order robust to latent confounders.

The order is built from the sink up. Each step regresses every candidate on the others and takes
as the next sink the one whose residual is most independent of the rest, by gamma-HSIC p-values
pooled with Fisher's method. The search stops when the best pooled p-value falls below a
Bonferroni-corrected threshold; the variables left form an order-unknown block, confounded or not
orderable. Adaptive Lasso on each variable's ordered predecessors gives ``B``
(``x_i = sum_j B[i, j] x_j``), with ``NaN`` on each unordered pair.

References
----------
T. Tashiro, S. Shimizu, A. Hyvarinen, T. Washio. "ParceLiNGAM: a causal ordering method robust
against latent confounders." Neural Computation 26(1): 57-83, 2014.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammainc

from andrey.core.independence import gamma_gram, hsic_gamma_from_grams
from andrey.core.units import unit_factors
from andrey.lingam import adjacency
from andrey.search._parallel import _KEY_DP
from andrey.search.hc import tie_ranks

if TYPE_CHECKING:
    from andrey.core.knowledge import Constraints

_Gram = tuple[np.ndarray, float]


@dataclass(frozen=True)
class BottomUpParceLiNGAMResult:
    """Fitted BottomUpParceLiNGAM weights and partial causal order.

    ``causal_order_`` is source-first and *nested*: an order-unknown block rides as a sub-list at
    the front. ``adjacency_matrix_`` is ``B`` with ``B[i, j]`` weighting edge ``j -> i``; unresolved
    pair ``(i, j)`` is ``NaN`` at both ``B[i, j]`` and ``B[j, i]``.
    """

    causal_order_: list
    adjacency_matrix_: np.ndarray


# ---- Fisher-aggregated HSIC independence test ---------------------------------------------------


def _fisher_hsic_test(
    predictor_grams: list[_Gram], residual: np.ndarray, n: int, bound: float
) -> tuple[float, float]:
    """Aggregate the per-predictor LiNGAM HSIC tests of ``residual`` by Fisher's method.

    Each predictor's independence from ``residual`` is scored by the median-bandwidth gamma HSIC in
    :mod:`andrey.core.independence`. The predictor grams are cached (they are fixed data columns);
    only the residual's gram is built here, once, and reused across predictors. Returns
    ``(fisher_p, fisher_stat)``. One predictor short-circuits to its HSIC ``(p, stat)``. Otherwise
    ``fisher_stat = sum_k -2 log(p_k)`` and ``fisher_p`` is its upper chi-squared tail on
    ``2 * n_predictors`` degrees of freedom (a zero HSIC p-value sends ``fisher_stat`` to infinity).

    ``bound`` is the smallest Fisher statistic seen so far in the current sink search, rounded to
    ``_KEY_DP`` decimals. Because the running sum only grows (each ``-2 log(p_k) >= 0``), once it
    rounds above ``bound`` this candidate can neither be the most-sink one nor tie with it, so the
    remaining predictor tests are skipped. The partial ``fisher_stat`` and its ``fisher_p`` then
    miss the incumbent and never win the comparison in :func:`_find_sink`, leaving the selected
    sink and its p-value unchanged.
    """
    residual_gram = gamma_gram(residual)
    if len(predictor_grams) == 1:
        statistic, p_value = hsic_gamma_from_grams(predictor_grams[0], residual_gram, n)
        return p_value, statistic

    fisher_stat = 0.0
    for gram in predictor_grams:
        _, p_value = hsic_gamma_from_grams(gram, residual_gram, n)
        fisher_stat += np.inf if p_value == 0 else -2 * np.log(p_value)
        if round(fisher_stat, _KEY_DP) > bound:
            break
    fisher_p = 1 - gammainc(len(predictor_grams), 0.5 * fisher_stat)
    return float(fisher_p), float(fisher_stat)


# ---- bottom-up causal-order search --------------------------------------------------------------


def _residual(X: np.ndarray, cov: np.ndarray, predictors: np.ndarray, target: int) -> np.ndarray:
    """Least-squares residual of ``target`` on ``predictors``, solved through the covariance.

    :func:`find_causal_order` passes unit-scale columns (:func:`~andrey.core.units.unit_factors`),
    so the pseudo-inverse's cutoff, relative to the largest singular value, does not read the
    predictors' units. The residual is then in the scaled target's units, which the HSIC test, on
    a median-distance bandwidth, does not read.
    """
    coef = np.dot(
        np.linalg.pinv(cov[np.ix_(predictors, predictors)]),
        cov[predictors, target].reshape(predictors.shape[0], 1),  # cov symmetric: [pred, target]
    )
    return X[:, [target]] - np.dot(X[:, predictors], coef)


def _find_sink(
    X: np.ndarray,
    cov: np.ndarray,
    U: np.ndarray,
    col_grams: list[_Gram],
    n: int,
    ranks: list[int],
    candidates: np.ndarray | None = None,
) -> tuple[int, float]:
    """Return the most-sink variable in ``U`` and its Fisher p-value.

    Each candidate is regressed on the others; the candidate whose residual is most independent of
    the others (smallest Fisher statistic, so largest p-value under the monotone gamma /
    chi-squared tail) is the sink. Statistics equal to ``_KEY_DP`` decimals go to the lower of
    ``ranks``, each variable's :func:`~andrey.search.hc.tie_ranks`. ``col_grams`` holds each
    variable's cached HSIC gram (keyed by variable index). ``candidates``, a subset of ``U``,
    limits the sinks considered; each is still regressed on all the others in ``U``.
    """
    pool = U if candidates is None else candidates
    best: tuple[float, int] | None = None
    sink, best_p = int(pool[0]), -np.inf
    for j in range(len(pool)):
        predictors = np.setdiff1d(U, pool[j])
        residual = _residual(X, cov, predictors, int(pool[j]))
        bound = np.inf if best is None else best[0]
        fisher_p, fisher_stat = _fisher_hsic_test(
            [col_grams[p] for p in predictors], residual, n, bound
        )
        key = (round(fisher_stat, _KEY_DP), ranks[int(pool[j])])
        if best is None or key < best:
            best, sink, best_p = key, int(pool[j]), fisher_p
    return sink, best_p


def find_causal_order(
    X: np.ndarray, *, alpha: float = 0.1, precedence: np.ndarray | None = None
) -> list:
    """Estimate a (possibly partial) nested causal order of centered ``X`` from the sink upward.

    Peels the most-sink variable while its Fisher HSIC p-value clears the Bonferroni threshold
    ``alpha / (d - 1)``; when the test is first rejected the remaining variables stay unresolved.
    The returned order is source-first with any order-unknown block (more than one leftover) nested
    as a front sub-list in ``causal_order_``. With ``precedence`` (``precedence[a, b]``: ``a``
    comes before ``b``), a variable is a sink candidate only when no remaining variable must come
    after it. The residuals are taken on unit-scale columns (:func:`_residual`).
    """
    d = X.shape[1]
    n = X.shape[0]
    unit = X * unit_factors(X)
    cov = np.cov(unit, rowvar=False)
    col_grams = [gamma_gram(X[:, [j]]) for j in range(d)]  # per-variable HSIC gram, cached once
    ranks = tie_ranks(X).tolist()
    U = np.arange(d)
    resolved_sinks: list[int] = []  # most-sink first

    if d > 1:
        threshold = alpha / (d - 1)
        while len(U) > 1:
            candidates = None
            if precedence is not None:
                candidates = U[~precedence[np.ix_(U, U)].any(axis=1)]
            sink, fisher_p = _find_sink(unit, cov, U, col_grams, n, ranks, candidates)
            if fisher_p < threshold:
                break
            resolved_sinks.append(sink)
            U = U[U != sink]

    bottom_up = list(reversed(resolved_sinks))  # source-first among the resolved
    unresolved = [int(v) for v in np.setdiff1d(np.arange(d), bottom_up)]
    order: list = [unresolved] if len(unresolved) > 1 else list(unresolved)
    order.extend(bottom_up)
    return order


# ---- adjacency estimation -----------------------------------------------------------------------


def _flatten(order: list) -> list[int]:
    """Flatten a nested causal order into a single list of variable indices."""
    flat: list[int] = []
    for item in order:
        if isinstance(item, list):
            flat.extend(_flatten(item))
        else:
            flat.append(int(item))
    return flat


def estimate_adjacency(
    X: np.ndarray, order: list, forbidden: np.ndarray | None = None
) -> np.ndarray:
    """Weighted adjacency ``B`` from a nested causal order: each vertex on its resolved parents.

    ``B[i, j]`` weights edge ``j -> i``, fitted by adaptive Lasso; an order-unknown block adds no
    directed edges and every pair inside it is set to ``NaN`` at both ``B[i, j]`` and ``B[j, i]``.
    Each fit lists its predictors by :func:`~andrey.search.hc.tie_ranks`, so the weights follow the
    columns when they are reordered. ``forbidden[j, i]`` drops predictor ``j`` of ``i``, and a pair
    of the block forbidden both ways stays ``0``.
    """
    d = X.shape[1]
    ranks = tie_ranks(X).tolist()
    B = np.zeros((d, d), dtype=np.float64)
    for position in range(1, len(order)):
        target = order[position]
        predictors = sorted(_flatten(order[:position]), key=ranks.__getitem__)
        if forbidden is not None:
            predictors = [j for j in predictors if not forbidden[j, target]]
        if not predictors:
            continue
        B[target, predictors] = adjacency.predict_adaptive_lasso(X, predictors, target)

    for block in order:
        if isinstance(block, list):
            for pos in range(len(block)):
                for other in range(pos + 1, len(block)):
                    xi, xj = int(block[pos]), int(block[other])
                    if forbidden is not None and forbidden[xi, xj] and forbidden[xj, xi]:
                        continue
                    B[xi, xj] = np.nan
                    B[xj, xi] = np.nan
    return B


def bottom_up_parce_lingam(
    X: np.ndarray, *, alpha: float = 0.1, constraints: Constraints | None = None
) -> BottomUpParceLiNGAMResult:
    """Fit BottomUpParceLiNGAM: a nested causal order plus a ``NaN``-flagged weighted adjacency.

    ``alpha`` is the significance level of the bottom-up independence test (Bonferroni-corrected
    across the ``d - 1`` predictors). Returns a :class:`BottomUpParceLiNGAMResult` exposing
    ``causal_order_`` and ``adjacency_matrix_``. With background knowledge ``constraints`` (no
    required edges), the order keeps their tiers (:func:`find_causal_order`) and the regressions
    drop the parents they forbid (:func:`estimate_adjacency`).
    """
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0)  # center before both the search and regression
    if constraints is None:
        order = find_causal_order(X, alpha=alpha)
        B = estimate_adjacency(X, order)
    else:
        order = find_causal_order(X, alpha=alpha, precedence=constraints.precedence())
        B = estimate_adjacency(X, order, constraints.forbidden)
    return BottomUpParceLiNGAMResult(causal_order_=order, adjacency_matrix_=B)
