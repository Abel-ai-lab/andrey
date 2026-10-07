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

import numpy as np
from scipy.special import gammainc

from andrey.core.independence import gamma_gram, hsic_gamma_from_grams
from andrey.lingam import adjacency

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
    predictor_grams: list[_Gram], residual: np.ndarray, n: int, best_stat: float
) -> tuple[float, float]:
    """Aggregate the per-predictor LiNGAM HSIC tests of ``residual`` by Fisher's method.

    Each predictor's independence from ``residual`` is scored by the median-bandwidth gamma HSIC in
    :mod:`andrey.core.independence`. The predictor grams are cached (they are fixed data columns);
    only the residual's gram is built here, once, and reused across predictors. Returns
    ``(fisher_p, fisher_stat)``. One predictor short-circuits to its HSIC ``(p, stat)``. Otherwise
    ``fisher_stat = sum_k -2 log(p_k)`` and ``fisher_p`` is its upper chi-squared tail on
    ``2 * n_predictors`` degrees of freedom (a zero HSIC p-value sends ``fisher_stat`` to infinity).

    ``best_stat`` is the smallest Fisher statistic seen so far in the current sink search. Because
    the running sum only grows (each ``-2 log(p_k) >= 0``), once it passes ``best_stat`` this
    candidate can no longer be the most-sink one, so the remaining predictor tests are skipped. The
    partial ``fisher_stat`` and its ``fisher_p`` then miss the incumbent and never win the
    comparison in :func:`_find_sink`, leaving the selected sink and its p-value unchanged.
    """
    residual_gram = gamma_gram(residual)
    if len(predictor_grams) == 1:
        statistic, p_value = hsic_gamma_from_grams(predictor_grams[0], residual_gram, n)
        return p_value, statistic

    fisher_stat = 0.0
    for gram in predictor_grams:
        _, p_value = hsic_gamma_from_grams(gram, residual_gram, n)
        fisher_stat += np.inf if p_value == 0 else -2 * np.log(p_value)
        if fisher_stat > best_stat:
            break
    fisher_p = 1 - gammainc(len(predictor_grams), 0.5 * fisher_stat)
    return float(fisher_p), float(fisher_stat)


# ---- bottom-up causal-order search --------------------------------------------------------------


def _residual(X: np.ndarray, cov: np.ndarray, predictors: np.ndarray, target: int) -> np.ndarray:
    """Least-squares residual of ``target`` on ``predictors``, solved through the covariance."""
    coef = np.dot(
        np.linalg.pinv(cov[np.ix_(predictors, predictors)]),
        cov[predictors, target].reshape(predictors.shape[0], 1),  # cov symmetric: [pred, target]
    )
    return X[:, [target]] - np.dot(X[:, predictors], coef)


def _find_sink(
    X: np.ndarray, cov: np.ndarray, U: np.ndarray, col_grams: list[_Gram], n: int
) -> tuple[int, float]:
    """Return the most-sink variable in ``U`` and its Fisher p-value.

    Each candidate is regressed on the others; the candidate whose residual is most independent of
    the others (smallest Fisher statistic) is the sink. Ties resolve to the smaller statistic, which
    is equivalent to the larger p-value under the monotone gamma / chi-squared tail. ``col_grams``
    holds each variable's cached HSIC gram (keyed by variable index).
    """
    best_stat = np.inf
    best_p = -np.inf
    sink_predictors = np.array([], dtype=np.int64)
    for j in range(len(U)):
        predictors = np.setdiff1d(U, U[j])
        residual = _residual(X, cov, predictors, int(U[j]))
        fisher_p, fisher_stat = _fisher_hsic_test(
            [col_grams[p] for p in predictors], residual, n, best_stat
        )
        if fisher_stat < best_stat or fisher_p > best_p:
            sink_predictors = predictors
            best_stat = fisher_stat
            best_p = fisher_p
    sink = int(np.setdiff1d(U, sink_predictors)[0])
    return sink, best_p


def find_causal_order(X: np.ndarray, *, alpha: float = 0.1) -> list:
    """Estimate a (possibly partial) nested causal order of centered ``X`` from the sink upward.

    Peels the most-sink variable while its Fisher HSIC p-value clears the Bonferroni threshold
    ``alpha / (d - 1)``; when the test is first rejected the remaining variables stay unresolved.
    The returned order is source-first with any order-unknown block (more than one leftover) nested
    as a front sub-list in ``causal_order_``.
    """
    d = X.shape[1]
    n = X.shape[0]
    cov = np.cov(X, rowvar=False)
    col_grams = [gamma_gram(X[:, [j]]) for j in range(d)]  # per-variable HSIC gram, cached once
    U = np.arange(d)
    resolved_sinks: list[int] = []  # most-sink first

    if d > 1:
        threshold = alpha / (d - 1)
        while len(U) > 1:
            sink, fisher_p = _find_sink(X, cov, U, col_grams, n)
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


def estimate_adjacency(X: np.ndarray, order: list) -> np.ndarray:
    """Weighted adjacency ``B`` from a nested causal order: each vertex on its resolved parents.

    ``B[i, j]`` weights edge ``j -> i``, fitted by adaptive Lasso; an order-unknown block adds no
    directed edges and every pair inside it is set to ``NaN`` at both ``B[i, j]`` and ``B[j, i]``.
    """
    d = X.shape[1]
    B = np.zeros((d, d), dtype=np.float64)
    for position in range(1, len(order)):
        target = order[position]
        predictors = _flatten(order[:position])
        if not predictors:
            continue
        B[target, predictors] = adjacency.predict_adaptive_lasso(X, predictors, target)

    for block in order:
        if isinstance(block, list):
            for pos in range(len(block)):
                for other in range(pos + 1, len(block)):
                    xi, xj = int(block[pos]), int(block[other])
                    B[xi, xj] = np.nan
                    B[xj, xi] = np.nan
    return B


def bottom_up_parce_lingam(X: np.ndarray, *, alpha: float = 0.1) -> BottomUpParceLiNGAMResult:
    """Fit BottomUpParceLiNGAM: a nested causal order plus a ``NaN``-flagged weighted adjacency.

    ``alpha`` is the significance level of the bottom-up independence test (Bonferroni-corrected
    across the ``d - 1`` predictors). Returns a :class:`BottomUpParceLiNGAMResult` exposing
    ``causal_order_`` and ``adjacency_matrix_``.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0)  # center before both the search and regression
    order = find_causal_order(X, alpha=alpha)
    B = estimate_adjacency(X, order)
    return BottomUpParceLiNGAMResult(causal_order_=order, adjacency_matrix_=B)
