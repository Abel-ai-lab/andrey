"""Sortability diagnostics: avoid unexpected leaks from synthetic DGP.

The module implements two sortability metrics:

- **varsortability** (Reisach et al. 2021, "Beware of the Simulated DAG!"): the fraction of directed
  paths ordered by increasing marginal *variance*.
- **r2sortability** (Reisach et al. 2023): the same, by increasing coefficient of determination
  (variance explained by all other variables). It is **scale-invariant**.

The full all-paths estimators use dense adjacency powers (``O(d^3)``), so they are for
small/moderate ``d``; :func:`varsortability_edges` is the
``O(nnz)`` edge-only variant for reporting at scale.
"""

from __future__ import annotations

import numpy as np


def directed_adjacency(edges: np.ndarray, d: int) -> np.ndarray:
    """Build a dense ``(d, d)`` binary adjacency, ``A[i, j] = 1`` iff edge ``i -> j``.

    Here ``i`` is the cause, ``j`` the effect. Dense by construction (the reference sortability
    estimators need it); use only for small/moderate ``d``. ``edges`` is an ``(m, 2)``
    ``(parent, child)`` array.
    """
    a = np.zeros((d, d), dtype=np.int8)
    if edges.shape[0]:
        a[edges[:, 0], edges[:, 1]] = 1
    return a


def varsortability(x: np.ndarray, adjacency: np.ndarray, *, tol: float = 1e-9) -> float:
    """Varsortability of data ``x`` under directed ``adjacency`` (Reisach et al. 2021), all paths.

    Matches the reference (``github.com/Scriddie/Varsortability``): counts, over directed paths of
    every length, the fraction ordered by increasing marginal variance; ties (variance ratio within
    ``tol`` of 1) score ``0.5``.

    Parameters
    ----------
    x : np.ndarray of shape (n, d)
        Data, one row per sample.
    adjacency : np.ndarray of shape (d, d)
        ``adjacency[i, j] != 0`` iff edge ``i -> j``.
    tol : float, default=1e-9
        Variance-ratio tie tolerance (the reference default; pinned so exact-0.5 results are the
        claim, not an accident).

    Returns
    -------
    float
        Varsortability in ``[0, 1]``, or ``nan`` for a path-free graph (zero denominator).
    """
    e = np.asarray(adjacency) != 0
    d = e.shape[0]
    var = np.var(x, axis=0, keepdims=True)  # (1, d), ddof=0 (reference convention)
    ek = e.copy()
    n_paths = 0.0
    n_correct = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_full = var / var.T  # (d, d): [i, j] = Var(X_j) / Var(X_i)
        for _ in range(d - 1):
            n_paths += ek.sum()
            gated = ek * ratio_full
            n_correct += (gated > 1 + tol).sum()
            n_correct += 0.5 * ((gated <= 1 + tol) & (gated > 1 - tol) & ek).sum()
            ek = ek.dot(e)
            if not ek.any():  # no paths of this length; every later power is zero too
                break
    return float(n_correct / n_paths) if n_paths else float("nan")


def r2sortability(x: np.ndarray, adjacency: np.ndarray, *, tol: float = 0.0) -> float:
    """R^2-sortability of ``x`` under directed ``adjacency`` (Reisach et al. 2023), all paths.

    Matches ``github.com/CausalDisco/CausalDisco`` (``analytics.py``) for ``n > d``: each node's
    score is ``1 - 1/diag(inv(corrcoef(x)))`` (variance explained by all other variables); counts
    directed paths ordered by increasing score. R^2 is derived from the correlation matrix, so this
    is **unchanged by column standardization** -- the caveat varsortability cannot see.

    For ``n <= d`` the correlation matrix is singular and this returns ``nan`` (a reported skip),
    rather than switching to a different estimator (CausalDisco falls back to per-variable
    regression) -- so R^2-sortability stays one well-defined quantity or is explicitly undefined.

    Returns
    -------
    float
        R^2-sortability in ``[0, 1]``; ``nan`` if ``n <= d`` or the graph is path-free.
    """
    x = np.asarray(x)
    n, d = x.shape
    if n <= d:
        return float("nan")  # singular correlation matrix -> R^2 undefined; never return garbage
    corr = np.corrcoef(x, rowvar=False)
    try:
        scores = 1.0 - 1.0 / np.diag(np.linalg.inv(corr))
    except np.linalg.LinAlgError:
        return float("nan")
    e = np.asarray(adjacency) != 0
    scores = scores.reshape(1, -1)
    diff = scores - scores.T  # (d, d): [i, j] = score_j - score_i
    ek = e.copy()
    n_paths = 0.0
    n_correct = 0.0
    for _ in range(e.shape[0] - 1):
        n_paths += ek.sum()
        n_correct += (ek & (diff >= -tol)).sum() / 2.0
        n_correct += (ek & (diff > tol)).sum() / 2.0
        ek = ek.dot(e)
        if not ek.any():
            break
    return float(n_correct / n_paths) if n_paths else float("nan")


def varsortability_edges(x: np.ndarray, edges: np.ndarray, *, tol: float = 1e-9) -> float:
    """Edge-only varsortability (adjacent pairs), ``O(nnz)`` -- scalable variant for large ``d``.

    A documented deviation from the all-paths reference: counts only direct edges, so it never
    materializes a dense ``(d, d)`` adjacency. Comparable across graph sizes; use for reporting at
    ``d`` in the thousands.

    Parameters
    ----------
    x : np.ndarray of shape (n, d)
        Data.
    edges : np.ndarray of shape (m, 2)
        Directed edges ``(parent, child)``.

    Returns
    -------
    float
        Edge-only varsortability in ``[0, 1]``, or ``nan`` for an edgeless graph.
    """
    if edges.shape[0] == 0:
        return float("nan")
    var = np.var(x, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = var[edges[:, 1]] / var[edges[:, 0]]  # Var(child) / Var(parent)
    correct = float((ratio > 1 + tol).sum())
    correct += 0.5 * float(((ratio <= 1 + tol) & (ratio > 1 - tol)).sum())
    return correct / edges.shape[0]


def standardize(x: np.ndarray) -> np.ndarray:
    """Z-score each column to zero mean, unit variance (drives varsortability to exactly 0.5).

    This removes varsortability but leaves :func:`r2sortability` **unchanged** -- standardized
    synthetic data is *not* scale-hard.
    """
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)  # leave constant columns unscaled (no divide-by-zero)
    return (x - x.mean(axis=0)) / sd
