"""Sortability diagnostics: avoid unexpected leaks from synthetic DGP.

The module implements two sortability metrics and a column-order check:

- **varsortability** (Reisach et al. 2021, "Beware of the Simulated DAG!"): the fraction of directed
  paths ordered by increasing marginal *variance*.
- **r2sortability** (Reisach et al. 2023): the same, by increasing coefficient of determination
  (variance explained by all other variables). It is **scale-invariant**.
- **index_correlation**: the rank correlation between column index and causal depth, which is near
  zero when the column order says nothing about the causal order.

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


def causal_depth(edges: np.ndarray, d: int) -> np.ndarray:
    """Return each node's causal depth: the edge count of the longest directed path into it.

    Roots have depth 0. The depth is a property of the graph, not of how its nodes are numbered, so
    it ranks the nodes in causal order under any column order.

    Parameters
    ----------
    edges : np.ndarray of shape (m, 2)
        Directed edges ``(parent, child)`` of a DAG over the nodes ``0 .. d-1``.
    d : int
        Number of nodes.

    Returns
    -------
    np.ndarray of shape (d,)
        The depth of each node.

    Raises
    ------
    ValueError
        If the edges form a directed cycle.
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    waiting = np.bincount(edges[:, 1], minlength=d)
    children: list[list[int]] = [[] for _ in range(d)]
    for parent, child in edges.tolist():
        children[parent].append(child)
    depth = np.zeros(d, dtype=np.int64)
    ready = np.flatnonzero(waiting == 0).tolist()
    done = 0
    while ready:
        node = ready.pop()
        done += 1
        for child in children[node]:
            depth[child] = max(depth[child], depth[node] + 1)
            waiting[child] -= 1
            if waiting[child] == 0:
                ready.append(child)
    if done != d:
        raise ValueError("the edges form a directed cycle")
    return depth


def index_correlation(edges: np.ndarray, d: int, *, n_observed: int | None = None) -> float:
    """Spearman's rank correlation between each column's index and its :func:`causal_depth`.

    How much a dataset's column order reveals of its causal order: ``1`` when the columns come in
    order of depth, ``-1`` in the reverse order, and near ``0`` when the order is unrelated to the
    graph. Only the first ``n_observed`` nodes are columns; hidden nodes after them count toward
    the depths.

    Parameters
    ----------
    edges : np.ndarray of shape (m, 2)
        Directed edges ``(parent, child)`` of a DAG over the nodes ``0 .. d-1``.
    d : int
        Number of nodes.
    n_observed : int or None, default=None
        Number of columns, the first nodes; ``None`` means every node is a column.

    Returns
    -------
    float
        Spearman's rho over the columns, with tied depths given their mean rank; ``nan`` when every
        column has the same depth.

    Notes
    -----
    Under a uniformly random column order, the correlation of one dataset with ``k`` columns has
    mean 0 and variance exactly ``1 / (k - 1)``, ties included: it is the permutation distribution
    of a Pearson correlation with a fixed vector. The mean over ``S`` independently shuffled
    datasets with ``k_1 .. k_S`` columns therefore has standard deviation
    ``sqrt(sum_s 1 / (k_s - 1)) / S``.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey.data.sortability import index_correlation
    >>> index_correlation(np.array([[0, 1], [1, 2]]), 3)  # the chain 0 -> 1 -> 2
    1.0
    >>> index_correlation(np.array([[2, 1], [1, 0]]), 3)  # the chain 2 -> 1 -> 0
    -1.0
    """
    k = d if n_observed is None else n_observed
    depth = causal_depth(edges, d)[:k].astype(np.float64)
    if k < 2 or np.all(depth == depth[0]):
        return float("nan")
    values, inverse = np.unique(depth, return_inverse=True)
    below = np.concatenate([[0], np.cumsum(np.bincount(inverse))])
    ranks = ((below[:-1] + below[1:] - 1) / 2)[inverse]  # each depth's mean rank
    index = np.arange(k, dtype=np.float64)
    return float(np.corrcoef(index, ranks)[0, 1])


def standardize(x: np.ndarray) -> np.ndarray:
    """Z-score each column to zero mean, unit variance (drives varsortability to exactly 0.5).

    This removes varsortability but leaves :func:`r2sortability` **unchanged** -- standardized
    synthetic data is *not* scale-hard.
    """
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)  # leave constant columns unscaled (no divide-by-zero)
    return (x - x.mean(axis=0)) / sd
