"""Prune a candidate DAG by significance tests of additive-model smooth terms.

Each node with candidate parents is fit as an additive model of them
(:func:`andrey.fcm.gam.fit_gam`), and a parent stays when its smooth term's p-value is at most
``cutoff`` (Buhlmann, Peters, Ernest 2014, Section 5.3). The paper declares significance at
p-values "lower or equal" to 0.001 at every sample size and suggests a larger cutoff for small
samples. CAM prunes the DAG of its order search with this step.

References
----------
Buhlmann, Peters, Ernest. "CAM: Causal additive models, high-dimensional order search and
penalized regression." Annals of Statistics 42(6):2526-2556, 2014.
"""

from __future__ import annotations

import logging

import numpy as np

from andrey.fcm.gam import BasisCache, fit_gam

_log = logging.getLogger(__name__)


def prune(
    X: np.ndarray,
    adjacency: np.ndarray,
    *,
    cutoff: float = 0.001,
    k: int = 10,
    bases: BasisCache | None = None,
) -> np.ndarray:
    """Keep the edges of ``adjacency`` whose smooth terms are significant.

    Parameters
    ----------
    X : np.ndarray
        ``(n, d)`` data matrix.
    adjacency : np.ndarray
        ``(d, d)`` candidate DAG; a nonzero ``adjacency[i, j]`` is the edge ``i -> j``.
    cutoff : float, default 0.001
        An edge stays when its p-value is at most ``cutoff``, the paper's "lower or equal".
    k : int, default 10
        Basis rank of each smooth term before :func:`andrey.fcm.gam.effective_k` adjusts it.
    bases : BasisCache or None, default None
        Bases of ``X``'s columns to reuse; ``None`` builds them.

    Returns
    -------
    np.ndarray
        ``(d, d)`` 0/1 integer adjacency of the kept edges, in the same orientation.

    Raises
    ------
    ValueError
        If the shapes disagree, ``adjacency`` has a self-loop, or ``cutoff`` is outside ``[0, 1]``.
    """
    X = np.asarray(X, dtype=np.float64)
    A = np.asarray(adjacency)
    d = X.shape[1] if X.ndim == 2 else -1
    if A.shape != (d, d):
        raise ValueError(f"adjacency must be ({d}, {d}) for {X.shape} data, got {A.shape}")
    if np.any(np.diag(A) != 0):
        raise ValueError("adjacency has a self-loop")
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError(f"cutoff must be in [0, 1], got {cutoff}")
    if bases is None:
        bases = BasisCache(X)
    elif bases.data.shape != X.shape or not np.array_equal(bases.data, X):
        raise ValueError("bases were built for different data")
    kept = np.zeros((d, d), dtype=np.int64)
    for child in range(d):
        parents = np.flatnonzero(A[:, child])
        if parents.size == 0:
            continue
        p_values = fit_gam(bases, child, parents, k=k).p_values()
        kept[parents[p_values <= cutoff], child] = 1
        _log.debug("node %d: parents %s, p-values %s", child, parents.tolist(), p_values)
    return kept
