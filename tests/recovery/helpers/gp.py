"""The simulation of the SCORE and NoGAM papers, at test size.

Rolland et al. (2022, Section 4.1) and Montagna et al. (2023, Section 6) draw a nonlinear additive
noise model ``x_j = f_j(parents) + e_j`` on an Erdos-Renyi graph: each ``f_j`` is one draw of a
Gaussian process over the vector of ``j``'s parents, with the RBF kernel
``exp(-||u - u'||^2 / 2)``, and each noise has a variance drawn uniformly from ``[0.4, 0.8]``.
"""

from __future__ import annotations

import numpy as np

from andrey.api._adapt import dag_from_adjacency
from andrey.core import GraphStructure

#: Added to the kernel's diagonal before its Cholesky factorization.
JITTER = 1e-6


def sample(
    d: int, n: int, n_edges: int, seed: int, *, noise: str = "gaussian"
) -> tuple[np.ndarray, GraphStructure]:
    """Draw ``n`` rows over ``d`` variables with ``n_edges`` edges; return ``(data, dag)``.

    The columns come in an order drawn from the seed, not the causal one. ``noise`` is
    ``"gaussian"`` or ``"laplace"``, each scaled to its drawn variance.
    """
    rng = np.random.default_rng(seed)
    causal = rng.permutation(d)
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    chosen = rng.choice(len(pairs), size=n_edges, replace=False)
    edges = {(int(causal[pairs[c][0]]), int(causal[pairs[c][1]])) for c in chosen}
    sd = np.sqrt(rng.uniform(0.4, 0.8, d))
    X = np.zeros((n, d))
    for j in causal.tolist():
        unit = rng.standard_normal(n) if noise == "gaussian" else rng.laplace(0, 2**-0.5, n)
        X[:, j] = sd[j] * unit
        parents = sorted(i for i, k in edges if k == j)
        if parents:
            u = X[:, parents]
            sq = np.sum(u * u, axis=1)
            cov = np.exp(-np.maximum(sq[:, None] + sq[None, :] - 2.0 * (u @ u.T), 0.0) / 2.0)
            cov[np.diag_indices(n)] += JITTER
            X[:, j] += np.linalg.cholesky(cov) @ rng.standard_normal(n)
    adjacency = np.zeros((d, d), dtype=bool)
    adjacency[tuple(zip(*edges, strict=True))] = True
    return X, dag_from_adjacency(adjacency)
