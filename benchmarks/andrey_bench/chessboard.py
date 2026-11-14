"""The chessboard network of the MB-by-MB paper's simulation, with binary variables.

The paper (Wang et al., 2014, Sect. 5.2) lays 2025 variables on a 45 x 45 board and directs every
edge from a square to the square below it and to the square to its right, so a square has at most
two parents and two children and the DAG has 3960 edges. It says only that the conditional
probability tables are drawn at random from faithful values; the tables here are this lane's
choice:

- ``P(X = 1 | parents) = logistic(b + sum_k w_k s_k)``, where ``s_k`` is ``+1`` or ``-1`` for the
  parent's state, ``b`` is drawn from ``U(-0.5, 0.5)``, and each weight ``w_k`` is
  ``U(0.8, 1.6)`` with a random sign;
- the tables are a function of the seed alone, as the paper draws them once per simulation and
  samples from them at each size, and the rows are a function of the seed and the sample size;
- the paper's targets are 10 squares drawn once, from the generator ``numpy.random.default_rng``
  seeded with :data:`TARGET_SEED`.

A node's index is ``row * side + column``, which is a topological order: parents come first.
:data:`SAMPLER` names this generator; a change to the tables or the rows is a new name, so a
run's records say which one drew them.

References
----------
Wang, C., Zhou, Y., Zhao, Q., and Geng, Z. (2014). Discovering and orienting the edges connected to
a target variable in a DAG via a sequential local learning approach. Computational Statistics and
Data Analysis 77, 252-266.
"""

from __future__ import annotations

import numpy as np

#: The board's side in the paper: 45 x 45 = 2025 variables.
SIDE = 45

#: The number of targets the paper answers for on the board.
N_TARGETS = 10

#: The seed of the generator that draws the targets.
TARGET_SEED = 2014

#: The name of this generator of tables and rows.
SAMPLER = "chessboard-v1"


def adjacency(side: int = SIDE) -> np.ndarray:
    """The board's DAG: ``adjacency[i, j] = 1`` for the edge ``i -> j``, ``int8``.

    Every square points to the square below it and to the square to its right.
    """
    d = side * side
    adj = np.zeros((d, d), dtype=np.int8)
    for row in range(side):
        for column in range(side):
            k = row * side + column
            if row + 1 < side:
                adj[k, k + side] = 1
            if column + 1 < side:
                adj[k, k + 1] = 1
    return adj


def labels(side: int = SIDE) -> list[str]:
    """The squares' names, ``r<row>c<column>``, in node order."""
    return [f"r{k // side}c{k % side}" for k in range(side * side)]


def targets(side: int = SIDE) -> list[int]:
    """The ``N_TARGETS`` squares the paper answers for, drawn once, ascending."""
    d = side * side
    return sorted(np.random.default_rng(TARGET_SEED).choice(d, N_TARGETS, replace=False).tolist())


def sample(adj: np.ndarray, n: int, seed: int) -> np.ndarray:
    """``n`` rows of ``0``/``1`` values of the board ``adj``, as ``float64``.

    The tables come from ``seed``, the rows from ``(seed, n)``, so a larger sample is a new draw
    from the same tables.
    """
    d = adj.shape[0]
    tables = np.random.default_rng([seed, 4545])
    bias = tables.uniform(-0.5, 0.5, d)
    weight = tables.uniform(0.8, 1.6, (d, d)) * tables.choice([-1.0, 1.0], (d, d))
    rows = np.random.default_rng([seed, n])
    data = np.zeros((n, d))
    for k in range(d):  # node order is topological
        parents = np.flatnonzero(adj[:, k])
        logit = bias[k] + sum(weight[p, k] * (2 * data[:, p] - 1) for p in parents)
        data[:, k] = rows.random(n) < 1 / (1 + np.exp(-logit))
    return data
