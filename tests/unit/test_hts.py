"""LHTS, NHTS, and ED under exact answers from a known model.

Each search runs on a tests object whose answers come from the true model: d-separation for the
marginal and conditional tests, the noise terms for LHTS's linear residuals (two linear forms of
independent non-Gaussian noises are independent exactly when no noise enters both, the
Darmois-Skitovich theorem), and, for NHTS's kernel residuals, the rule that the residual of ``x_j``
on a set ``S`` is independent of each member of ``S`` exactly when ``S`` holds every parent of
``x_j`` and no descendant: true of the paper's mechanism ``(sum_k w_k x_k)^2``, in which no parent
enters additively. Under these answers every layer must be the true one and ED must return the
true parents.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from andrey.data.latent import DSepOracle
from andrey.fcm.hts import edge_discovery, lhts_sort, nhts_sort, total_order

INDEPENDENT, DEPENDENT = 0.0, -math.inf


def dsep(adj: np.ndarray) -> DSepOracle:
    """The d-separation oracle of the DAG with adjacency ``adj`` (``[i, j]`` for ``i -> j``)."""
    return DSepOracle(nx.from_numpy_array(adj.astype(int), create_using=nx.DiGraph))


def random_dag(d: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """A DAG over ``d`` nodes, each edge of a random order present with probability ``p``."""
    order = rng.permutation(d)
    adj = np.zeros((d, d), dtype=bool)
    for a in range(d):
        for b in range(a + 1, d):
            adj[order[a], order[b]] = rng.random() < p
    return adj


def true_layers(adj: np.ndarray) -> list[list[int]]:
    """Each node's longest directed path from a root, as layers."""
    d = adj.shape[0]
    depth = np.zeros(d, dtype=int)
    for _ in range(d):
        for a, b in zip(*np.nonzero(adj), strict=True):
            depth[b] = max(depth[b], depth[a] + 1)
    return [sorted(np.flatnonzero(depth == k).tolist()) for k in range(depth.max() + 1)]


def descendants(adj: np.ndarray) -> np.ndarray:
    """``[i, j]``: ``j`` is a descendant of ``i``."""
    reach = adj.copy()
    for k in range(adj.shape[0]):
        reach |= np.outer(reach[:, k], reach[k])
    return reach


class PopulationLinear:
    """LHTS's operations in a linear model: a variable is its row of noise coefficients."""

    def __init__(self, weights: np.ndarray, noise_var: np.ndarray) -> None:
        d = weights.shape[0]
        self.mix = np.linalg.inv(np.eye(d) - weights.T)
        self.var = noise_var

    def column(self, i: int) -> np.ndarray:
        return self.mix[i]

    def _project(self, y: np.ndarray, regressors: list[np.ndarray]) -> np.ndarray:
        R = np.array(regressors)
        beta = np.linalg.solve(R * self.var @ R.T, R * self.var @ y)
        return y - R.T @ beta

    def residual(self, y: np.ndarray, regressors) -> np.ndarray:
        if not regressors:
            return y
        return self._project(y, list(regressors))

    def log_p(self, a: np.ndarray, b: np.ndarray) -> float:
        tol = 1e-9 * max(np.abs(a).max(), np.abs(b).max(), 1.0)
        shared = (np.abs(a) > tol) & (np.abs(b) > tol)
        return DEPENDENT if shared.any() else INDEPENDENT


class GraphNonlinear:
    """NHTS's questions in a nonlinear model with no additive parent, answered from its DAG."""

    def __init__(self, adj: np.ndarray) -> None:
        self.adj = adj
        self.below = descendants(adj)
        self.dsep = dsep(adj)

    def _clean(self, j: int, regressors) -> bool:
        S = set(regressors)
        return set(np.flatnonzero(self.adj[:, j])) <= S and not any(self.below[j, s] for s in S)

    def marginal(self, i: int, j: int) -> float:
        return INDEPENDENT if self.dsep(i, j, []) else DEPENDENT

    def parent(self, i: int, j: int, regressors) -> float:
        return INDEPENDENT if self._clean(j, regressors) else DEPENDENT

    def layer(self, i: int, done) -> list[float]:
        return [INDEPENDENT if self._clean(i, done) else DEPENDENT] * len(done)

    def conditional(self, x: int, y: int, given) -> float:
        return INDEPENDENT if self.dsep(x, y, given) else DEPENDENT


def linear_model(adj: np.ndarray, rng: np.random.Generator) -> PopulationLinear:
    weights = adj * rng.choice([-1.0, 1.0], adj.shape) * rng.uniform(0.5, 1.5, adj.shape)
    return PopulationLinear(weights, rng.uniform(0.5, 2.0, adj.shape[0]))


#: The walk-through DAG of the paper's Appendices A.11 and B.8: A -> B, A -> C, B -> D, C -> D,
#: D -> E.
WALK = np.zeros((5, 5), dtype=bool)
for _a, _b in ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4)):
    WALK[_a, _b] = True


def test_lhts_walk_through_layers() -> None:
    model = linear_model(WALK, np.random.default_rng(0))
    result = lhts_sort(model, 5, 0.05)
    assert result.layers == [[0], [1, 2], [3], [4]]
    assert result.forced == 0


def test_nhts_walk_through_layers() -> None:
    result = nhts_sort(GraphNonlinear(WALK), 5, 0.05)
    assert result.layers == [[0], [1, 2], [3], [4]]
    assert result.forced == 0


def test_ed_walk_through_parents() -> None:
    parents = edge_discovery(true_layers(WALK), dsep(WALK), 5, 0.05)
    assert (parents == WALK).all()


def test_ed_tests_a_candidate_again_given_the_other_candidates() -> None:
    """a -> m <- q, m -> b, q -> b: a and q share layer 0, and given m alone a and b are
    dependent through q, so a -> b is dropped only when tested again given m and q."""
    adj = np.zeros((4, 4), dtype=bool)
    a, q, m, b = range(4)
    adj[a, m] = adj[q, m] = adj[m, b] = adj[q, b] = True
    parents = edge_discovery(true_layers(adj), dsep(adj), 4, 0.05)
    assert (parents == adj).all()


@pytest.mark.parametrize("seed", range(4))
def test_oracle_sorts_and_parents_are_true_on_random_dags(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(50):
        d = int(rng.integers(2, 10))
        adj = random_dag(d, float(rng.uniform(0.15, 0.6)), rng)
        layers = true_layers(adj)
        linear = lhts_sort(linear_model(adj, rng), d, 0.05)
        assert linear.layers == layers
        assert linear.forced == 0
        nonlinear = nhts_sort(GraphNonlinear(adj), d, 0.05)
        assert nonlinear.layers == layers
        assert nonlinear.forced == 0
        assert (edge_discovery(layers, dsep(adj), d, 0.05) == adj).all()
        position = {v: k for k, v in enumerate(total_order(layers))}
        assert all(position[a] < position[b] for a, b in zip(*np.nonzero(adj), strict=True))
