"""NOTEARS and DAGMA recover small linear graphs, and ``robust=True`` does not depend on the scale.

The first graph has equal noise variances, where least squares identifies the DAG. The second is a
collider with unequal noise variances, whose CPDAG is the DAG itself, fit on standardized columns.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey.api._adapt import dag_from_adjacency
from andrey.metrics import shd, to_cpdag

# 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 4, with weights of both signs.
TREE = {(0, 1): 1.2, (0, 2): -0.9, (1, 3): 1.0, (2, 4): 0.8}
# 0 -> 2 <- 1 and 2 -> 3: every edge is compelled.
COLLIDER = {(0, 2): 1.0, (1, 2): -0.8, (2, 3): 0.9}
METHODS = [andrey.notears, andrey.dagma]


def _sample(edges: dict[tuple[int, int], float], stds: list[float], n: int, seed: int) -> np.ndarray:
    """A linear SEM whose columns are in causal order, with noise standard deviations ``stds``."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, len(stds)))
    for j, std in enumerate(stds):
        X[:, j] = std * rng.standard_normal(n)
        for (i, k), weight in edges.items():
            if k == j:
                X[:, j] += weight * X[:, i]
    return X


def _fit(method, X: np.ndarray, **kwargs) -> andrey.StructureOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return method(X, **kwargs)


def _edges(out: andrey.StructureOutput) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j in zip(*np.nonzero(out.weighted_adjacency), strict=True)}


@pytest.fixture(scope="module")
def tree() -> np.ndarray:
    return _sample(TREE, [1.0] * 5, 2000, 0)


@pytest.fixture(scope="module")
def collider() -> np.ndarray:
    X = _sample(COLLIDER, [1.0, 2.0, 0.5, 1.0], 3000, 1)
    return (X - X.mean(axis=0)) / X.std(axis=0)


# ---- fixed cases ---------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_defaults_recover_a_tree_with_equal_noise_variances(method, tree) -> None:
    """With equal noise variances least squares identifies the DAG, signs and sizes to within the l1 shrinkage."""
    out = _fit(method, tree)
    assert _edges(out) == set(TREE)
    for (i, j), weight in TREE.items():
        assert out.weighted_adjacency[i, j] == pytest.approx(weight, abs=0.2)


@pytest.mark.parametrize("method", METHODS)
def test_robust_recovers_a_collider_with_unequal_noise_variances(method, collider) -> None:
    """The likelihood of a column does not depend on its scale, so standardizing keeps the DAG."""
    out = _fit(method, collider, robust=True)
    assert _edges(out) == set(COLLIDER)
    assert out.metadata["robust"] is True
    assert out.metadata["path_steps"] >= 1


@pytest.mark.parametrize("method", METHODS)
def test_independent_columns_have_no_edges(method) -> None:
    out = _fit(method, np.random.default_rng(0).standard_normal((1000, 4)))
    assert not out.weighted_adjacency.any()
    assert out.metadata["path_steps"] == 0


# ---- determinism and column order ----------------------------------------------------------------


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_a_repeated_fit_is_identical(method, robust, collider) -> None:
    first = _fit(method, collider, robust=robust)
    second = _fit(method, collider, robust=robust)
    np.testing.assert_array_equal(first.weighted_adjacency, second.weighted_adjacency)


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_reordering_the_columns_only_reorders_the_weights(method, robust, collider) -> None:
    """The columns are fit in the order of their ranks, so the order they arrive in is lost."""
    order = np.array([2, 0, 3, 1])
    base = _fit(method, collider, robust=robust).weighted_adjacency
    moved = _fit(method, collider[:, order], robust=robust).weighted_adjacency
    np.testing.assert_allclose(moved, base[np.ix_(order, order)], atol=1e-6)


# ---- the default and the robust fit on the same data ----------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_robust_is_closer_to_the_truth_than_the_default_on_standardized_unequal_variances(
    method, collider
) -> None:
    """Standardizing removes the variance order that least squares reads, so it misorients."""
    truth = np.zeros((4, 4), dtype=int)
    for i, j in COLLIDER:
        truth[i, j] = 1
    robust = to_cpdag(_fit(method, collider, robust=True).structure)
    default = to_cpdag(_fit(method, collider).structure)
    target = to_cpdag(dag_from_adjacency(truth))
    assert shd(robust, target) == 0
    assert shd(default, target) >= shd(robust, target)
