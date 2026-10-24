"""NOTEARS and DAGMA recover small linear graphs, and ``robust=True`` adds a likelihood path.

The first graph has equal noise variances, where least squares identifies the DAG. The robust fits
are checked for the path they take, for determinism and for column order, and on one standardized
case with unequal noise variances.
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
# 0 -> 1 -> 2.
CHAIN = {(0, 1): 1.0, (1, 2): -0.8}
FITS = {
    "notears": lambda X, **kw: andrey.notears(X, **kw),
    "dagma": lambda X, **kw: andrey.dagma(X, **kw),
}
# The robust DAGMA fits below repeat a five-stage central path; fewer steps keep the test short.
QUICK = {"notears": {}, "dagma": {"warm_iter": 3000, "max_iter": 6000}}
NAMES = list(FITS)


def _sample(
    edges: dict[tuple[int, int], float], stds: list[float], n: int, seed: int
) -> np.ndarray:
    """A linear SEM whose columns are in causal order, with noise standard deviations ``stds``."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, len(stds)))
    for j, std in enumerate(stds):
        X[:, j] = std * rng.standard_normal(n)
        for (i, k), weight in edges.items():
            if k == j:
                X[:, j] += weight * X[:, i]
    return X


def _fit(name: str, X: np.ndarray, **kwargs) -> andrey.StructureOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FITS[name](X, **kwargs)


def _edges(out: andrey.StructureOutput) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j in zip(*np.nonzero(out.weighted_adjacency), strict=True)}


def _unequal_variances(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Standardized data of a five-variable graph with unequal noise variances, and its DAG."""
    rng = np.random.default_rng(seed)
    d = 5
    B = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < 0.45:
                B[i, j] = rng.uniform(0.6, 1.5) * rng.choice([-1, 1])
    stds = rng.uniform(0.4, 2.0, d)
    X = np.zeros((2000, d))
    for j in range(d):
        X[:, j] = stds[j] * rng.standard_normal(2000) + X @ B[:, j]
    return (X - X.mean(axis=0)) / X.std(axis=0), (B != 0).astype(int)


@pytest.fixture(scope="module")
def tree() -> np.ndarray:
    return _sample(TREE, [1.0] * 5, 2000, 0)


@pytest.fixture(scope="module")
def chain() -> np.ndarray:
    return _sample(CHAIN, [1.0] * 3, 1000, 2)


# ---- fixed cases ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_defaults_recover_a_tree_with_equal_noise_variances(name, tree) -> None:
    """With equal noise variances least squares identifies the DAG, to within the l1 shrinkage."""
    out = _fit(name, tree)
    assert _edges(out) == set(TREE)
    for (i, j), weight in TREE.items():
        assert out.weighted_adjacency[i, j] == pytest.approx(weight, abs=0.2)
    assert out.metadata["robust"] is False
    assert out.metadata["path_steps"] == 0


@pytest.mark.parametrize("name", NAMES)
def test_independent_columns_have_no_edges(name) -> None:
    out = _fit(name, np.random.default_rng(0).standard_normal((1000, 4)))
    assert not out.weighted_adjacency.any()


@pytest.mark.parametrize("name", NAMES)
def test_robust_recovers_a_chain_and_takes_a_path_step(name, chain) -> None:
    out = _fit(name, chain, robust=True, **QUICK[name])
    assert _edges(out) == set(CHAIN)
    assert out.metadata["robust"] is True
    assert out.metadata["path_steps"] >= 1


def test_robust_dagma_recovers_the_class_of_standardized_data_with_unequal_variances() -> None:
    """Here least squares is one edit from the CPDAG, and the likelihood path reaches it."""
    X, adjacency = _unequal_variances(8)
    truth = to_cpdag(dag_from_adjacency(adjacency))
    default = to_cpdag(_fit("dagma", X).structure)
    robust = to_cpdag(_fit("dagma", X, robust=True).structure)
    assert shd(robust, truth) == 0
    assert shd(robust, truth) <= shd(default, truth)


# ---- determinism and column order ----------------------------------------------------------------


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_a_repeated_fit_is_identical(name, robust, chain) -> None:
    first = _fit(name, chain, robust=robust, **QUICK[name])
    second = _fit(name, chain, robust=robust, **QUICK[name])
    np.testing.assert_array_equal(first.weighted_adjacency, second.weighted_adjacency)


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_reordering_the_columns_only_reorders_the_weights(name, robust, chain) -> None:
    """The columns are fit in the order of their ranks, so the order they arrive in is lost."""
    order = np.array([2, 0, 1])
    base = _fit(name, chain, robust=robust, **QUICK[name]).weighted_adjacency
    moved = _fit(name, chain[:, order], robust=robust, **QUICK[name]).weighted_adjacency
    np.testing.assert_allclose(moved, base[np.ix_(order, order)], atol=1e-6)
