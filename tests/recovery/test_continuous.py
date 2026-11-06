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
    """Fit ``name``, which is experimental, so its ``ExperimentalWarning`` is expected."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        return FITS[name](X, **kwargs)


def _quick_fit(name: str, X: np.ndarray, **kwargs) -> andrey.StructureOutput:
    """Fit ``name`` on the ``QUICK`` schedule, which can stop DAGMA at its iteration limit."""
    with warnings.catch_warnings():
        if name == "dagma":
            warnings.simplefilter("ignore", andrey.SearchLimitWarning)
        return _fit(name, X, **kwargs, **QUICK[name])


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
    assert out.metadata["path_steps"] == 0


@pytest.mark.parametrize("name", NAMES)
def test_independent_columns_have_no_edges(name) -> None:
    out = _fit(name, np.random.default_rng(0).standard_normal((1000, 4)))
    assert not out.weighted_adjacency.any()


@pytest.mark.parametrize("name", NAMES)
def test_robust_recovers_a_chain_and_takes_a_path_step(name, chain) -> None:
    out = _quick_fit(name, chain, robust=True)
    assert _edges(out) == set(CHAIN)
    assert out.metadata["path_steps"] >= 1


@pytest.mark.filterwarnings("ignore:dagma stopped at its limit of max_iter")
@pytest.mark.parametrize("name", NAMES)
def test_max_path_caps_the_accepted_steps_and_warns(name, chain) -> None:
    """The path accepts its first step whatever its likelihood, so a cap of one is reached."""
    with pytest.warns(andrey.SearchLimitWarning, match="max_path=1"):
        capped = _fit(name, chain, robust=True, max_path=1, **QUICK[name])
    assert capped.metadata["path_steps"] == 1


@pytest.mark.parametrize("name", NAMES)
def test_an_mcp_wider_than_every_weight_empties_the_path(name, chain) -> None:
    """With ``mcp_delta`` above every weight, the quasi-MCP's slope reaches each one, and at
    ``mcp_lambda=5`` it pulls them all to zero. The path accepts its first step whatever its
    likelihood, so it returns the empty graph, and the next step does not lower the likelihood."""
    out = _quick_fit(name, chain, robust=True, mcp_lambda=5.0, mcp_delta=2.0)
    assert not out.weighted_adjacency.any()
    assert out.metadata["path_steps"] == 1


def test_h_tol_ends_the_dual_steps(tree) -> None:
    """The dual steps end once the acyclicity function falls to ``h_tol``."""
    tight, loose = _fit("notears", tree), _fit("notears", tree, h_tol=1e-2)
    assert tight.metadata["h_final"] <= 1e-8 < loose.metadata["h_final"] <= 1e-2


def test_robust_dagma_recovers_the_class_of_standardized_data_with_unequal_variances() -> None:
    """The likelihood path reaches the CPDAG."""
    X, adjacency = _unequal_variances(8)
    truth = to_cpdag(dag_from_adjacency(adjacency))
    robust = to_cpdag(_fit("dagma", X, robust=True).structure)
    assert shd(robust, truth) == 0


@pytest.mark.parametrize("name", NAMES)
def test_standardize_makes_the_fit_independent_of_the_units(name, tree) -> None:
    """Scaling by powers of two leaves the standardized columns equal bit for bit."""
    units = np.array([4.0, 0.25, 1.0, 8.0, 0.5])
    plain = _fit(name, tree, standardize=True)
    rescaled = _fit(name, tree * units, standardize=True)
    np.testing.assert_array_equal(rescaled.weighted_adjacency, plain.weighted_adjacency)


# ---- determinism and column order ----------------------------------------------------------------


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_a_repeated_fit_is_identical(name, robust, chain) -> None:
    first = _quick_fit(name, chain, robust=robust)
    second = _quick_fit(name, chain, robust=robust)
    np.testing.assert_array_equal(first.weighted_adjacency, second.weighted_adjacency)


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_reordering_the_columns_only_reorders_the_weights(name, robust, chain) -> None:
    """The columns are fit in the order of their ranks, so the order they arrive in is lost."""
    order = np.array([2, 0, 1])
    base = _quick_fit(name, chain, robust=robust).weighted_adjacency
    moved = _quick_fit(name, chain[:, order], robust=robust).weighted_adjacency
    np.testing.assert_allclose(moved, base[np.ix_(order, order)], atol=1e-6)
