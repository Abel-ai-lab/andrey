"""CALM checks a fixed case, ground truth, its recorded output, and same-seed determinism.

The tree fixture has a CPDAG that can be checked by hand. The ground-truth case is a random
linear-Gaussian SEM whose Markov equivalence class CALM must return. The recorded output holds
only the graph: the weights come from a torch optimizer.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.core import GraphStructure
from andrey.metrics import markov_equivalent, to_cpdag
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output


def _random_sem(d: int, n: int, p: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``(X, B)``: an ``(n, d)`` linear-Gaussian sample and its true weight matrix ``B``."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    B = np.zeros((d, d))
    for a in range(d):
        for b in range(a + 1, d):
            i, j = order[a], order[b]
            if rng.random() < p:
                B[i, j] = rng.uniform(0.5, 2.0) * rng.choice([-1.0, 1.0])
    X = rng.normal(size=(n, d)) @ np.linalg.inv(np.eye(d) - B)
    return X, B


@pytest.fixture(scope="module")
def tree_fit():
    """CALM's default fit on ``gauss_tree_5v``, shared by the tests that read it."""
    pytest.importorskip("torch")
    return andrey.calm(datagen.generate("gauss_tree_5v"))


# ---- fixed cases ---------------------------------------------------------------------------------


def test_calm_finds_no_edges_between_independent_columns() -> None:
    """No pair enters the moral graph, so no edge is a candidate."""
    pytest.importorskip("torch")
    out = andrey.calm(np.random.default_rng(0).standard_normal((1000, 4)))
    assert out.structure.oriented_edges() == []
    assert not out.weighted_adjacency.any()


def test_calm_recovers_the_class_of_a_tree(tree_fit) -> None:
    """The tree 0 -> 1, 0 -> 2, 1 -> 3, 1 -> 4 has no collider, so its CPDAG is undirected."""
    cpdag = datagen.to_variables("gauss_tree_5v", to_cpdag(tree_fit.structure))
    assert cpdag.oriented_edges(index=True) == [
        (0, 1, "undirected"),
        (0, 2, "undirected"),
        (1, 3, "undirected"),
        (1, 4, "undirected"),
    ]


def test_calm_matches_the_recorded_output(tree_fit) -> None:
    assert_recorded_output(tree_fit, "CALM", case="gauss_tree_5v")


# ---- ground truth --------------------------------------------------------------------------------


def test_calm_recovers_the_true_class() -> None:
    pytest.importorskip("torch")
    X, B = _random_sem(5, 2000, 0.4, seed=7)
    out = andrey.calm(X)
    assert out.structure.kind == "dag"
    assert markov_equivalent(out.structure, (B != 0).astype(int))  # [i, j] = 1 is i -> j


# ---- what the method promises --------------------------------------------------------------------


def test_calm_is_deterministic() -> None:
    pytest.importorskip("torch")
    X, _ = _random_sem(6, 800, 0.35, seed=3)
    first = andrey.calm(X, seed=0, subproblem_iter=100, max_iter=2)
    second = andrey.calm(X, seed=0, subproblem_iter=100, max_iter=2)
    assert first.structure.oriented_edges() != []
    assert first == second


def test_calm_output_contract(tree_fit) -> None:
    assert isinstance(tree_fit.structure, GraphStructure)
    assert tree_fit.structure.kind == "dag"
    assert tree_fit.metadata["algorithm"] == "calm"
    assert tree_fit.metadata["seed"] == 0
    assert tree_fit.weighted_adjacency.shape == (5, 5)


def test_calm_rejects_a_non_finite_loss() -> None:
    """Data whose covariance overflows makes the loss non-finite, so the fit stops with an error."""
    pytest.importorskip("torch")
    X, _ = _random_sem(4, 300, 0.4, seed=1)
    with pytest.raises(ValueError, match="not finite"):
        andrey.calm(X * 1e170, use_moral_graph=False, subproblem_iter=100, max_iter=2)


def test_calm_leaves_the_global_torch_generator_alone() -> None:
    """CALM draws from its own seeded generator, so a caller's global torch stream is unchanged."""
    torch = pytest.importorskip("torch")
    X, _ = _random_sem(4, 300, 0.4, seed=1)
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    andrey.calm(X, seed=0, subproblem_iter=60, max_iter=1)
    assert torch.equal(torch.rand(3), expected)
