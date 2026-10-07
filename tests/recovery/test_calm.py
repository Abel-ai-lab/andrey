"""CALM checks ground truth and same-seed determinism.

Optimizer output can drift across torch versions, so no baseline is committed.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.core import GraphStructure


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


def _skeleton(mask: np.ndarray) -> np.ndarray:
    upper = np.triu(mask | mask.T, 1)
    return upper[np.triu_indices(mask.shape[0], 1)]


def test_calm_recovers_skeleton() -> None:
    pytest.importorskip("torch")
    X, B = _random_sem(5, 2000, 0.4, seed=7)
    out = andrey.calm(X, seed=0, subproblem_iter=2000)
    assert isinstance(out.structure, GraphStructure)
    assert out.structure.kind == "dag"
    est = _skeleton(out.weighted_adjacency != 0)
    tru = _skeleton(B != 0)
    assert np.array_equal(est, tru), f"skeleton mismatch est={est} tru={tru}"


def test_calm_is_deterministic() -> None:
    pytest.importorskip("torch")
    X, _ = _random_sem(6, 800, 0.35, seed=3)
    first = andrey.calm(X, seed=0, subproblem_iter=400)
    second = andrey.calm(X, seed=0, subproblem_iter=400)
    assert first == second


def test_calm_output_contract() -> None:
    pytest.importorskip("torch")
    X, _ = _random_sem(5, 600, 0.4, seed=1)
    out = andrey.calm(X, seed=0, subproblem_iter=300)
    assert out.structure.kind == "dag"
    assert out.metadata["algorithm"] == "CALM"
    assert out.metadata["seed"] == 0
    assert out.weighted_adjacency.shape == (5, 5)
