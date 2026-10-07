"""Check grow-shrink branch priority and BOSS's tolerance for score ties.

``_GSTNode.trace`` takes the first branch allowed by the prefix. Branch order therefore determines
the search path. BOSS's grow and shrink steps break score ties by variable index; its order move
takes the leftmost tied position. Each step requires an improvement larger than ``_SCORE_EPS``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore
from andrey.search import boss as boss_mod
from andrey.search import grasp as grasp_mod

D_NODES, N_SAMPLES = 12, 400


def _sample() -> np.ndarray:
    """Return a sample with several improving additions."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N_SAMPLES, D_NODES))
    for j in range(1, D_NODES):
        x[:, j] += 0.8 * x[:, j - 1]
    return x


def _descending(scores: list[float], tolerance: float = 0.0) -> bool:
    return all(a >= b - tolerance for a, b in zip(scores, scores[1:]))


def test_boss_grows_best_scoring_branch_first() -> None:
    """BOSS's tree orders a node's improving additions best-first, up to ``_SCORE_EPS`` ties."""
    x = _sample()
    score = BICScore(x, lambda_value=1.0)
    tree = boss_mod._GrowShrinkTree(0, score, DeltaBICScore(score), D_NODES)

    node = tree.root
    node._grow([v for v in range(D_NODES) if v != 0], [])

    # Require two branches so `_descending` cannot pass vacuously.
    assert len(node.branches) >= 2, "fixture yields too few improving additions to order"
    assert _descending([b.grow_score for b in node.branches], boss_mod._SCORE_EPS)


@pytest.mark.parametrize(
    "gains, expected",
    [
        ({0: 10.0, 1: 10.0 + 0.5e-6, 2: 0.5e-6, 3: 2e-6}, [0, 1, 3]),
        ({0: 10.0, 1: 10.0 + 0.75e-6, 2: 10.0 + 1.5e-6}, [1, 2, 0]),
    ],
)
def test_boss_grow_ties_use_variable_index_and_require_an_improvement(gains, expected) -> None:
    delta = SimpleNamespace(
        score_many_with_base=lambda vertex, parents, moves: [-gains[u] for _, u in moves]
    )
    tree = SimpleNamespace(vertex=4, empty_score=0.0, delta=delta)
    node = boss_mod._GSTNode(tree)

    node._grow(sorted(gains, reverse=True), [])

    assert [branch.add for branch in node.branches] == expected


def test_boss_shrink_ties_use_variable_index_and_require_an_improvement() -> None:
    eps = boss_mod._SCORE_EPS
    gains = {0: 10.0, 1: 10.0 + eps / 2}
    delta = SimpleNamespace(
        score_many_with_base=lambda vertex, parents, moves: [-gains[u] for _, u in moves]
    )
    tree = SimpleNamespace(vertex=2, empty_score=0.0, delta=delta)
    node = boss_mod._GSTNode(tree)
    parents = [1, 0]

    node._shrink(parents)

    assert node.remove == [0]
    assert parents == [1]
    assert node.shrink_score == gains[0]


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (10.0, 10.0, [1, 0, 2]),
        (10.0, 10.0 + 0.5e-6, [1, 0, 2]),
        (10.0, 10.0 + 2e-6, [0, 2, 1]),
        (0.75e-6, 1.5e-6, [0, 2, 1]),
        (0.75e-6, 0.5e-6, [0, 1, 2]),
        (1e-6, 1e-6, [0, 1, 2]),
    ],
)
def test_boss_mutation_prefers_the_leftmost_improving_tied_position(left, right, expected) -> None:
    # Moving variable 1 to either end changes the total from zero to `left` or `right`.
    gsts = [
        SimpleNamespace(trace=lambda prefix: left if 1 in prefix else 0),
        SimpleNamespace(trace=lambda prefix: right if 2 in prefix else 0),
        SimpleNamespace(trace=lambda prefix: 0),
    ]
    order = [0, 1, 2]
    positions = {v: v for v in order}

    assert boss_mod._better_mutation(1, order, gsts, positions) == (expected != [0, 1, 2])
    assert order == expected
    assert positions == {v: i for i, v in enumerate(expected)}


def test_grasp_grows_best_scoring_branch_first() -> None:
    """GRaSP's tree orders a node's improving additions best-first."""
    x = _sample()
    score = BICScore(x, lambda_value=1.0)
    tree = grasp_mod._GST(0, score, DeltaBICScore(score), D_NODES)

    node = tree.root
    node.grow([v for v in range(D_NODES) if v != 0], [])

    # Require two branches so `_descending` cannot pass vacuously.
    assert len(node.branches) >= 2, "fixture yields too few improving additions to order"
    assert _descending([b.grow_score for b in node.branches])
