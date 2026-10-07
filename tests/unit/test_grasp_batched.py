"""Batched GRaSP grow scores preserve candidate order and the settled search result."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore
from andrey.search import grasp as grasp_mod


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


def _sample(d: int) -> np.ndarray:
    x = np.random.default_rng(0).standard_normal((100, d))
    x[:, 1] += x[:, 0]
    x[:, 2] += x[:, 1]
    if d == 4:
        x[:, 3] += x[:, 0] + x[:, 2]
    return x


def _sibling_tie_sample() -> np.ndarray:
    contrasts = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    variances = np.array([0.1, 0.1, 2.0])
    corr = np.array([[1.0, 0.8, 0.8], [0.8, 1.0, 0.99], [0.8, 0.99, 1.0]])
    cov = corr * np.sqrt(variances[:, None] * variances)
    return contrasts @ np.linalg.cholesky(cov).T * np.sqrt(3 / 4)


def _shrink_tie_sample() -> tuple[np.ndarray, float]:
    bits = np.indices((2, 2, 2)).reshape(3, -1).T * 2 - 1
    independent = np.column_stack([bits, bits[:, 0] * bits[:, 1]])
    coefficients = np.array([[1.2, 1, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [1, -3, 1, 0]])
    x = independent @ coefficients.T * np.sqrt(7 / 8)
    raw = BICScore(x, lambda_value=0.0)
    lam = (raw.score(0, [2, 3]) - raw.score(0, [1, 2, 3])) / raw.log_n
    for _ in range(4):
        lam = np.nextafter(lam, np.inf)
    return x, lam


@pytest.mark.parametrize("parents", [[], [2]], ids=["empty-base", "nonempty-base"])
def test_grow_preserves_candidates_and_parents(parents: list[int]) -> None:
    score = BICScore(_sample(4))
    tree = grasp_mod._GST(0, score, DeltaBICScore(score), 4)
    node = grasp_mod._GSTNode(tree, score=-score.score(0, parents))
    available = [3, 1]
    expected = [(add, -score.score(0, [*parents, add])) for add in available]
    expected = sorted(
        [(add, value) for add, value in expected if value > node.grow_score],
        key=lambda item: item[1],
        reverse=True,
    )
    original_parents = parents.copy()

    node.grow(available, parents)

    assert parents == original_parents
    assert available == [3, 1]
    assert node.branches is not None
    assert expected
    assert [branch.add for branch in node.branches] == [add for add, _ in expected]
    np.testing.assert_allclose(
        [branch.grow_score for branch in node.branches],
        [value for _, value in expected],
        rtol=0,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    "seed, parents, steps",
    [(0, [], 0), (1, [1], 3)],
    ids=["empty-base-threshold", "nonempty-base-threshold"],
)
def test_grow_matches_full_scores_at_inclusion_threshold(seed, parents, steps) -> None:
    a = np.random.default_rng(seed).standard_normal((4, 4))
    cov = a @ a.T
    available = [i for i in range(1, 4) if i not in parents]
    raw = BICScore.from_cov(cov, 100, lambda_value=0.0)
    lam = (raw.score(0, parents) - raw.score(0, [*parents, available[0]])) / raw.log_n
    for _ in range(steps):
        lam = np.nextafter(lam, -np.inf)
    score = BICScore.from_cov(cov, 100, lambda_value=lam)
    tree = grasp_mod._GST(0, score, DeltaBICScore(score), 4)
    base = -score.score(0, parents)
    node = grasp_mod._GSTNode(tree, score=base)
    expected = [(add, -score.score(0, [*parents, add])) for add in available]
    expected = sorted(
        [(add, value) for add, value in expected if value > base],
        key=lambda item: item[1],
        reverse=True,
    )

    node.grow(available, parents)

    assert [branch.add for branch in node.branches] == [add for add, _ in expected]


@pytest.mark.parametrize(
    "gain, candidate_error, base_error",
    [(-1e-10, 5e-10, 0), (1e-10, -5e-10, 0), (-1e-10, 0, -5e-10), (1e-10, 0, 5e-10)],
    ids=["false-add", "missed-add", "base-too-low", "base-too-high"],
)
def test_grow_rechecks_both_sides_of_near_ties(
    monkeypatch: pytest.MonkeyPatch, gain: float, candidate_error: float, base_error: float
) -> None:
    parents = [1]
    raw = BICScore(_sample(4), lambda_value=0.0)
    lam = (raw.score(0, parents) - raw.score(0, [1, 2]) - gain) / raw.log_n
    score = BICScore.from_cov(raw.cov, raw.n, lambda_value=lam)
    delta = DeltaBICScore(score)
    tree = grasp_mod._GST(0, score, delta, 4)
    base = -score.score(0, parents)
    candidate = -score.score(0, [1, 2])
    node = grasp_mod._GSTNode(tree, score=base + base_error)

    def rounded_batch(vertex, base_parents, deltas):
        return [
            score.score(vertex, [*base_parents, add]) - (candidate_error if add == 2 else 0)
            for _, add in deltas
        ]

    # Rounding within the batched scorer's error bound reverses the strict decision.
    improves = candidate > base
    assert improves == (gain > 0)
    assert (candidate + candidate_error > node.grow_score) != improves
    monkeypatch.setattr(delta, "score_many_with_base", rounded_batch)

    node.grow([2, 3], parents)

    assert (2 in [branch.add for branch in node.branches]) == improves
    assert node.grow_score == node.shrink_score == base
    assert parents == [1]


@pytest.mark.parametrize("available", [[1, 2], [2, 1]])
def test_grow_preserves_input_order_for_tied_siblings(monkeypatch, available) -> None:
    cov = np.array([[1.0, 0.8, 0.8], [0.8, 1.0, 0.99], [0.8, 0.99, 1.0]])
    score = BICScore.from_cov(cov, 100)
    delta = DeltaBICScore(score)
    tree = grasp_mod._GST(0, score, delta, 3)
    exact = -score.score(0, [1])
    assert exact == -score.score(0, [2])
    assert exact - tree.empty_score > grasp_mod._SCORE_EPS

    def rounded_batch(vertex, parents, deltas):
        return [score.score(vertex, [add]) - i * 1e-10 for i, (_, add) in enumerate(deltas)]

    monkeypatch.setattr(delta, "score_many_with_base", rounded_batch)

    tree.root.grow(available, [])

    assert [branch.add for branch in tree.root.branches] == available


def test_shrink_rechecks_base_when_no_grow_candidates_remain() -> None:
    coefficients = np.array([[1.3, 1, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [1, -3, 1, 0]])
    cov = coefficients @ coefficients.T
    raw = BICScore.from_cov(cov, 100, lambda_value=0.0)
    lam = (raw.score(0, [2, 3]) - raw.score(0, [1, 2, 3])) / raw.log_n
    for _ in range(4):
        lam = np.nextafter(lam, -np.inf)
    score = BICScore.from_cov(cov, 100, lambda_value=lam)
    parents = [1, 2, 3]
    exact = -score.score(0, parents)
    assert all(-score.score(0, [p for p in parents if p != drop]) <= exact for drop in parents)
    tree = grasp_mod._GST(0, score, DeltaBICScore(score), 4)
    node = grasp_mod._GSTNode(tree, score=exact - 5e-10)

    node.trace([], parents, set(parents))

    assert parents == [1, 2, 3]
    assert node.shrink_score == exact


@pytest.mark.parametrize("case", ["empty-base", "nonempty-base", "sibling-tie", "shrink-tie"])
def test_grasp_matches_per_candidate_order_and_graph(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    lam, seed = 1.0, 0
    if case == "sibling-tie":
        x, seed = _sibling_tie_sample(), 1
    elif case == "shrink-tie":
        x, lam = _shrink_tie_sample()
        seed = 2
    else:
        x = _sample(3 if case == "empty-base" else 4)
    orders = []
    batches = []
    original_init = grasp_mod._Order.__init__
    original_batch = DeltaBICScore.score_many_with_base

    def capture_order(self, *args):
        original_init(self, *args)
        orders.append(self)

    def capture_batch(self, vertex, parents, deltas):
        batches.append((vertex, tuple(parents), tuple(deltas)))
        return original_batch(self, vertex, parents, deltas)

    monkeypatch.setattr(grasp_mod._Order, "__init__", capture_order)
    monkeypatch.setattr(DeltaBICScore, "score_many_with_base", capture_batch)
    batched_graph, batched_score = grasp_mod.grasp(x, lambda_value=lam, random_state=seed)
    batched_requests = batches.copy()
    batches.clear()

    def per_candidate(self, vertex, parents, deltas):
        batches.append((vertex, tuple(parents), tuple(deltas)))
        assert all(op == "add" for op, _ in deltas)
        return [self.score_obj.score(vertex, [*parents, add]) for _, add in deltas]

    monkeypatch.setattr(DeltaBICScore, "score_many_with_base", per_candidate)
    scalar_graph, scalar_score = grasp_mod.grasp(x, lambda_value=lam, random_state=seed)

    assert any(len(deltas) > 1 for _, _, deltas in batched_requests)
    if x.shape[1] == 4:
        assert any(parents and len(deltas) > 1 for _, parents, deltas in batched_requests)
    assert batched_requests == batches
    assert orders[0].order == orders[1].order
    assert orders[0].parents == orders[1].parents
    np.testing.assert_array_equal(batched_graph.to_numpy(), scalar_graph.to_numpy())
    assert batched_score == scalar_score
