"""Batched scores leave GRaSP's search path and result as full scores give them."""

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
    raw = BICScore(x, penalty_discount=0.0)
    lam = (raw.score(0, [2, 3]) - raw.score(0, [1, 2, 3])) / raw.log_n
    for _ in range(4):
        lam = np.nextafter(lam, np.inf)
    return x, lam


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

    def capture_batch(self, vertex, parents, deltas, **options):
        batches.append((vertex, tuple(parents), tuple(deltas)))
        return original_batch(self, vertex, parents, deltas, **options)

    monkeypatch.setattr(grasp_mod._Order, "__init__", capture_order)
    monkeypatch.setattr(DeltaBICScore, "score_many_with_base", capture_batch)
    batched_graph, batched_score = grasp_mod.grasp(x, penalty_discount=lam, seed=seed)
    batched_requests = batches.copy()
    batches.clear()

    def per_candidate(self, vertex, parents, deltas, *, bounds):
        batches.append((vertex, tuple(parents), tuple(deltas)))
        bounds[:] = [0.0] * len(deltas)
        return [
            self.score_obj.score(
                vertex, sorted([*parents, u]) if op == "add" else [p for p in parents if p != u]
            )
            for op, u in deltas
        ]

    monkeypatch.setattr(DeltaBICScore, "score_many_with_base", per_candidate)
    scalar_graph, scalar_score = grasp_mod.grasp(x, penalty_discount=lam, seed=seed)

    assert any(len(deltas) > 1 for _, _, deltas in batched_requests)
    if x.shape[1] == 4:
        assert any(parents and len(deltas) > 1 for _, parents, deltas in batched_requests)
    assert batched_requests == batches
    assert orders[0].order == orders[1].order
    assert orders[0].parents == orders[1].parents
    np.testing.assert_array_equal(batched_graph.to_numpy(), scalar_graph.to_numpy())
    assert batched_score == scalar_score
