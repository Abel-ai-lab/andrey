"""The central path of DAGMA: what it does with the diagonal of the weight matrix."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.search import continuous
from andrey.search.continuous import (
    Canonical,
    central_path,
    l1_penalty,
    least_squares,
    log_det_acyclicity,
)


def _canonical(cov: np.ndarray, forbidden: np.ndarray | None = None) -> Canonical:
    d = cov.shape[0]
    none = np.zeros((d, d), dtype=bool)
    order = np.arange(d)
    return Canonical(cov, order, order, none if forbidden is None else forbidden, none)


def _path(canonical: Canonical, *, stages: int, free_diagonal: bool) -> np.ndarray:
    return central_path(
        canonical,
        loss=least_squares,
        penalty=l1_penalty(0.03),
        start=np.zeros_like(canonical.cov),
        stages=stages,
        warm_iter=30000,
        max_iter=60000,
        lr=3e-4,
        free_diagonal=free_diagonal,
    )


def test_a_free_diagonal_is_optimized_and_a_fixed_one_stays_at_zero() -> None:
    """One stage on independent unit-variance columns: each self-weight ``w`` settles where
    ``-(1 - w) + 0.03 + 2 * w / (1 - w**2) = 0``, the stationary point of the stage's objective.
    """
    free = _path(_canonical(np.eye(3)), stages=1, free_diagonal=True)
    w = np.diagonal(free)
    np.testing.assert_allclose(-(1 - w) + 0.03 + 2 * w / (1 - w**2), 0.0, atol=1e-3)
    assert 0.25 < w.min() <= w.max() < 0.35
    fixed = _path(_canonical(np.eye(3)), stages=1, free_diagonal=False)
    np.testing.assert_array_equal(np.diagonal(fixed), 0.0)


def test_the_free_diagonal_falls_toward_zero_along_the_path() -> None:
    """The acyclicity function counts a self-weight as a cycle, so the full path removes it."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 3))
    x[:, 1] += 1.5 * x[:, 0]
    x[:, 2] -= 1.2 * x[:, 1]
    x -= x.mean(axis=0)
    weights = _path(_canonical(x.T @ x / 500), stages=5, free_diagonal=True)
    assert 0.0 < np.abs(np.diagonal(weights)).max() < 1e-3
    assert log_det_acyclicity(weights) < 1e-6
    assert set(zip(*np.nonzero(np.abs(weights) >= 0.3), strict=True)) == {(0, 1), (1, 2)}


def test_a_forbidden_entry_stays_at_zero_beside_a_free_diagonal() -> None:
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    forbidden = np.array([[False, True], [False, False]])
    weights = _path(_canonical(cov, forbidden), stages=1, free_diagonal=True)
    assert weights[0, 1] == 0.0
    assert weights[1, 0] != 0.0


def test_dagma_frees_the_diagonal_in_the_least_squares_fit_only(monkeypatch) -> None:
    """The likelihood steps of ``robust`` hold the diagonal at zero; the returned weights have a
    zero diagonal either way.
    """
    seen: list[bool] = []
    solve = continuous.central_path

    def recording(canonical, **kwargs):
        seen.append(kwargs["free_diagonal"])
        return solve(canonical, **kwargs)

    monkeypatch.setattr(continuous, "central_path", recording)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((400, 2))
    x[:, 1] += x[:, 0]
    with pytest.warns(match="dagma stopped at its limit of max_iter"):
        result = continuous.dagma(
            x,
            lambda1=0.03,
            w_threshold=0.3,
            stages=5,
            warm_iter=2000,
            max_iter=4000,
            lr=3e-4,
            robust=True,
            mcp_lambda=0.4,
            mcp_delta=0.2,
            mcp_decay=0.8,
            max_path=2,
        )
    assert seen[0] is True
    assert len(seen) > 1
    assert not any(seen[1:])
    np.testing.assert_array_equal(np.diagonal(result.weighted_adjacency), 0.0)
