"""The LiNGAM causal-order search breaks score ties by each column's own data.

Two columns that are interchangeable in the sample have mathematically equal scores, which differ
only by rounding, and the rounding depends on the CPU's BLAS kernel. A test cannot switch kernels,
so it checks the invariant that rules out a kernel-dependent pick: reordering the columns reorders
the causal order and changes nothing else. Continuous data without ties give the orders of a plain
argmax of the scores.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from andrey.data.scm import sample_scm
from andrey.lingam import multi_group, pwling

# Ranks / 8 of two columns with the same values, whose sample is invariant to swapping them.
_SWAP_PAIR_N8 = np.array([[5, 4], [4, 6], [2, 8], [3, 5], [7, 2], [1, 7], [6, 3], [8, 1]]) / 8


def _exchangeable_pair(seed: int, n: int, k: int) -> tuple[np.random.Generator, np.ndarray]:
    """Ranks / ``n`` of ``a`` and of ``b`` = ``a`` with ``k`` disjoint adjacent ranks swapped."""
    rng = np.random.default_rng(seed)
    a = rng.permutation(n) + 1.0
    b = a.copy()
    for r in rng.choice(np.arange(1, n, 2), size=k, replace=False):
        i, j = np.flatnonzero(a == r)[0], np.flatnonzero(a == r + 1)[0]
        b[i], b[j] = r + 1, r
    return rng, np.column_stack([a, b]) / n


def _balanced_binary_pair(seed: int, n: int) -> np.ndarray:
    """Median splits of two correlated Gaussians: both columns hold ``n / 2`` ones."""
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.5], [0.5, 1.0]], size=n)
    return (z > np.median(z, axis=0)).astype(np.float64)


def _assert_order_follows_columns(fit, groups: list[np.ndarray]) -> None:
    """Every column permutation of ``groups`` gives the same order, relabeled."""
    order = fit(groups)
    for perm in itertools.permutations(range(groups[0].shape[1])):
        relabeled = fit([X[:, list(perm)] for X in groups])
        assert [perm[j] for j in relabeled] == order, perm


def _direct(groups: list[np.ndarray]) -> list[int]:
    return pwling.find_causal_order(groups[0])


def _shared(groups: list[np.ndarray]) -> list[int]:
    return multi_group.multi_group_direct_lingam(groups)[0]


def test_swap_symmetric_pair_order_follows_columns() -> None:
    _assert_order_follows_columns(_direct, [_SWAP_PAIR_N8])


@pytest.mark.parametrize("seed", [3, 6])
def test_exchangeable_pair_with_an_independent_column(seed: int) -> None:
    """A near-copy pair is the top tie at a score near zero; the third column is independent."""
    rng, pair = _exchangeable_pair(seed, 1000, 50)
    X = np.column_stack([pair, (rng.permutation(1000) + 1.0) / 1000])
    _assert_order_follows_columns(_direct, [X])


def test_balanced_binary_pair_order_follows_columns() -> None:
    _assert_order_follows_columns(_direct, [_balanced_binary_pair(0, 1000)])


def test_multi_group_order_follows_columns() -> None:
    groups = [_balanced_binary_pair(0, 600), _balanced_binary_pair(1, 400)]
    _assert_order_follows_columns(_shared, groups)


def test_pick_ties_within_the_tolerance_only() -> None:
    """Rounding-sized gaps tie and go to the lower rank; a real gap keeps the higher score."""
    ranks = np.array([1, 0])
    assert pwling.pick(np.array([0.0, -7.9e-31]), [0, 1], ranks) == 1
    assert pwling.pick(np.array([-0.25, -0.25 - 1e-15]), [0, 1], ranks) == 1
    assert pwling.pick(np.array([0.0, -1e-15]), [0, 1], ranks) == 0
    assert pwling.pick(np.array([-0.25, -0.25 - 1e-9]), [0, 1], ranks) == 0


# Orders a plain argmax of the scores gives on continuous data without ties.
_CONTINUOUS_ORDERS = {
    (6, "uniform", 0): [4, 2, 0, 3, 1, 5],
    (8, "laplace", 1): [6, 2, 1, 5, 7, 3, 0, 4],
    (10, "exponential", 2): [2, 6, 0, 7, 1, 9, 4, 8, 3, 5],
    (12, "gumbel", 3): [2, 0, 1, 9, 4, 7, 8, 10, 6, 11, 3, 5],
    (15, "uniform", 4): [10, 5, 3, 7, 11, 6, 1, 12, 2, 8, 0, 14, 9, 4, 13],
}
_SHARED_ORDERS = {0: [2, 6, 1, 4, 0, 5, 3], 1: [3, 4, 5, 0, 1, 6, 2]}


@pytest.mark.parametrize(("d", "noise", "seed"), list(_CONTINUOUS_ORDERS))
def test_continuous_orders_are_unchanged(d: int, noise: str, seed: int) -> None:
    X = sample_scm(noise=noise, d=d, n=500, seed=seed).data
    assert pwling.find_causal_order(X) == _CONTINUOUS_ORDERS[d, noise, seed]


@pytest.mark.parametrize("seed", list(_SHARED_ORDERS))
def test_continuous_shared_orders_are_unchanged(seed: int) -> None:
    groups = [sample_scm(noise="uniform", d=7, n=300 + 100 * g, seed=seed).data for g in range(2)]
    assert _shared(groups) == _SHARED_ORDERS[seed]
