"""Bitmask and set-based path checks agree with an independent traversal on sampled graphs.

The tests include graphs with more than 64 nodes and compare both implementations against a
separately written traversal that reads each successor mask one bit at a time, so a bug the two
share cannot pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core import _bitset
from andrey.core.structure import TAIL
from andrey.search import operators as ops


def _reaches_setbased(succ: list[int], start: int, target: int, barrier: int) -> bool:
    """Independent plain-graph reachability oracle over the successor bitmasks."""
    if start == target:
        return True
    d = len(succ)
    bar = int(barrier)
    removed = {v for v in range(d) if (bar >> v) & 1 and v not in (start, target)}
    seen = {start}
    frontier = [start]
    while frontier:
        u = frontier.pop()
        m = int(succ[u])
        v = 0
        while m:
            if m & 1:
                if v == target:
                    return True
                if v not in removed and v not in seen:
                    seen.add(v)
                    frontier.append(v)
            m >>= 1
            v += 1
    return False


def _random_tail_adj(rng: np.random.Generator, d: int) -> np.ndarray:
    """A random ``(d, d)`` endpoint-mark matrix; only the ``TAIL`` marks drive reachability."""
    adj = (rng.random((d, d)) < 0.3).astype(np.int8) * TAIL
    np.fill_diagonal(adj, 0)
    return adj


@pytest.mark.parametrize("seed", range(40))
def test_setbfs_matches_oracle_across_d(seed):
    # operators.blocks_semi_directed_paths must agree with the independent traversal on sampled
    # graph sizes. Both treat start == target as reachable through an empty path.
    rng = np.random.default_rng(seed)
    d = int(rng.integers(2, 130))
    adj = _random_tail_adj(rng, d)
    succ_lists = ops.tail_successor_lists(adj)
    pyint = [int(sum(1 << c for c in row)) for row in succ_lists]  # correct, un-truncated masks
    for _ in range(6):
        start = int(rng.integers(0, d))
        target = int(rng.integers(0, d))
        blocked = {int(v) for v in np.nonzero(rng.random(d) < 0.2)[0]}
        want = _reaches_setbased(pyint, start, target, _bitset.bitmask(blocked))
        got = not ops.blocks_semi_directed_paths(succ_lists, start, target, set(blocked))
        assert got == want, (seed, d, start, target, blocked)


def test_setbfs_path_through_high_bit():
    # The only semi-directed path 0 ~> 1 runs through node 70, a bit past one 64-bit word.
    d = 80
    adj = np.zeros((d, d), dtype=np.int8)
    adj[0, 70] = TAIL
    adj[70, 1] = TAIL
    succ_lists = ops.tail_successor_lists(adj)
    assert not ops.blocks_semi_directed_paths(succ_lists, 0, 1, set())  # 1 reachable from 0 via 70
    assert ops.blocks_semi_directed_paths(succ_lists, 0, 1, {70})  # blocked once 70 is removed
    pyint = [int(sum(1 << c for c in row)) for row in succ_lists]  # independent oracle agrees
    assert _reaches_setbased(pyint, 0, 1, 0) is True
    assert _reaches_setbased(pyint, 0, 1, _bitset.bitmask({70})) is False
    # start == target is the empty path: reachable, so nothing blocks it.
    assert ops.blocks_semi_directed_paths(succ_lists, 5, 5, set()) is False
    assert ops.blocks_semi_directed_paths(succ_lists, 5, 5, {5}) is False


@pytest.mark.parametrize("seed", range(40))
def test_reach_set_matches_oracle_across_d(seed):
    """One reach-set traversal answers every target outside the barrier, past 64 nodes too."""
    rng = np.random.default_rng(seed)
    d = int(rng.integers(2, 200))
    succ = _bitset.row_masks(_random_tail_adj(rng, d) == TAIL)
    for _ in range(4):
        start = int(rng.integers(0, d))
        barrier = _bitset.bitmask(v for v in range(d) if v != start and rng.random() < 0.2)
        reach = _bitset.semidirected_reach_set(succ, start, barrier)
        assert (reach >> start) & 1
        for target in range(d):
            if target == start or (barrier >> target) & 1:
                continue
            want = _reaches_setbased(succ, start, target, barrier)
            assert bool((reach >> target) & 1) == want, (seed, d, start, target)


@pytest.mark.parametrize("d", [1, 8, 9, 63, 64, 65, 128, 129])
def test_row_masks_across_word_boundary(d):
    """Each row mask sets exactly the row's columns, at widths on both sides of a 64-bit word."""
    matrix = np.random.default_rng(d).random((d, d)) < 0.3
    for i, mask in enumerate(_bitset.row_masks(matrix)):
        assert _bitset.nodes(mask) == np.flatnonzero(matrix[i]).tolist(), (d, i)


@pytest.mark.parametrize("seed", range(20))
def test_nodes_inverts_bitmask(seed):
    """``nodes`` lists a mask's nodes in ascending order, for sparse and dense masks alike."""
    rng = np.random.default_rng(seed)
    d = int(rng.integers(1, 300))
    for size in {0, 1, min(d, 24), min(d, 25), int(rng.integers(0, d + 1)), d}:
        chosen = sorted(rng.choice(d, size=size, replace=False).tolist())
        assert _bitset.nodes(_bitset.bitmask(chosen)) == chosen, (seed, d, size)
