"""Bitset reachability: the numpy reference, the numba kernel, and an independent oracle agree.

The numpy reference and the numba kernel share the bitmask algorithm, so they could share a bug; the
set-based oracle here is written independently to break that correlation. All three must return the
same booleans on random graphs, including the contract's quirks (``start == target`` is reachable;
``start`` / ``target`` are never removed even when barred).
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core import _bitset, backend
from andrey.core.structure import TAIL
from andrey.search import operators as ops


def _reaches_setbased(succ: np.ndarray, start: int, target: int, barrier: int) -> bool:
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


def _random_succ(rng: np.random.Generator, d: int) -> np.ndarray:
    adj = rng.random((d, d)) < 0.3
    np.fill_diagonal(adj, False)
    weights = np.uint64(1) << np.arange(d, dtype=np.uint64)
    return (adj.astype(np.uint64) * weights).sum(axis=1)


def _rand_barrier(rng: np.random.Generator, d: int) -> int:
    """A random d-bit barrier as a raw Python int; covers bit 63 at d = 64 (rng.integers cannot)."""
    mask = (1 << d) - 1
    return int.from_bytes(rng.bytes(8), "little") & mask


@pytest.mark.parametrize("seed", range(60))
def test_reference_matches_oracle(seed):
    rng = np.random.default_rng(seed)
    d = int(rng.integers(2, 65))  # up to the d <= 64 contract limit
    succ = _random_succ(rng, d)
    for _ in range(6):
        start, target = int(rng.integers(0, d)), int(rng.integers(0, d))
        barrier = _rand_barrier(rng, d)
        want = _reaches_setbased(succ, start, target, barrier)
        got = _bitset._semidirected_reaches_numpy(succ, start, target, barrier)
        assert got == want, (seed, start, target, barrier)


@pytest.mark.parametrize("seed", range(60))
def test_numba_matches_oracle(seed):
    if backend.numba() is None:
        pytest.skip("numba (the [numba] extra) is not installed")
    reaches = _bitset.reaches_impl(backend="numba")  # the production path: a raw Python-int barrier
    rng = np.random.default_rng(seed)
    d = int(rng.integers(2, 65))
    succ = _random_succ(rng, d)
    for _ in range(6):
        start, target = int(rng.integers(0, d)), int(rng.integers(0, d))
        barrier = _rand_barrier(rng, d)
        want = _reaches_setbased(succ, start, target, barrier)
        got = bool(reaches(succ, start, target, barrier))
        assert got == want, (seed, start, target, barrier)


def test_contract_quirks():
    succ = np.array([0b010, 0b100, 0b000], dtype=np.uint64)  # 0 -> 1 -> 2
    reach = _bitset._semidirected_reaches_numpy
    assert reach(succ, 0, 0, 0) is True  # start == target
    assert reach(succ, 0, 2, 0) is True  # 0 -> 1 -> 2
    assert reach(succ, 0, 2, _bitset.bitmask([1])) is False  # barrier at 1 cuts the only path
    # start / target are never removed even when their bits are barred.
    assert reach(succ, 0, 2, _bitset.bitmask([0, 2])) is True


def test_reaches_impl_hoist_and_dispatch():
    # A numpy pin yields the reference itself; the default resolves to numba (a wrapper over the
    # njit kernel that casts the barrier) when available, else the numpy reference.
    assert _bitset.reaches_impl(backend="numpy") is _bitset._semidirected_reaches_numpy
    if backend.numba() is not None:
        assert _bitset.reaches_impl() is not _bitset._semidirected_reaches_numpy
    else:
        assert _bitset.reaches_impl() is _bitset._semidirected_reaches_numpy


def test_numba_d64_bit63_barrier():
    # Regression: a raw Python-int barrier with bit 63 set (d = 64) overflows numba's int64 argument
    # typing unless it is cast at the dispatch boundary. The two backends must agree, not crash.
    if backend.numba() is None:
        pytest.skip("numba (the [numba] extra) is not installed")
    d = 64
    succ = np.zeros(d, dtype=np.uint64)  # a chain 0 -> 1 -> ... -> 63
    for u in range(d - 1):
        succ[u] = np.uint64(1) << np.uint64(u + 1)
    barrier = 1 << 63  # the exact input that overflows int64
    reaches_nb = _bitset.reaches_impl(backend="numba")
    for start, target in [(0, 63), (0, 62), (63, 0)]:
        want = _bitset._semidirected_reaches_numpy(succ, start, target, barrier)
        assert bool(reaches_nb(succ, start, target, barrier)) == want, (start, target)


def test_ges_numpy_and_numba_agree():
    """GES returns the same CPDAG whether reachability runs on numpy or numba (same result)."""
    if backend.numba() is None:
        pytest.skip("numba (the [numba] extra) is not installed")
    from andrey.search.ges import ges

    x = np.random.default_rng(0).standard_normal((300, 6))
    with backend.config(backend="numpy"):
        cpdag_np, _ = ges(x)
    with backend.config(backend="numba"):
        cpdag_nb, _ = ges(x)
    assert cpdag_np == cpdag_nb  # GraphStructure value-equality: the same CPDAG


def test_ges_runs_without_numba(monkeypatch):
    """The default (no [numba]) install: numba is unavailable, GES uses the numpy reference."""
    monkeypatch.setattr(backend, "numba", lambda: None)
    assert backend.resolve_bitset() == "numpy"
    assert _bitset.reaches_impl() is _bitset._semidirected_reaches_numpy
    from andrey.search.ges import ges

    x = np.random.default_rng(1).standard_normal((200, 5))
    cpdag, _ = ges(x)
    assert cpdag.n_nodes == 5  # runs to completion on the numpy path


def _random_tail_adj(rng: np.random.Generator, d: int) -> np.ndarray:
    """A random ``(d, d)`` endpoint-mark matrix; only the ``TAIL`` marks drive reachability."""
    adj = (rng.random((d, d)) < 0.3).astype(np.int8) * TAIL
    np.fill_diagonal(adj, 0)
    return adj


@pytest.mark.parametrize("seed", range(40))
def test_setbfs_matches_oracle_across_d(seed):
    # The width-unlimited set BFS (operators.blocks_semi_directed_paths, the d > 64 GES fallback)
    # must agree with the independent bit-iterating oracle at every d, including past the uint64 cap
    # -- and with the uint64 bitmask BFS wherever that one is valid (d <= 64), start == target
    # included (all three treat the empty path as reachable).
    rng = np.random.default_rng(seed)
    d = int(rng.integers(2, 130))  # spans the d <= 64 packing cap
    adj = _random_tail_adj(rng, d)
    succ_lists = ops.tail_successor_lists(adj)
    pyint = [int(sum(1 << c for c in row)) for row in succ_lists]  # correct, un-truncated masks
    u64 = _bitset.tail_successor_masks(adj) if d <= 64 else None
    for _ in range(6):
        start = int(rng.integers(0, d))
        target = int(rng.integers(0, d))
        blocked = {int(v) for v in np.nonzero(rng.random(d) < 0.2)[0]}
        want = _reaches_setbased(pyint, start, target, _bitset.bitmask(blocked))
        got = not ops.blocks_semi_directed_paths(succ_lists, start, target, set(blocked))
        assert got == want, (seed, d, start, target, blocked)
        if u64 is not None:
            bitmask = _bitset._semidirected_reaches_numpy(
                u64, start, target, _bitset.bitmask(blocked)
            )
            assert bitmask == want, (seed, d, start, target, blocked)


def test_setbfs_path_through_high_bit():
    # The adversarial case for a single-uint64 packing: the only semi-directed path 0 ~> 1
    # runs through node 70, a bit no uint64 holds. The set BFS handles it; the packing raises.
    d = 80
    adj = np.zeros((d, d), dtype=np.int8)
    adj[0, 70] = TAIL
    adj[70, 1] = TAIL
    succ_lists = ops.tail_successor_lists(adj)
    assert not ops.blocks_semi_directed_paths(succ_lists, 0, 1, set())  # 1 reachable from 0 via 70
    assert ops.blocks_semi_directed_paths(succ_lists, 0, 1, {70})  # blocked once 70 is removed
    with pytest.raises(ValueError):  # past the width it raises, never truncates
        _bitset.tail_successor_masks(adj)
    pyint = [int(sum(1 << c for c in row)) for row in succ_lists]  # independent oracle agrees
    assert _reaches_setbased(pyint, 0, 1, 0) is True
    assert _reaches_setbased(pyint, 0, 1, _bitset.bitmask({70})) is False
    # start == target is the empty path: reachable, so nothing blocks it (matches the bitmask BFS).
    assert ops.blocks_semi_directed_paths(succ_lists, 5, 5, set()) is False
    assert ops.blocks_semi_directed_paths(succ_lists, 5, 5, {5}) is False
