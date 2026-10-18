"""Bitset reachability kernels over dense endpoint-mark graphs.

A graph on ``d <= 64`` nodes packs each node's out-neighbors into a single ``uint64``: bit ``c`` of
``succ[node]`` is set iff the edge ``node -> c`` is present. Reachability then reduces to a
breadth-first OR-expansion of these bitmasks, one machine word per frontier step. Every operation is
exact integer arithmetic, so every backend returns the same booleans.

Two backends share one contract (see :func:`semidirected_reaches`): a pure-Python int-bitmask BFS
(:func:`_semidirected_reaches_numpy`, the always-present reference and the default install's path)
and a :func:`numba.njit`-compiled ``uint64`` kernel (the ``numba`` backend, compiled lazily only
when selected). :func:`reaches_impl` resolves the pair for the active backend; hot loops hoist once.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import numpy as np

from . import backend as _backend
from .structure import TAIL


def tail_successor_masks(adj: np.ndarray) -> np.ndarray:
    """Pack each node's semi-directed successors into a per-node ``uint64`` bitmask.

    ``adj`` is a dense ``(d, d)`` endpoint-mark matrix; a semi-directed successor of ``node`` is any
    ``c`` with ``adj[node, c] == TAIL`` (a directed ``node -> c`` or an undirected ``node -- c``).
    Bit ``c`` of the returned ``succ[node]`` is set iff ``c`` is such a successor. Requires
    ``d <= 64`` (bit ``c >= 64`` cannot fit a ``uint64``); raises above it rather than truncating
    silently -- use the width-unlimited :func:`andrey.search.operators.blocks_semi_directed_paths`.
    """
    d = adj.shape[0]
    if d > 64:
        raise ValueError(
            f"tail_successor_masks packs successors into one uint64, so it requires d <= 64; got "
            f"d={d}. Above 64 use a width-unlimited set traversal over successor lists instead."
        )
    weights = np.uint64(1) << np.arange(d, dtype=np.uint64)
    return ((adj == TAIL).astype(np.uint64) * weights).sum(axis=1)


def bitmask(nodes: Iterable[int]) -> int:
    """Return the integer with bit ``v`` set for each ``v`` in ``nodes``."""
    m = 0
    for v in nodes:
        m |= 1 << int(v)
    return m


def nodes(mask: int) -> list[int]:
    """The nodes whose bits are set in ``mask``, ascending: the inverse of :func:`bitmask`."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def row_masks(matrix: np.ndarray) -> list[int]:
    """Each row of a square boolean matrix as a Python int, bit ``j`` set iff ``matrix[i, j]``.

    Each row is reversed and left-padded to whole bytes, so its packed big-endian bytes read as the
    int with bit ``j`` at column ``j``.
    """
    n = matrix.shape[0]
    width = (n + 7) // 8
    padded = np.zeros((n, 8 * width), dtype=bool)
    padded[:, 8 * width - n :] = matrix[:, ::-1]
    return list(map(int.from_bytes, np.packbits(padded, axis=1).view(f"V{width}").ravel().tolist()))


def semidirected_reach_set(succ: list[int], start: int, barrier: int) -> int:
    """Every node reachable from ``start`` once the ``barrier`` nodes are removed, as a bitmask.

    ``succ`` holds each node's successors as a Python int, as :func:`row_masks` packs them, so the
    breadth-first OR-expansion holds for any node count. ``start`` is never removed and always in
    the result. For ``t`` outside ``barrier``, bit ``t`` is set iff a successor path from ``start``
    reaches ``t``: one traversal answers every target that shares a start and a barrier.
    """
    allowed = ~barrier
    seen = 1 << start
    frontier = seen
    while frontier:
        nxt = 0
        bits = frontier
        while bits:
            lsb = bits & (-bits)
            nxt |= succ[lsb.bit_length() - 1]
            bits ^= lsb
        frontier = nxt & allowed & ~seen
        seen |= frontier
    return seen


def _semidirected_reaches_numpy(succ: np.ndarray, start: int, target: int, barrier: int) -> bool:
    """Pure-Python int-bitmask BFS: the always-present reference and the default-install path.

    Same contract as :func:`semidirected_reaches`, over Python ints (fast when interpreted, unlike a
    ``uint64`` kernel run without numba). ``start`` / ``target`` are never treated as removed.
    """
    if start == target:
        return True
    start_bit = 1 << start
    target_bit = 1 << target
    bar = int(barrier) & ~(start_bit | target_bit)
    visited = start_bit
    frontier = start_bit
    while frontier:
        nxt = 0
        bits = frontier
        while bits:
            lsb = bits & (-bits)
            idx = lsb.bit_length() - 1
            nxt |= int(succ[idx])
            bits ^= lsb
        nxt &= ~bar & ~visited
        if nxt & target_bit:
            return True
        visited |= nxt
        frontier = nxt
    return False


def _semidirected_reaches_kernel(succ, start, target, barrier):
    """The bitmask BFS as branch-free ``uint64`` arithmetic; :func:`numba.njit` lowers it."""
    if start == target:
        return True
    one = np.uint64(1)
    zero = np.uint64(0)
    start_bit = one << np.uint64(start)
    target_bit = one << np.uint64(target)
    bar = np.uint64(barrier) & ~(start_bit | target_bit)
    visited = start_bit
    frontier = start_bit
    while frontier != zero:
        nxt = zero
        bits = frontier
        while bits != zero:
            lsb = bits & (~bits + one)
            idx = 0
            probe = lsb
            while probe > one:
                probe = probe >> one
                idx += 1
            nxt |= succ[idx]
            bits = bits ^ lsb
        nxt = nxt & ~bar & ~visited
        if (nxt & target_bit) != zero:
            return True
        visited = visited | nxt
        frontier = nxt
    return False


@lru_cache(maxsize=1)
def _numba_reaches():
    """Lazily :func:`numba.njit`-compile the kernel (only when numba is the chosen backend)."""
    nb = _backend.numba()
    return nb.njit(cache=True)(_semidirected_reaches_kernel)


def reaches_impl(*, backend: str | None = None):
    """Resolve the reachability impl for the active backend (numba when chosen, else numpy).

    Hoist once at the start of an algorithm run and call the returned function in the hot loop, so
    the backend is resolved once per run rather than once per call. The numba callable casts the
    barrier to ``uint64`` at the boundary: numba types a Python-int argument as ``int64``, which
    overflows a barrier with bit 63 set (``d`` up to 64), so both backends accept the same
    in-contract Python-int input.
    """
    if _backend.resolve_bitset(backend=backend) == "numba":
        kernel = _numba_reaches()

        def _reaches_numba(succ, start, target, barrier):
            return kernel(succ, start, target, np.uint64(barrier))

        return _reaches_numba
    return _semidirected_reaches_numpy


def semidirected_reaches(
    succ: np.ndarray, start: int, target: int, barrier: int, *, backend: str | None = None
) -> bool:
    """True iff ``target`` is reachable from ``start`` over successor edges, skipping ``barrier``.

    Resolves the backend per call; hot loops should hoist :func:`reaches_impl` instead. ``succ`` is
    the per-node successor bitmask array from :func:`tail_successor_masks`; ``barrier`` is a bitmask
    of nodes removed from the graph before the traversal (``start`` / ``target`` are never removed
    even when their bits appear in ``barrier``). Expands the frontier one breadth-first step at a
    time, stopping as soon as ``target`` enters the frontier.
    """
    return reaches_impl(backend=backend)(succ, start, target, barrier)
