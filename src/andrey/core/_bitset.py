"""Python-int bitmasks of node sets and successor-set reachability.

Bit ``v`` of a mask stands for node ``v``. :func:`row_masks` packs a boolean matrix into one mask
per row, and :func:`semidirected_reach_set` expands a breadth-first frontier with integer ORs, so
each step is exact for any node count.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


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
