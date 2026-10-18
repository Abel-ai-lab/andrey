"""Node sets stored as Python integers, with traversal along semi-directed paths.

A set bit ``v`` means that node ``v`` belongs to the set. :func:`row_masks` converts a boolean
matrix to one such integer per row. :func:`semidirected_reach_set` follows the successors in
these sets to find reachable nodes while excluding a specified set of blocked nodes. Python
integers allow these sets to represent graphs with more than 64 nodes without losing bits.
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

    Bit ``v`` in ``succ[u]`` means that traversal from node ``u`` to node ``v`` is allowed;
    :func:`row_masks` builds these integers from a boolean successor matrix. Set bits in
    ``barrier`` identify blocked nodes. ``start`` is never removed and is always in the result,
    even if its bit is set in ``barrier``. For every other node ``t``, bit ``t`` in the result is
    set exactly when a path from ``start`` reaches it without entering a blocked node. One
    traversal answers every target with the same start and barrier, for any node count.
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
