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


#: Above this many set bits, :func:`nodes` reads the mask's bytes with NumPy instead of looping.
_UNPACK_BITS = 24


def nodes(mask: int) -> list[int]:
    """The nodes whose bits are set in ``mask``, ascending: the inverse of :func:`bitmask`."""
    if mask.bit_count() > _UNPACK_BITS:
        raw = np.frombuffer(mask.to_bytes((mask.bit_length() + 7) // 8, "little"), dtype=np.uint8)
        return np.flatnonzero(np.unpackbits(raw, bitorder="little")).tolist()
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def row_masks(matrix: np.ndarray) -> list[int]:
    """Each row of a square boolean matrix as a Python int, bit ``j`` set iff ``matrix[i, j]``.

    Each row packs to bytes with column ``j`` at bit ``j % 8`` of byte ``j // 8``, so its bytes
    read little-endian as the int with bit ``j`` at column ``j``.
    """
    packed = np.packbits(np.ascontiguousarray(matrix), axis=1, bitorder="little")
    rows = packed.view(f"V{packed.shape[1]}").ravel().tolist()
    from_bytes = int.from_bytes
    return [from_bytes(row, "little") for row in rows]


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
        for u in nodes(frontier):
            nxt |= succ[u]
        frontier = nxt & allowed & ~seen
        seen |= frontier
    return seen
