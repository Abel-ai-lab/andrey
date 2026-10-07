"""The built-in layouts of ``andrey.viz``, used when Graphviz is not installed.

``layered`` is a longest-path (Sugiyama-style) layering from the directed edges, right for a DAG or
CPDAG; it returns ``None`` when the directed subgraph has a cycle, so the caller falls back to
``circular``. Both are deterministic (no randomness), so a structure always renders the same way.
``unit`` fits any positions into the unit square without changing their proportions.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

COLUMN, ROW = 100.0, 52.0  # a layer's width and a node's height in ``layered``, in pixels


def layered(n: int, directed: list[tuple[int, int]]) -> dict[int, tuple[float, float]] | None:
    """Longest-path layering from directed ``(src, dst)`` edges.

    Returns positions in pixels keyed by node, or ``None`` if the directed subgraph is cyclic.
    """
    succ: list[list[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for s, d in directed:
        succ[s].append(d)
        indeg[d] += 1

    layer = [0] * n
    remaining = indeg[:]
    q = deque(i for i in range(n) if remaining[i] == 0)
    seen = 0
    while q:
        u = q.popleft()
        seen += 1
        for v in succ[u]:
            if layer[u] + 1 > layer[v]:
                layer[v] = layer[u] + 1
            remaining[v] -= 1
            if remaining[v] == 0:
                q.append(v)
    if seen < n:  # a cycle blocked the topological sweep
        return None

    # Within-layer order is node index, with no crossing reduction.
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[layer[i]].append(i)
    height = (max(len(ids) for ids in groups.values()) - 1) * ROW
    pos: dict[int, tuple[float, float]] = {}
    for lyr, ids in groups.items():
        m = len(ids)
        for k, i in enumerate(ids):
            y = height / 2 if m == 1 else k * height / (m - 1)
            pos[i] = (lyr * COLUMN, y)
    return pos


def circular(n: int) -> dict[int, tuple[float, float]]:
    """Evenly spaced nodes on a circle -- the fallback for cyclic or circle-heavy graphs."""
    if n == 1:
        return {0: (0.5, 0.5)}  # a lone node sits at the canvas center, not on the ring
    pos: dict[int, tuple[float, float]] = {}
    for i in range(n):
        a = -math.pi / 2 + 2 * math.pi * i / max(n, 1)
        pos[i] = (0.5 + 0.42 * math.cos(a), 0.5 + 0.42 * math.sin(a))
    return pos


def unit(points: dict[int, tuple[float, float]]) -> dict[int, tuple[float, float]]:
    """``points`` scaled into the unit square, proportions kept, the shorter side centered."""
    if not points:
        return {}
    xs = [x for x, _ in points.values()]
    ys = [y for _, y in points.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    if span == 0:
        return dict.fromkeys(points, (0.5, 0.5))
    left = min(xs) - (span - (max(xs) - min(xs))) / 2
    top = min(ys) - (span - (max(ys) - min(ys))) / 2
    return {k: ((x - left) / span, (y - top) / span) for k, (x, y) in points.items()}
