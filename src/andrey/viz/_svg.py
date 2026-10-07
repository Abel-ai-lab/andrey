"""Pure-Python SVG-string emitter for ``andrey.viz`` scenes. No third-party dependencies.

Every mark is drawn by shape -- an arrowhead (``ARROW``), a hollow circle (``CIRCLE``), or a plain
tail (nothing) -- so the grammar is colorblind-safe and the emitted string is self-contained. A
2-cycle is drawn as two parallel curved edges (feedback), distinct from a single double-headed
bidirected edge (a latent confounder). An edge that would cross another node bends around it.
"""

from __future__ import annotations

import math
from html import escape

from andrey.core.structure import ARROW, CIRCLE, TAIL

from .theme import EDGE_COLOR, Palette

# geometry
NODE_R = 13.0
ALEN = 10.0  # arrowhead length
AW = 5.0  # arrowhead half-width
CR = 4.2  # circle-endpoint radius
SW = 2.0  # stroke width
PAD = 32.0  # canvas padding
HALO_R = 9.0  # ring around a highlighted endpoint
CLEAR = NODE_R + 4.0  # minimum distance from an edge to a node it does not join
_BEND_STEP = NODE_R
_BEND_STEPS = 8
_SAMPLES = 16  # line segments approximating an arc for the clearance check


def _f(v: float) -> str:
    return f"{v:.2f}"


def _line(x1: float, y1: float, x2: float, y2: float, color: str) -> str:
    return (
        f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" '
        f'stroke="{color}" stroke-width="{SW}" stroke-linecap="round"/>'
    )


def _poly(points: list[tuple[float, float]], fill: str) -> str:
    pts = " ".join(f"{_f(x)},{_f(y)}" for x, y in points)
    return f'<polygon points="{pts}" fill="{fill}" stroke-linejoin="round"/>'


def _circle(
    cx: float,
    cy: float,
    r: float,
    fill: str,
    stroke: str | None = None,
    dash: str | None = None,
    sw: float = 1.8,
) -> str:
    s = f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(r)}" fill="{fill}"'
    if stroke:
        s += f' stroke="{stroke}" stroke-width="{sw}"'
    if dash:
        s += f' stroke-dasharray="{dash}"'
    return s + "/>"


def _path(d: str, stroke: str) -> str:
    return (
        f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{SW}" stroke-linecap="round"/>'
    )


LABEL_W = 7.0  # the widest a label character draws at font-size 11, weight 600
LABEL_H = 16.0  # a label's drop below its node: 12 to the baseline, 4 of descender


def footprint(label: str, inside: bool) -> tuple[float, float]:
    """Width and height of what a node draws: its circle, and the label under it unless inside."""
    if inside:
        return 2 * NODE_R, 2 * NODE_R
    return max(2 * NODE_R, len(label) * LABEL_W), 2 * (NODE_R + LABEL_H)


def _text(x: float, y: float, s: str, color: str, *, center: bool = False) -> str:
    baseline = ' dominant-baseline="central"' if center else ""
    return (
        f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="middle" font-size="11" font-weight="600" '
        f'fill="{color}"{baseline}>{escape(s)}</text>'
    )


def _head(px: float, py: float, dx: float, dy: float, glyph: str, color: str, surface: str) -> str:
    """Draw the endpoint glyph at node boundary ``(px, py)``; ``(dx, dy)`` points into the edge."""
    if glyph == "arrow":
        nx, ny = -dy, dx
        b1 = (px + dx * ALEN + nx * AW, py + dy * ALEN + ny * AW)
        b2 = (px + dx * ALEN - nx * AW, py + dy * ALEN - ny * AW)
        return _poly([(px, py), b1, b2], color)
    if glyph == "circle":
        return _circle(px + dx * CR, py + dy * CR, CR, surface, color, sw=SW)
    return ""


def _glyph(mark: int) -> str:
    return {ARROW: "arrow", CIRCLE: "circle"}.get(mark, "tail")


def _stop(px: float, py: float, dx: float, dy: float, mark: int) -> tuple[float, float]:
    if mark == ARROW:
        return px + dx * ALEN, py + dy * ALEN
    if mark == CIRCLE:
        return px + dx * 2 * CR, py + dy * 2 * CR
    return px, py


def _control(ax: float, ay: float, bx: float, by: float, bend: float) -> tuple[float, float]:
    """Place the quadratic control point ``bend`` px left of the A -> B midpoint."""
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy) or 1.0
    return (ax + bx) / 2 - dy / L * bend, (ay + by) / 2 + dx / L * bend


def _edge(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    ma: int,
    mb: int,
    color: str,
    palette: Palette,
    *,
    bend: float = 0.0,
    halo: tuple[bool, bool] = (False, False),
) -> str:
    """Draw marks ``ma`` at A and ``mb`` at B on a line, or on a quadratic arc when ``bend`` is set.

    A haloed end draws its mark in the accent color inside a ring.
    """
    mx, my = _control(ax, ay, bx, by, bend)
    ends = []
    for x, y, mark, ringed in ((ax, ay, ma, halo[0]), (bx, by, mb, halo[1])):
        length = math.hypot(mx - x, my - y) or 1.0
        ux, uy = (mx - x) / length, (my - y) / length
        ends.append((x + ux * NODE_R, y + uy * NODE_R, ux, uy, mark, ringed))
    (sax, say), (sbx, sby) = (_stop(px, py, ux, uy, mark) for px, py, ux, uy, mark, _ in ends)
    if bend:
        parts = [_path(f"M {_f(sax)} {_f(say)} Q {_f(mx)} {_f(my)} {_f(sbx)} {_f(sby)}", color)]
    else:
        parts = [_line(sax, say, sbx, sby, color)]
    for px, py, ux, uy, mark, ringed in ends:
        fill = palette.accent if ringed else color
        parts.append(_head(px, py, ux, uy, _glyph(mark), fill, palette.surface))
        if ringed:
            reach = {ARROW: ALEN / 2, CIRCLE: CR}.get(mark, 0.0)
            parts.append(
                f'<circle class="highlight" cx="{_f(px + ux * reach)}" cy="{_f(py + uy * reach)}" '
                f'r="{_f(HALO_R)}" fill="none" stroke="{palette.accent}" stroke-width="{SW}"/>'
            )
    return "".join(parts)


def _clearance(
    ax: float, ay: float, bx: float, by: float, bend: float, others: list[tuple[float, float]]
) -> float:
    """Measure the smallest distance from any of ``others`` to the A-B line or arc."""
    mx, my = _control(ax, ay, bx, by, bend)
    points = [
        (
            (1 - t) ** 2 * ax + 2 * t * (1 - t) * mx + t**2 * bx,
            (1 - t) ** 2 * ay + 2 * t * (1 - t) * my + t**2 * by,
        )
        for t in (k / _SAMPLES for k in range(_SAMPLES + 1))
    ]
    nearest = math.inf
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        sx, sy = x2 - x1, y2 - y1
        length2 = sx * sx + sy * sy or 1.0
        for px, py in others:
            t = min(1.0, max(0.0, ((px - x1) * sx + (py - y1) * sy) / length2))
            nearest = min(nearest, math.hypot(x1 + t * sx - px, y1 + t * sy - py))
    return nearest


def _route(ax: float, ay: float, bx: float, by: float, others: list[tuple[float, float]]) -> float:
    """Pick the smallest bend that keeps the edge clear of every non-endpoint node.

    A straight edge through another node would read as two edges meeting there. When no candidate
    clears every node, the bend with the widest clearance wins.
    """
    reach = _BEND_STEP * _BEND_STEPS / 2 + CLEAR
    near = [
        (px, py)
        for px, py in others
        if min(ax, bx) - reach <= px <= max(ax, bx) + reach
        and min(ay, by) - reach <= py <= max(ay, by) + reach
    ]
    best, widest = 0.0, -1.0
    for bend in [0.0] + [s * k * _BEND_STEP for k in range(1, _BEND_STEPS + 1) for s in (1, -1)]:
        clearance = _clearance(ax, ay, bx, by, bend, near) if near else math.inf
        if clearance >= CLEAR:
            return bend
        if clearance > widest:
            best, widest = bend, clearance
    return best


def _widen(
    ax: float, ay: float, bx: float, by: float, curv: float, others: list[tuple[float, float]]
) -> float:
    """Widen a 2-cycle's pair of opposite arcs until both clear every non-endpoint node."""
    best, widest = curv, -1.0
    for bend in (curv + k * _BEND_STEP for k in range(_BEND_STEPS + 1)):
        clearance = min(
            _clearance(ax, ay, bx, by, bend, others), _clearance(bx, by, ax, ay, bend, others)
        )
        if clearance >= CLEAR:
            return bend
        if clearance > widest:
            best, widest = bend, clearance
    return best


def _self_loop(x: float, y: float, color: str) -> str:
    """Draw a small loop above a node, arrowhead returning into it (a temporal / digraph loop)."""
    top = y - NODE_R
    d = (
        f"M {_f(x - 4)} {_f(top)} C {_f(x - 13)} {_f(top - 16)} "
        f"{_f(x + 13)} {_f(top - 16)} {_f(x + 4)} {_f(top)}"
    )
    tri = [(x + 4, top), (x + 0.5, top - 7), (x + 9, top - 6)]
    return _path(d, color) + _poly(tri, color)


def _node(
    x: float, y: float, label: str, palette: Palette, latent: bool, *, label_inside: bool
) -> str:
    if latent:
        node = _circle(x, y, NODE_R, palette.paper, palette.muted, dash="3 3")
    else:
        node = _circle(x, y, NODE_R, palette.surface, palette.ink)
    if not label:
        return node
    if label_inside:
        color = palette.muted if latent else palette.ink
        return node + _text(x, y, label, color, center=True)
    return node + _text(x, y + NODE_R + 12, label, palette.muted)


def render(
    *,
    n: int,
    pos: dict[int, tuple[float, float]],
    edges: list[tuple[int, int, int, int, str]],
    labels: list[str],
    latent: set[int],
    width: float,
    height: float,
    palette: Palette,
    background: bool = True,
    highlight: frozenset[tuple[int, int]] = frozenset(),
) -> str:
    """Emit the full SVG string. Optional background, then edges, then nodes on top.

    ``highlight`` holds ``(node, neighbor)`` pairs: the mark at ``node`` on that edge is ringed.
    """
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="100%" font-family="Inter, system-ui, sans-serif">'
    ]
    if background:
        parts.append(f'<rect width="{width:.0f}" height="{height:.0f}" fill="{palette.paper}"/>')
    for i, j, mi, mj, et in edges:
        color = getattr(palette, EDGE_COLOR[et])
        ax, ay = pos[i]
        if i == j:
            parts.append(_self_loop(ax, ay, palette.edge_directed))  # a loop is a directed edge
            continue
        bx, by = pos[j]
        at_i, at_j = (i, j) in highlight, (j, i) in highlight
        others = [pos[k] for k in range(n) if k != i and k != j]
        if et == "twocycle":
            curv = _widen(ax, ay, bx, by, max(12.0, 0.14 * math.hypot(bx - ax, by - ay)), others)
            for (x1, y1, x2, y2), ringed in (((ax, ay, bx, by), at_j), ((bx, by, ax, ay), at_i)):
                parts.append(
                    _edge(
                        x1, y1, x2, y2, TAIL, ARROW, color, palette, bend=curv, halo=(False, ringed)
                    )
                )
        else:
            bend = _route(ax, ay, bx, by, others)
            parts.append(
                _edge(ax, ay, bx, by, mi, mj, color, palette, bend=bend, halo=(at_i, at_j))
            )
    # Use one placement for the figure; short labels fit clear of endpoint marks inside nodes.
    labels_inside = all(len(label) <= 2 for label in labels)
    for k in range(n):
        x, y = pos[k]
        parts.append(_node(x, y, labels[k], palette, k in latent, label_inside=labels_inside))
    parts.append("</svg>")
    return "".join(parts)
