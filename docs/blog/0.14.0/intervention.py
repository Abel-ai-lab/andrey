"""The 0.14.0 post's figure on why an intervention shows which way an effect runs.

One pair ``x -> y`` with ``y = 0.8 x + e`` and Gaussian ``x`` and ``e``, in three panels: (a) the
observational rows alone, (b) those rows and rows under ``do(x)``, and (c) those rows and rows
under ``do(y)``. Under (b) and (c), a two-sample Kolmogorov-Smirnov test asks whether the variable
that was not intervened on changed between the observational and the interventional rows: an
intervention on the cause moves the effect, and one on the effect leaves the cause alone.

The figure is inline SVG written next to this script as ``intervention.html``, whose colors are
``--andrey-*`` tokens, so it follows the page's light or dark theme. Run with
``uv run python docs/blog/0.14.0/intervention.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

HERE = Path(__file__).parent
N = 300
SEED = 0
SET_MEAN, SET_SD = 2.5, 0.5  # the value an intervention sets, and its spread

PANEL = 170  # panel side, in SVG units
GAP = 30
LEFT = 12
TOP = 34  # room for the panel titles
LO, HI = -4.0, 5.0  # the shared axis range of x and y


def data(rng: np.random.Generator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The observational rows and the rows of each intervention, each ``(x, y)``."""
    x = rng.normal(size=N)
    observed = (x, 0.8 * x + 0.6 * rng.normal(size=N))
    x = rng.normal(SET_MEAN, SET_SD, size=N)
    set_x = (x, 0.8 * x + 0.6 * rng.normal(size=N))
    set_y = (rng.normal(size=N), rng.normal(SET_MEAN, SET_SD, size=N))
    return {"observed": observed, "set_x": set_x, "set_y": set_y}


def to_svg(v: np.ndarray, lo: float, hi: float, flip: bool = False) -> np.ndarray:
    """Map ``v`` from the shared axis range onto ``[lo, hi]``."""
    t = (np.clip(v, LO, HI) - LO) / (HI - LO)
    return lo + (1 - t if flip else t) * (hi - lo)


def dots(x0: float, y0: float, xy: tuple[np.ndarray, np.ndarray], cls: str) -> str:
    """One cloud of points in the panel at ``(x0, y0)``."""
    px = to_svg(xy[0], x0, x0 + PANEL)
    py = to_svg(xy[1], y0, y0 + PANEL, flip=True)
    return (
        f'<g class="{cls}">'
        + "".join(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="1.6"/>' for a, b in zip(px, py))
        + "</g>"
    )


def verdict(name: str, before: np.ndarray, after: np.ndarray) -> str:
    """The note under an interventional panel: did the variable not intervened on change?"""
    p = ks_2samp(before, after).pvalue
    shown = "p &lt; 0.001" if p < 0.001 else f"p = {p:.2f}"
    return f"{name} {'changes' if p < 0.05 else 'does not change'}, {shown}"


def figure() -> str:
    """The figure as HTML."""
    d = data(np.random.default_rng(SEED))
    observed = d["observed"]
    panels = [
        ("(a) Observation", None, "x -> y and y -> x fit alike"),
        ("(b) do(x)", d["set_x"], verdict("y", observed[1], d["set_x"][1])),
        ("(c) do(y)", d["set_y"], verdict("x", observed[0], d["set_y"][0])),
    ]
    width = LEFT + 3 * PANEL + 2 * GAP + 8
    height = TOP + PANEL + 44
    parts = []
    for c, (title, intervened, note) in enumerate(panels):
        x0 = LEFT + c * (PANEL + GAP)
        mid = x0 + PANEL / 2
        parts.append(f'<text x="{mid}" y="{TOP - 14}" class="iv-title">{title}</text>')
        frame = f'x="{x0}" y="{TOP}" width="{PANEL}" height="{PANEL}"'
        parts.append(f'<rect {frame} class="iv-frame"/>')
        parts.append(dots(x0, TOP, observed, "iv-observed" if intervened is not None else "iv-set"))
        if intervened is not None:
            parts.append(dots(x0, TOP, intervened, "iv-set"))
        parts.append(f'<text x="{mid}" y="{TOP + PANEL + 16}" class="iv-axis">x</text>')
        parts.append(f'<text x="{x0 - 5}" y="{TOP + PANEL / 2}" class="iv-axis">y</text>')
        parts.append(f'<text x="{mid}" y="{TOP + PANEL + 34}" class="iv-note">{note}</text>')
    style = (
        "<style>"
        ".iv-figure svg{width:100%;height:auto;max-width:640px;display:block;margin:0 auto}"
        ".iv-frame{fill:none;stroke:var(--andrey-line)}"
        ".iv-set circle{fill:var(--andrey-categorical-1);fill-opacity:.5}"
        ".iv-observed circle{fill:var(--andrey-muted);fill-opacity:.3}"
        ".iv-figure text{font-family:var(--andrey-font-sans);fill:var(--andrey-ink)}"
        ".iv-title{font-size:12px;font-weight:600;text-anchor:middle}"
        ".iv-axis{font-size:11px;font-style:italic;text-anchor:middle;dominant-baseline:middle;"
        "fill:var(--andrey-muted)!important}"
        ".iv-note{font-size:10.5px;text-anchor:middle;fill:var(--andrey-muted)!important}"
        "</style>"
    )
    svg = (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="A pair x causes y, observed, '
        f'under do(x), and under do(y)">{"".join(parts)}</svg>'
    )
    caption = (
        "<figcaption>One pair in which x causes y. In (b) and (c), the gray points are the "
        "observational rows of (a), and the blue points the rows of the intervention.</figcaption>"
    )
    return f'{style}<figure class="iv-figure">{svg}{caption}</figure>\n'


if __name__ == "__main__":
    (HERE / "intervention.html").write_text(figure())
