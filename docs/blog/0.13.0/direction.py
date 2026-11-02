"""The 0.13.0 post's figure on why a functional causal model can tell cause from effect.

Three pairs ``x -> y``, one per column: a straight line with Gaussian noise, a straight line with
uniform noise, and a curve with Gaussian noise. The first row draws the data, the second the
residual of a smooth fit of ``y`` on ``x`` against ``x``, and the third the residual of a fit of
``x`` on ``y`` against ``y``. Each residual panel gives the p-value of an HSIC test of the residual
against the regressor: a small p-value means the residual still depends on it. Only in the first
column do both directions leave independent residuals, so only there the direction cannot be told.

The figure is inline SVG written next to this script as ``direction.html``, whose colors are
``--andrey-*`` tokens, so it follows the page's light or dark theme. Run with
``uv run python docs/blog/0.13.0/direction.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from andrey.core.independence import hsic_test

HERE = Path(__file__).parent
N = 400
DEGREE = 5  # the smooth fit: a polynomial of this degree
SEED = 1

PANEL = 150  # panel side, in SVG units
GAP = 28
LEFT = 92  # room for the row labels
TOP = 46  # room for the column titles
ROW_LABELS = ["Data", "Fit y from x", "Fit x from y"]


def pairs(rng: np.random.Generator) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The three pairs, each ``(title, x, y)`` with ``x`` the cause."""
    x = rng.normal(size=N)
    gaussian = ("Line, Gaussian noise", x, 0.8 * x + 0.6 * rng.normal(size=N))
    x = rng.uniform(-1.7, 1.7, size=N)
    uniform = ("Line, uniform noise", x, 0.8 * x + 0.6 * rng.uniform(-1.7, 1.7, size=N))
    x = rng.uniform(-2.0, 2.0, size=N)
    curve = ("Curve, Gaussian noise", x, 0.5 * x**3 - x + 0.5 * rng.normal(size=N))
    return [gaussian, uniform, curve]


def residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """What is left of ``b`` after a smooth fit on ``a``."""
    return b - np.polyval(np.polyfit(a, b, DEGREE), a)


def scale(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map ``v`` onto ``[lo, hi]`` with a margin."""
    low, high = v.min(), v.max()
    pad = 0.06 * (high - low)
    return lo + (v - low + pad) / (high - low + 2 * pad) * (hi - lo)


def panel(x0: float, y0: float, u: np.ndarray, v: np.ndarray, note: str | None) -> str:
    """One scatter panel at ``(x0, y0)``, with its note under it."""
    px = scale(u, x0, x0 + PANEL)
    py = scale(-v, y0, y0 + PANEL)
    dots = "".join(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="1.6"/>' for a, b in zip(px, py))
    out = [
        f'<rect x="{x0}" y="{y0}" width="{PANEL}" height="{PANEL}" class="fcm-frame"/>',
        f'<g class="fcm-dots">{dots}</g>',
    ]
    if note:
        out.append(
            f'<text x="{x0 + PANEL / 2}" y="{y0 + PANEL + 14}" class="fcm-note">{note}</text>'
        )
    return "".join(out)


def verdict(p: float) -> str:
    """The note under a residual panel."""
    shown = "p &lt; 0.001" if p < 0.001 else f"p = {p:.2f}"
    return f"{'independent' if p >= 0.05 else 'dependent'}, {shown}"


def figure() -> str:
    """The figure as HTML."""
    rng = np.random.default_rng(SEED)
    cols = pairs(rng)
    width = LEFT + 3 * PANEL + 2 * GAP + 8
    height = TOP + 3 * PANEL + 3 * GAP + 8
    parts = []
    for c, (title, x, y) in enumerate(cols):
        x0 = LEFT + c * (PANEL + GAP)
        parts.append(f'<text x="{x0 + PANEL / 2}" y="{TOP - 14}" class="fcm-title">{title}</text>')
        forward, backward = residual(x, y), residual(y, x)
        rows = [
            (x, y, None),
            (x, forward, verdict(hsic_test(x, forward))),
            (y, backward, verdict(hsic_test(y, backward))),
        ]
        for r, (u, v, note) in enumerate(rows):
            parts.append(panel(x0, TOP + r * (PANEL + GAP), u, v, note))
    for r, label in enumerate(ROW_LABELS):
        y_mid = TOP + r * (PANEL + GAP) + PANEL / 2
        parts.append(f'<text x="{LEFT - 12}" y="{y_mid}" class="fcm-row">{label}</text>')
    style = (
        "<style>"
        ".fcm-figure svg{width:100%;height:auto;max-width:640px;display:block;margin:0 auto}"
        ".fcm-frame{fill:none;stroke:var(--andrey-line)}"
        ".fcm-dots circle{fill:var(--andrey-categorical-1);fill-opacity:.45}"
        ".fcm-figure text{font-family:var(--andrey-font-sans);fill:var(--andrey-ink)}"
        ".fcm-title{font-size:12px;font-weight:600;text-anchor:middle}"
        ".fcm-row{font-size:11px;text-anchor:end;dominant-baseline:middle}"
        ".fcm-note{font-size:10.5px;text-anchor:middle;fill:var(--andrey-muted)!important}"
        "</style>"
    )
    svg = (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Residuals of fits in the '
        f'two directions for three pairs">{"".join(parts)}</svg>'
    )
    caption = (
        "<figcaption>Each column is a pair in which x causes y. The second row is what is left of "
        "y after a smooth fit on x, the third what is left of x after a fit on y, each tested "
        "against the variable it was fit on.</figcaption>"
    )
    return f'{style}<figure class="fcm-figure">{svg}{caption}</figure>\n'


if __name__ == "__main__":
    (HERE / "direction.html").write_text(figure())
