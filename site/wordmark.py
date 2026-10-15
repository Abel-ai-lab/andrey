"""Write the Andrey wordmark and icon: the word "Andrey" in Unbounded, as outlines, in the
palette's ink, its A's right stroke in amber; the icon is that A alone.

Outlines render the same with or without the font installed. The light and dark wordmarks take
their ink from ``andrey.viz.LIGHT.ink`` and ``andrey.viz.DARK.ink``; the amber is the art's
(``andrey.viz.LIGHT.categorical[1]``). The A stays one letter, its apex joined and its crossbar
whole: only its right stroke, cut along the counter's inner edge, changes color. Rerun after a
palette change:

    uv run --with fonttools --with uharfbuzz --with skia-pathops python site/wordmark.py \\
        Unbounded[wght].ttf

The font file comes from Google Fonts (``ofl/unbounded``, SIL Open Font License); it is not kept
in the repository.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pathops
import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

import andrey.viz

WORD = "Andrey"
WEIGHT = 600  # Unbounded's SemiBold
HEIGHT_PX = 40  # intrinsic height; the viewBox keeps it scalable
BRAND = Path(__file__).resolve().parent / "brand"
AMBER = andrey.viz.LIGHT.categorical[1]


def instance(font_path: Path) -> bytes:
    """The font at ``WEIGHT``, as font-file bytes (a static font passes through)."""
    font = TTFont(font_path)
    if "fvar" in font:
        font = instantiateVariableFont(font, {"wght": WEIGHT})
    out = io.BytesIO()
    font.save(out)
    return out.getvalue()


def glyph_path(glyphs, name: str, offset: tuple[float, float]) -> pathops.Path:
    """A glyph's outline, its overlaps merged, moved by ``offset``, in font units (y up)."""
    path = pathops.Path()
    glyphs[name].draw(TransformPen(path.getPen(), (1, 0, 0, 1, *offset)))
    return pathops.simplify(path)


def right_stroke(a: pathops.Path) -> pathops.Path:
    """The A's right stroke: the part beyond the line along the counter's right edge (its inner
    edge), extended through the letter, which leaves the crossbar and the left stroke whole."""
    counter = min(
        (pathops.Path(contour) for contour in a.contours), key=lambda contour: abs(contour.area)
    )
    points = [point for _, segment in counter.segments for point in segment]
    (x0, y0) = max(points, key=lambda p: p[1])  # its top
    (x1, y1) = max(points, key=lambda p: p[0] - 0.2 * p[1])  # its lower right
    dx, dy = x1 - x0, y1 - y0
    reach = 50.0
    far = 1e5 / (dx * dx + dy * dy) ** 0.5
    nx, ny = -dy * far, dx * far  # toward +x, the stroke's outer side
    a0, a1 = (x0 - dx * reach, y0 - dy * reach), (x0 + dx * reach, y0 + dy * reach)
    half = pathops.Path()
    pen = half.getPen()
    pen.moveTo(a0)
    pen.lineTo(a1)
    pen.lineTo((a1[0] + nx, a1[1] + ny))
    pen.lineTo((a0[0] + nx, a0[1] + ny))
    pen.closePath()
    return pathops.op(a, half, pathops.PathOp.INTERSECTION)


def outline(font_path: Path) -> tuple[pathops.Path, pathops.Path, pathops.Path]:
    """Shape the word with its kerning: the whole word, its A, and the A's right stroke, in font
    units (y up)."""
    data = instance(font_path)
    shaped = hb.Buffer()
    shaped.add_str(WORD)
    shaped.guess_segment_properties()
    hb.shape(hb.Font(hb.Face(hb.Blob(data))), shaped, {"kern": True, "liga": True})
    font = TTFont(io.BytesIO(data))
    glyphs, order = font.getGlyphSet(), font.getGlyphOrder()
    word, x, first = pathops.Path(), 0, None
    for info, pos in zip(shaped.glyph_infos, shaped.glyph_positions, strict=True):
        glyph = glyph_path(glyphs, order[info.codepoint], (x + pos.x_offset, pos.y_offset))
        first = first or glyph
        word = pathops.op(word, glyph, pathops.PathOp.UNION)
        x += pos.x_advance
    return word, first, right_stroke(first)


def svg_d(path: pathops.Path) -> str:
    """``path`` as SVG path data, flipped so y points down."""
    pen = SVGPathPen(None, ntos=lambda v: f"{v:.1f}".rstrip("0").rstrip("."))
    path.draw(TransformPen(pen, (1, 0, 0, -1, 0, 0)))
    return pen.getCommands()


def document(
    paths: list[tuple[str, str]], bounds: tuple[float, float, float, float], label: str
) -> str:
    """The ``(fill, path data)`` pairs in a standalone SVG with a tight viewBox, y down."""
    x0, y0, x1, y1 = bounds
    w, h = x1 - x0, y1 - y0
    width = round(HEIGHT_PX * w / h, 1)
    body = "".join(f'<path fill="{fill}" d="{d}"/>' for fill, d in paths)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:g} {-y1:g} {w:g} {h:g}" '
        f'width="{width:g}" height="{HEIGHT_PX}" role="img" aria-label="{label}">{body}</svg>\n'
    )


def icon(a: pathops.Path, stroke: pathops.Path) -> str:
    """The A alone on a square, its ink following the browser's light or dark bar."""
    x0, y0, x1, y1 = a.bounds
    side = max(x1 - x0, y1 - y0) * 1.12
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rest = pathops.op(a, stroke, pathops.PathOp.DIFFERENCE)
    light, dark = andrey.viz.LIGHT.ink, andrey.viz.DARK.ink
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{cx - side / 2:g} {-cy - side / 2:g} '
        f'{side:g} {side:g}"><style>.ink{{fill:{light}}}@media (prefers-color-scheme: dark)'
        f"{{.ink{{fill:{dark}}}}}</style>"
        f'<path class="ink" d="{svg_d(rest)}"/><path fill="{AMBER}" d="{svg_d(stroke)}"/></svg>\n'
    )


def main() -> None:
    """Write the light and dark wordmarks and the icon (``brand/favicon.svg``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("font", type=Path, help="Unbounded's .ttf (variable or SemiBold)")
    word, a, stroke = outline(parser.parse_args().font)
    BRAND.mkdir(exist_ok=True)
    for name, palette in (("light", andrey.viz.LIGHT), ("dark", andrey.viz.DARK)):
        paths = [(palette.ink, svg_d(word)), (AMBER, svg_d(stroke))]
        (BRAND / f"wordmark-{name}.svg").write_text(document(paths, word.bounds, WORD))
    (BRAND / "favicon.svg").write_text(icon(a, stroke))


if __name__ == "__main__":
    main()
