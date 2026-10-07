"""Write the Andrey wordmark: the word "Andrey" in Inter, as outlines, in the palette's ink.

Outlines render the same with or without the font installed. The light and dark files take their
fill from ``andrey.viz.LIGHT.ink`` and ``andrey.viz.DARK.ink``. Rerun after a palette change:

    uv run --with fonttools --with uharfbuzz python site/wordmark.py InterDisplay-SemiBold.otf

The font file comes from the Inter release (``extras/otf/``); it is not kept in the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

import andrey.viz

WORD = "Andrey"
HEIGHT_PX = 40  # intrinsic height; the viewBox keeps it scalable
BRAND = Path(__file__).resolve().parent / "brand"


def outline(font_path: Path) -> tuple[str, tuple[float, float, float, float]]:
    """Shape the word with its kerning and return one SVG path and its bounds in font units."""
    blob = hb.Blob.from_file_path(str(font_path))
    shaped = hb.Buffer()
    shaped.add_str(WORD)
    shaped.guess_segment_properties()
    hb.shape(hb.Font(hb.Face(blob)), shaped, {"kern": True, "liga": True})

    font = TTFont(font_path)
    glyphs = font.getGlyphSet()
    order = font.getGlyphOrder()
    svg = SVGPathPen(glyphs)
    bounds = BoundsPen(glyphs)
    x = 0
    for info, pos in zip(shaped.glyph_infos, shaped.glyph_positions):
        glyph = glyphs[order[info.codepoint]]
        flip = (1, 0, 0, -1, x + pos.x_offset, -pos.y_offset)  # font y points up, SVG y down
        glyph.draw(TransformPen(svg, flip))
        glyph.draw(TransformPen(bounds, flip))
        x += pos.x_advance
    return svg.getCommands(), bounds.bounds


def document(path: str, bounds: tuple[float, float, float, float], fill: str) -> str:
    """Wrap the outline in a standalone SVG with a tight viewBox."""
    x0, y0, x1, y1 = bounds
    w, h = x1 - x0, y1 - y0
    width = round(HEIGHT_PX * w / h, 1)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:g} {y0:g} {w:g} {h:g}" '
        f'width="{width:g}" height="{HEIGHT_PX}" role="img" aria-label="{WORD}">'
        f'<path fill="{fill}" d="{path}"/></svg>\n'
    )


def main() -> None:
    """Write ``brand/wordmark-light.svg`` and ``brand/wordmark-dark.svg``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("font", type=Path, help="an Inter .otf or .ttf file")
    path, bounds = outline(parser.parse_args().font)
    BRAND.mkdir(exist_ok=True)
    for name, palette in (("light", andrey.viz.LIGHT), ("dark", andrey.viz.DARK)):
        (BRAND / f"wordmark-{name}.svg").write_text(document(path, bounds, palette.ink))


if __name__ == "__main__":
    main()
