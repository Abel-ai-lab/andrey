"""Link-preview cards (Open Graph, Twitter): 1200 x 630 JPEGs drawn at build time.

Each card is the stipple ground (``grain.py``), the title, a one-line subtitle, the Andrey
wordmark, and "by" with the Abel logotype under it. It shows no version: the page a card opens
does. ``card_svg`` builds the image as SVG; ``card_jpeg`` renders it and compresses it under the
size WhatsApp still previews.
"""

from __future__ import annotations

import base64
import html
import importlib.util
import io
import itertools
import re
from pathlib import Path

from PIL import Image, ImageFont

import andrey.viz

HERE = Path(__file__).resolve().parent
BRAND = HERE / "brand"
FONTS = sorted((BRAND / "fonts").glob("*.ttf"))
_spec = importlib.util.spec_from_file_location("andrey_grain", HERE / "grain.py")
grain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grain)

W, H = 1200, 630
MAX_BYTES = 300 * 1024  # WhatsApp is widely reported to skip preview images above about 300 KB
QUALITY = 75  # about 20 KB under MAX_BYTES at 75, 11 KB at 76, 1 KB at 77. No banding at 75.
# The title's sizes: two lines at 76 px, else three at 64 px, each line no wider than the band.
TITLE_FITS = ((76, 2), (64, 3))
TITLE_WIDTH = 1040
SUBTITLE_SIZE = 34
SUBTITLE = "{lead} than other popular packages"


def subtitle(lead: str) -> str:
    """The speed claim's lead (``site/build.py``'s ``speed_claim``), against other packages."""
    return SUBTITLE.format(lead=lead)


def _width(text: str, size: int, weight: str) -> float:
    font = next(f for f in FONTS if f.stem.endswith(weight))
    return ImageFont.truetype(str(font), size).getlength(text)


def _view_box(path: Path) -> tuple[float, float, float, float]:
    box = re.search(r'viewBox="([-\d.]+) ([-\d.]+) ([\d.]+) ([\d.]+)"', path.read_text())
    x, y, width, height = (float(v) for v in box.groups())
    return x, y, width, height


def _scaled_width(path: Path, height: float) -> float:
    """How wide the SVG at ``path`` is when drawn ``height`` tall."""
    _, _, width, box_height = _view_box(path)
    return height * width / box_height


def _placed(path: Path, x: float, y: float, height: float, opacity: float = 1.0) -> str:
    """The SVG at ``path`` in white, ``height`` tall with its top left at (``x``, ``y``)."""
    body = re.sub(r"^.*?<svg[^>]*>|</svg>\s*$", "", path.read_text(), flags=re.S)
    body = re.sub(r'fill="#[0-9a-fA-F]{6}"', 'fill="#ffffff"', body)
    body = body.replace('fill="white"', 'fill="#ffffff"')
    bx, by, bw, bh = _view_box(path)
    return (
        f'<svg x="{x:.1f}" y="{y:.1f}" width="{_scaled_width(path, height):.1f}" '
        f'height="{height:.1f}" viewBox="{bx} {by} {bw} {bh}" opacity="{opacity}">{body}</svg>'
    )


def _lines(title: str, size: int, width: float, most: int) -> list[str] | None:
    """``title`` on the fewest lines, up to ``most``, no wider than ``width`` at ``size``; among
    the breaks that fit, the one whose longest line is shortest. None when none fits."""
    words = title.split()
    for count in range(1, min(most, len(words)) + 1):
        best = None
        for breaks in itertools.combinations(range(1, len(words)), count - 1):
            ends = (0, *breaks, len(words))
            lines = [" ".join(words[i:j]) for i, j in zip(ends, ends[1:], strict=False)]
            widest = max(_width(line, size, "SemiBold") for line in lines)
            if widest <= width and (best is None or widest < best[0]):
                best = (widest, lines)
        if best:
            return best[1]
    return None


def _title_lines(title: str) -> tuple[int, list[str]]:
    """``title`` wrapped by measured width: the largest size in ``TITLE_FITS`` it fits at."""
    for size, most in TITLE_FITS:
        if lines := _lines(title, size, TITLE_WIDTH, most):
            return size, lines
    size, most = TITLE_FITS[-1]
    raise ValueError(f"the card title {title!r} needs more than {most} lines at {size} px")


def _credit() -> str:
    """The Andrey wordmark, and under it, small, "by" and the Abel logotype."""
    wordmark = BRAND / "wordmark-light.svg"
    logo = BRAND / "abel-logo-white.svg"
    wy = H - 132
    by = _width("by", 17, "Regular")
    x0 = (W - (by + 8 + _scaled_width(logo, 17))) / 2
    return (
        _placed(wordmark, (W - _scaled_width(wordmark, 44)) / 2, wy, 44)
        + f'<text x="{x0:.1f}" y="{wy + 76:.1f}" font-family="Inter" font-size="17" '
        'fill="#ffffff" opacity="0.72">by</text>' + _placed(logo, x0 + by + 8, wy + 62, 17, 0.72)
    )


def card_svg(title: str, subtitle: str) -> str:
    """The card as SVG: ``title`` on up to three lines, and ``subtitle`` under it.

    Raises ``ValueError`` when the title does not fit, rather than dropping any of it.
    """
    size, lines = _title_lines(title)
    lead = size * 1.18
    top = H / 2 - 40 - (len(lines) - 1) * lead / 2
    below = top + (len(lines) - 1) * lead + size * 0.95
    # Darker bands behind the title block and the wordmark keep the white type readable.
    bands = (
        grain.Band(600, (top + below) / 2 - 11, 520, 95 + 45 * (len(lines) - 1)),
        grain.Band(600, 545, 200, 55),
    )
    buffer = io.BytesIO()
    grain.stipple(W, H, bands).save(buffer, "PNG")
    ground = (
        f'<image href="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}" '
        f'width="{W}" height="{H}"/>'
    )
    text = "".join(
        f'<text x="{W // 2}" y="{top + i * lead:.0f}" text-anchor="middle" '
        'dominant-baseline="middle" font-family="Inter" font-weight="600" '
        f'font-size="{size}" fill="#ffffff">{html.escape(line)}</text>'
        for i, line in enumerate(lines)
    )
    text += (
        f'<text x="{W // 2}" y="{below:.0f}" text-anchor="middle" dominant-baseline="middle" '
        f'font-family="Inter" font-size="{SUBTITLE_SIZE}" fill="{andrey.viz.LIGHT.ramp[0]}">'
        f"{html.escape(subtitle)}</text>"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">'
        f"{ground}{text}{_credit()}</svg>"
    )


def _render(svg: str) -> Image.Image:
    """``svg`` rendered with the brand fonts, transparency kept."""
    import resvg_py  # the docs group's renderer; the rest of the module needs only Pillow

    png = resvg_py.svg_to_bytes(
        svg_string=svg,
        font_files=[str(f) for f in FONTS],
        skip_system_fonts=True,
        font_family="Inter",
    )
    return Image.open(io.BytesIO(bytes(png))).convert("RGBA")


def _jpeg(svg: str, quality: int) -> bytes:
    """``svg`` rendered, as a JPEG with full-resolution color so the amber specks stay sharp."""
    out = io.BytesIO()
    _render(svg).convert("RGB").save(out, "JPEG", quality=quality, subsampling=0, optimize=True)
    return out.getvalue()


def card_jpeg(title: str, subtitle: str, quality: int = QUALITY) -> bytes:
    """The card as a JPEG."""
    return _jpeg(card_svg(title, subtitle), quality)


BANNER_W, BANNER_H = 1600, 480  # shown about 830 px wide on GitHub, so sharp at twice that
# The README chart's 16-unit corner on its 960-unit width, at the banner's width: both are shown
# at the README's width, so their corners match.
BANNER_RADIUS = 27
BANNER_QUALITY = 65  # its base64 inside the SVG is about 290 KB, under MAX_BYTES
TAGLINE = "A very fast causal discovery package"


def banner_svg() -> str:
    """The README header: the wordmark, the tagline, and "by" with the Abel logotype, centered on
    the cards' ground. It carries no version, so it never goes stale."""
    buffer = io.BytesIO()
    grain.stipple(BANNER_W, BANNER_H, (grain.Band(600, 300, 470, 190, 0.75),)).save(buffer, "PNG")
    ground = (
        f'<image href="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}" '
        f'width="{BANNER_W}" height="{BANNER_H}"/>'
    )
    wordmark = BRAND / "wordmark-light.svg"
    logo = BRAND / "abel-logo-white.svg"
    by = _width("by", 22, "Regular")
    x0 = (BANNER_W - (by + 10 + _scaled_width(logo, 22))) / 2
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{BANNER_W}" height="{BANNER_H}">{ground}'
        + _placed(wordmark, (BANNER_W - _scaled_width(wordmark, 104)) / 2, 118, 104)
        + f'<text x="{BANNER_W / 2}" y="292" text-anchor="middle" font-family="Inter" '
        f'font-weight="600" font-size="46" fill="#ffffff">{TAGLINE}</text>'
        + f'<text x="{x0:.1f}" y="370" font-family="Inter" font-size="22" fill="#ffffff" '
        'opacity="0.72">by</text>' + _placed(logo, x0 + by + 10, 352, 22, 0.72) + "</svg>"
    )


def readme_banner(quality: int = BANNER_QUALITY) -> str:
    """The README header as the repository keeps it: the banner rendered to a JPEG, inside an SVG
    that clips it to the README chart's rounded corners. The text is in the pixels, so no font has
    to load, and GitHub's and PyPI's image proxies accept an SVG where they refuse WebP."""
    out = io.BytesIO()
    _render(banner_svg()).convert("RGB").save(
        out, "JPEG", quality=quality, subsampling=0, optimize=True
    )
    data = base64.b64encode(out.getvalue()).decode()
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{BANNER_W}" height="{BANNER_H}" '
        f'viewBox="0 0 {BANNER_W} {BANNER_H}"><clipPath id="corners">'
        f'<rect width="{BANNER_W}" height="{BANNER_H}" rx="{BANNER_RADIUS}"/></clipPath>'
        f'<image href="data:image/jpeg;base64,{data}" width="{BANNER_W}" height="{BANNER_H}" '
        'clip-path="url(#corners)"/></svg>\n'
    )


# ---- blog post covers ---------------------------------------------------------------------------
# A cover sits under a post's title: 1600 x 640, a JPEG under MAX_BYTES, drawn once per build.
COVER_W, COVER_H = 1600, 640
# 60 leaves about 40 KB under MAX_BYTES on both covers; 65 leaves 7 KB on the graph. No banding.
COVER_QUALITY = 60
# The pair cover: Kolmogorov bottom left, turned to level his eyes, and Markov bottom right, with
# each head's eye midpoint and crown in its file's pixels.
KOLMOGOROV = BRAND / "andrey-kolmogorov.png", (416.8, 290.1), 18.7
MARKOV = BRAND / "andrey-markov.png", (264.5, 295.5), 29.9
KOLMOGOROV_TURN = 9.0
# The graph cover: a n d r e y as a W of white nodes, each edge directed, with a dashed amber
# two-headed arc from a to y.
LETTERS = {"a": (450, 183), "n": (600, 457), "d": (730, 269), "r": (870, 269), "e": (1000, 457)}
LETTERS["y"] = (1150, 183)
CHAIN = (("a", "n"), ("n", "d"), ("d", "r"), ("r", "e"), ("e", "y"))


def _png_href(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def _pair(width: int, height: int) -> Image.Image:
    """Kolmogorov and Markov as navy line screens, heads matched in size and eye height."""
    (k_path, k_eye, k_crown), (m_path, m_eye, m_crown) = KOLMOGOROV, MARKOV
    kolmogorov = grain.Portrait(
        Image.open(k_path), k_eye, k_crown, turn=KOLMOGOROV_TURN, floor=0.25, gamma=1.0, fade=0.12
    )
    markov = grain.Portrait(Image.open(m_path), m_eye, m_crown)
    return grain.pair(width, height, kolmogorov, markov)


def pair_cover_svg() -> str:
    """The pair cover, with no text: the post's title sits above it on the page."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{COVER_W}" height="{COVER_H}">'
        f'<image href="{_png_href(_pair(COVER_W, COVER_H))}" width="{COVER_W}" height="{COVER_H}"/>'
        "</svg>"
    )


def _marker(name: str, colour: str) -> str:
    return (
        f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
        f'fill="{colour}"/></marker>'
    )


def graph_cover_svg() -> str:
    """The graph cover: the letters of Andrey as a small directed graph on the cards' ground."""
    ground = grain.stipple(COVER_W, COVER_H, (grain.Band(600, 320, 420, 240, 0.35),))
    ink, navy, amber = "#ffffff", grain.NAVY, andrey.viz.LIGHT.categorical[1]
    radius = 40
    parts = [
        f"<defs>{_marker('ink', ink)}{_marker('amber', amber)}"
        '<filter id="shade" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="6"/></filter></defs>'
    ]
    for u, v in CHAIN:
        (x0, y0), (x1, y1) = LETTERS[u], LETTERS[v]
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        parts.append(
            f'<line x1="{x0 + ux * (radius + 6):.1f}" y1="{y0 + uy * (radius + 6):.1f}" '
            f'x2="{x1 - ux * (radius + 10):.1f}" y2="{y1 - uy * (radius + 10):.1f}" stroke="{ink}" '
            'stroke-width="4" stroke-linecap="round" marker-end="url(#ink)"/>'
        )
    (ax, ay), (yx, yy) = LETTERS["a"], LETTERS["y"]
    parts.append(
        f'<path d="M{ax + 30},{ay - 26} Q{(ax + yx) / 2},{ay - 170} {yx - 30},{yy - 26}" '
        f'fill="none" stroke="{amber}" stroke-width="3" stroke-dasharray="10 9" '
        'marker-start="url(#amber)" marker-end="url(#amber)" opacity="0.95"/>'
    )
    for letter, (x, y) in LETTERS.items():
        parts.append(
            f'<circle cx="{x}" cy="{y + 5}" r="{radius}" fill="{navy}" opacity="0.3" '
            'filter="url(#shade)"/>'
            f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{ink}"/>'
            f'<text x="{x}" y="{y + 2}" text-anchor="middle" dominant-baseline="middle" '
            f'font-family="Inter" font-weight="600" font-size="42" fill="{navy}">{letter}</text>'
        )
    # Drawn at 1.3x around the graph's middle, so it spans about two thirds of the cover's width.
    cx, cy = 800, 291
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{COVER_W}" height="{COVER_H}">'
        f'<image href="{_png_href(ground)}" width="{COVER_W}" height="{COVER_H}"/>'
        f'<g transform="translate({cx} {cy}) scale(1.3) translate({-cx} {-cy})">'
        f"{''.join(parts)}</g></svg>"
    )


COVERS = {
    "pair": "portraits of Andrey Kolmogorov and Andrey Markov",
    "graph": "the letters of Andrey as a graph",
}


def cover_jpeg(kind: str, quality: int = COVER_QUALITY) -> bytes:
    """A post's cover as a JPEG: ``kind`` is ``pair`` or ``graph``, the default."""
    if kind not in COVERS:
        raise ValueError(f"unknown cover {kind!r}; expected one of {sorted(COVERS)}")
    return _jpeg(pair_cover_svg() if kind == "pair" else graph_cover_svg(), quality)


def cover_alt(kind: str) -> str:
    """The cover's alt text."""
    if kind == "pair":
        return "Portraits of Andrey Kolmogorov and Andrey Markov drawn in navy lines"
    return "The letters a, n, d, r, e, y as a small directed graph, with a dashed arc from a to y"
