"""Link-preview cards (Open Graph, Twitter): 1200 x 630 JPEGs drawn at build time.

Each card is the stipple ground (``grain.py``), the title, the Andrey wordmark, and "by" with the
Abel logotype under it. It shows no version: the page a card opens
does. ``card_svg`` builds the image as SVG; ``card_jpeg`` renders it and compresses it under the
size WhatsApp still previews.
"""

from __future__ import annotations

import base64
import functools
import html
import importlib.util
import io
import itertools
import re
from pathlib import Path

import numpy as np
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
# The title's sizes: two lines at 92 px, else three at 72 px, each line no wider than the band.
# LinkedIn shows a card at 480 px wide, where 92 px becomes about 37.
TITLE_FITS = ((92, 2), (72, 3))
TITLE_WIDTH = 1040
# The accent word (``ACCENT`` on the homepage's card and the README banner) is drawn at rest, as the
# homepage's motion ends: amber, slanted like Inter's italic, and sharp, with the trace of its
# letters on its right: copies of the word behind it, each further right and fainter, blurred
# sideways and fading out by TRAIL past its end, inside the space before the next word. ROOM is
# the space added after the word, which keeps the gaps on its two sides even (the slant leans it
# toward the previous word); TRAIL and BLUR are in font sizes too; TRACE is the copies' opacities,
# nearest first.
SLANT = 11
ROOM = 0.04
TRAIL = 0.2
BLUR = 0.06
TRACE = (0.5, 0.4, 0.3, 0.22, 0.14, 0.08)


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
    wy = H - 146
    by = _width("by", 20, "Regular")
    x0 = (W - (by + 9 + _scaled_width(logo, 20))) / 2
    return (
        _placed(wordmark, (W - _scaled_width(wordmark, 52)) / 2, wy, 52)
        + f'<text x="{x0:.1f}" y="{wy + 88:.1f}" font-family="Inter" font-size="20" '
        'fill="#ffffff" opacity="0.72">by</text>' + _placed(logo, x0 + by + 9, wy + 71, 20, 0.72)
    )


def _text(word: str, x: float, y: float, size: int, fill: str, extra: str = "") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" dominant-baseline="middle" font-family="Inter" '
        f'font-weight="600" font-size="{size}" fill="{fill}"{extra}>{html.escape(word)}</text>'
    )


def _accent(word: str, x: float, y: float, size: int, key: str) -> tuple[str, str]:
    """``word`` in amber at (``x``, ``y``), slanted, with its trace after it: its defs and its
    drawing."""
    width, trail = _width(word, size, "SemiBold"), size * TRAIL
    defs = (
        _text(word, x, y, size, grain.AMBER, f' id="{key}"')
        # The trace shows at half strength across the word, at full near its end, and fades out
        # by the end of its reach.
        + f'<linearGradient id="{key}-fade" gradientUnits="userSpaceOnUse" x1="{x:.1f}" '
        f'x2="{x + width + trail:.1f}" y1="0" y2="0"><stop offset="0" stop-color="#ffffff" '
        'stop-opacity="0.5"/><stop offset="0.65" stop-color="#ffffff"/>'
        '<stop offset="1" stop-color="#ffffff" stop-opacity="0"/></linearGradient>'
        f'<mask id="{key}-trace"><rect x="{x:.1f}" y="{y - size:.1f}" width="{width + trail:.1f}" '
        f'height="{size * 2}" fill="url(#{key}-fade)"/></mask>'
        f'<filter id="{key}-blur" x="-50%" y="-50%" width="200%" height="200%">'
        f'<feGaussianBlur stdDeviation="{size * BLUR:.1f} 0"/></filter>'
    )
    step = trail / len(TRACE)
    copies = "".join(
        f'<use href="#{key}" x="{(k + 1) * step:.1f}" opacity="{alpha}"/>'
        for k, alpha in enumerate(TRACE)
    )
    slant = f"translate({x:.1f} {y:.1f}) skewX(-{SLANT}) translate({-x:.1f} {-y:.1f})"
    return defs, (
        f'<g transform="{slant}"><g mask="url(#{key}-trace)"><g filter="url(#{key}-blur)">'
        f'{copies}</g></g><use href="#{key}"/></g>'
    )


def _title(lines: list[str], size: int, middles: list[float], center: float, accent: str) -> str:
    """``lines`` in white, each centered on ``center`` at its middle in ``middles``; each word equal
    to ``accent`` drawn as the accent, with its trace's space added after it."""
    space = _width(" ", size, "SemiBold")
    defs, drawing = [], []
    for row, (line, y) in enumerate(zip(lines, middles, strict=True)):
        words = line.split(" ")
        widths = [_width(word, size, "SemiBold") for word in words]
        rooms = [size * ROOM if accent and word == accent else 0 for word in words]
        x = center - (sum(widths) + space * (len(words) - 1) + sum(rooms)) / 2
        for i, (word, width, room) in enumerate(zip(words, widths, rooms, strict=True)):
            if accent and word == accent:
                more, drawn = _accent(word, x, y, size, f"accent-{row}-{i}")
                defs.append(more)
                drawing.append(drawn)
            else:
                drawing.append(_text(word, x, y, size, "#ffffff"))
            x += width + space + room
    return f"<defs>{''.join(defs)}</defs>{''.join(drawing)}"


def card_svg(title: str, accent: str = "") -> str:
    """The card as SVG: ``title`` on up to three lines, each word equal to ``accent`` drawn as the
    accent.

    Raises ``ValueError`` when the title does not fit, rather than dropping any of it.
    """
    size, lines = _title_lines(title)
    lead = size * 1.18
    middle = H / 2 - 30
    top = middle - (len(lines) - 1) * lead / 2
    # Darker bands behind the title and the wordmark keep the white type readable.
    bands = (
        grain.Band(600, middle, 520, 70 + 45 * (len(lines) - 1)),
        grain.Band(600, 540, 220, 62),
    )
    buffer = io.BytesIO()
    grain.stipple(W, H, bands).save(buffer, "PNG")
    ground = (
        f'<image href="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}" '
        f'width="{W}" height="{H}"/>'
    )
    text = _title(lines, size, [top + i * lead for i in range(len(lines))], W / 2, accent)
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


def card_jpeg(title: str, accent: str = "", quality: int = QUALITY) -> bytes:
    """The card as a JPEG."""
    return _jpeg(card_svg(title, accent), quality)


BANNER_W, BANNER_H = 1600, 480  # shown about 830 px wide on GitHub, so sharp at twice that
# The README chart's 16-unit corner on its 960-unit width, at the banner's width: both are shown
# at the README's width, so their corners match.
BANNER_RADIUS = 27
BANNER_QUALITY = 65  # its base64 inside the SVG is about 290 KB, under MAX_BYTES
TAGLINE = "A very fast causal discovery package"
ACCENT = "fast"  # the tagline's word drawn as the accent, on the homepage's card and the banner


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
        + _title([TAGLINE], 46, [276], BANNER_W / 2, ACCENT)
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
# A post's cover is a banner, 1600 x 640, under the post's title: on the left, over a navy fade,
# the wordmark, the title, and a short line (its release, or its kind); on the right, the post's
# art. Each art is stored as drawn (``ART``), its subject on the right, and recolored at build time
# to the brand's tones (``brand_tones``), so cobalt and amber match across posts. A post without
# art shows the stipple ground. The JPEG takes the first quality in ``COVER_QUALITIES`` under
# MAX_BYTES.
COVER_W, COVER_H = 1600, 640
COVER_QUALITIES = (75, 70, 65, 60, 55)
ART = {kind: BRAND / f"art-{kind}.jpg" for kind in ("pair", "lanes", "iceberg", "contour")}
COVERS = {
    "pair": "portraits of Andrey Markov and Andrey Kolmogorov, side by side",
    "lanes": "lanes rising into the sky, one amber streak far ahead",
    "iceberg": "an iceberg whose two peaks join into one faint mass under the waterline",
    "contour": "two summits drawn in contour lines, one dotted trail ending on each",
    "ground": "the stipple ground",
}
BANNER_LEFT = 104  # the text column's left edge
BANNER_FADE = 0.55  # the navy fade behind the text ends at this share of the width
BANNER_FADE_PEAK = 0.6  # its opacity at the left edge
BANNER_TITLE = (64, 560, 2)  # the title's size, its widest line, and its most lines
# The blog's front page shows each earlier post's art as a square tile of TILE px: the art's
# square centered at TILE_CENTER of its width, where every art's subject sits. 240 px stays sharp
# at three times the rows' 76 px.
TILE = 240
TILE_QUALITY = 75
TILE_CENTER = 0.72
# Lightness and warmth of the brand's tones. LUMA weighs red, green, and blue into lightness.
LUMA = np.array([0.299, 0.587, 0.114])


def _rgb(color: str) -> np.ndarray:
    return np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], float)


@functools.cache
def _ground_lightness() -> tuple[float, float, float]:
    """The stipple ground's 10th, 50th, and 90th percentiles of lightness."""
    ground = np.asarray(grain.stipple(400, 160).convert("RGB"), float) @ LUMA
    return tuple(np.percentile(ground, [10, 50, 90]))


def brand_tones(image: Image.Image, match_ground: bool = True) -> Image.Image:
    """``image`` in the brand's tones, its drawing unchanged.

    With ``match_ground``, lightness moves to the stipple ground's: the image's 10th, 50th, and
    90th percentiles map to the ground's, and its brightest highlights stay bright. Without it (a
    release card's art, dark where its text sits), lightness stays.

    A cool pixel takes the blue of its lightness along navy, the palette's ramp, and white. A warm
    pixel takes amber at the lightness it was drawn with. Warmth is how far red exceeds blue,
    against the image's own blue ground, so a soft haze of amber stays dim; pale, colorless pixels
    (white dots and trails) stay cool."""
    pixels = np.asarray(image.convert("RGB"), float)
    drawn = pixels @ LUMA
    knots = np.percentile(drawn, [10, 50, 90, 99.5])
    top = max(knots[3], knots[2] + 1)
    target = [0, *_ground_lightness(), max(top, 200), 255]
    light = np.interp(drawn, [0, *knots[:3], top, 255], target) if match_ground else drawn
    ramp = andrey.viz.LIGHT.ramp  # palest first
    colors = np.array(
        [np.zeros(3), _rgb(grain.NAVY), *(_rgb(c) for c in ramp[::-1][1:]), np.full(3, 255.0)]
    )
    cool = np.stack([np.interp(light, colors @ LUMA, colors[:, c]) for c in range(3)], -1)
    amber = _rgb(grain.AMBER)
    amber_light = amber @ LUMA
    above = np.clip((drawn - amber_light) / (255 - amber_light), 0, 1)[..., None]
    warm = np.where(
        (drawn <= amber_light)[..., None],
        amber * (drawn / amber_light)[..., None],
        amber * (1 - above) + 255 * above,
    )
    red_minus_blue = pixels[..., 0] - pixels[..., 2]
    ground = np.percentile(red_minus_blue, 50)
    warmest = max(np.percentile(red_minus_blue, 99.7), ground + 60)
    warmth = np.clip((red_minus_blue - ground) / (warmest - ground), 0, 1) ** 2.2
    colorfulness = (pixels.max(-1) - pixels.min(-1)) / np.maximum(pixels.max(-1), 1)
    warmth *= 1 - np.clip((drawn - 185) / 45, 0, 1) * (colorfulness < 0.25)
    out = cool * (1 - warmth[..., None]) + warm * warmth[..., None]
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def _art(kind: str) -> Image.Image:
    """The post's art in the brand's tones at the cover's size, or the stipple ground."""
    if kind not in COVERS:
        raise ValueError(f"unknown cover {kind!r}; expected one of {sorted(COVERS)}")
    if kind == "ground":
        return grain.stipple(COVER_W, COVER_H).convert("RGB")
    with Image.open(ART[kind]) as art:
        return brand_tones(art.convert("RGB").resize((COVER_W, COVER_H), Image.LANCZOS))


def _jpeg_href(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=92)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def cover_svg(kind: str, title: str, line: str) -> str:
    """The banner as SVG: ``kind``'s art, and over its left side the wordmark, ``title``, and
    ``line`` after a short amber rule."""
    size, width, most = BANNER_TITLE
    lines = _lines(title, size, width, most) or [title]
    top = 268 if len(lines) == most else 300
    navy = grain.NAVY
    stops = ((0, BANNER_FADE_PEAK), (BANNER_FADE * 0.55, BANNER_FADE_PEAK * 0.75), (BANNER_FADE, 0))
    fade = "".join(
        f'<stop offset="{offset:.2f}" stop-color="{navy}" stop-opacity="{opacity:.2f}"/>'
        for offset, opacity in stops
    )
    title_text = "".join(
        f'<text x="{BANNER_LEFT}" y="{top + i * 76}" font-family="Inter" font-weight="600" '
        f'font-size="{size}" letter-spacing="-1.5" fill="#ffffff">{html.escape(text)}</text>'
        for i, text in enumerate(lines)
    )
    rule_y = top + (len(lines) - 1) * 76 + 66
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{COVER_W}" height="{COVER_H}">'
        f'<defs><linearGradient id="fade" x1="0" x2="1" y1="0" y2="0">{fade}</linearGradient>'
        f'</defs><image href="{_jpeg_href(_art(kind))}" width="{COVER_W}" height="{COVER_H}"/>'
        f'<rect width="{COVER_W}" height="{COVER_H}" fill="url(#fade)"/>'
        + _placed(BRAND / "wordmark-dark.svg", BANNER_LEFT, 104, 44)
        + title_text
        + (
            f'<rect x="{BANNER_LEFT}" y="{rule_y}" width="34" height="4" fill="{grain.AMBER}"/>'
            f'<text x="{BANNER_LEFT + 50}" y="{rule_y + 16}" font-family="Inter" '
            f'font-weight="600" font-size="30" fill="#ffffff" opacity="0.9">'
            f"{html.escape(line)}</text>"
            if line
            else ""
        )
        + "</svg>"
    )


def cover_jpeg(kind: str, title: str, line: str = "") -> bytes:
    """The banner (``cover_svg``) as a JPEG under MAX_BYTES."""
    image = _render(cover_svg(kind, title, line)).convert("RGB")
    for quality in COVER_QUALITIES:
        out = io.BytesIO()
        image.save(out, "JPEG", quality=quality, subsampling=0, optimize=True)
        if out.tell() <= MAX_BYTES:
            break
    return out.getvalue()


def tile_jpeg(kind: str) -> bytes:
    """The post's art, without the banner's text, as a square tile (``TILE``)."""
    art = _art(kind)
    left = min(max(round(TILE_CENTER * COVER_W - COVER_H / 2), 0), COVER_W - COVER_H)
    tile = art.crop((left, 0, left + COVER_H, COVER_H)).resize((TILE, TILE), Image.LANCZOS)
    out = io.BytesIO()
    tile.save(out, "JPEG", quality=TILE_QUALITY, subsampling=0, optimize=True)
    return out.getvalue()


def cover_alt(kind: str) -> str:
    """The cover's alt text: its art, in a sentence."""
    if kind not in COVERS:
        raise ValueError(f"unknown cover {kind!r}; expected one of {sorted(COVERS)}")
    return f"Banner: {COVERS[kind]}, drawn in navy lines"
