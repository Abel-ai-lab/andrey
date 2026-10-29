"""Link previews, the README banner, release and post art, and the brand kit, drawn at build time.

The homepage and non-post pages share the repository's social preview (``social_jpeg``), which
GitHub also shows. Each post's link preview is its banner (``preview_jpeg``).
"""

from __future__ import annotations

import base64
import functools
import html
import importlib.util
import io
import itertools
import re
import runpy
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
    """The SVG at ``path`` in white, ``height`` tall, with its top left at (``x``, ``y``).

    The wordmark's right stroke stays amber.
    """
    body = re.sub(r"^.*?<svg[^>]*>|</svg>\s*$", "", path.read_text(), flags=re.S)
    keep = grain.AMBER.lower()
    body = re.sub(
        r'fill="(#[0-9a-fA-F]{6})"',
        lambda m: m[0] if m[1].lower() == keep else 'fill="#ffffff"',
        body,
    )
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


def _text(word: str, x: float, y: float, size: int, fill: str) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" dominant-baseline="middle" font-family="Inter" '
        f'font-weight="600" font-size="{size}" fill="{fill}">{html.escape(word)}</text>'
    )


def _title(
    lines: list[str],
    size: int,
    middles: list[float],
    x0: float,
    accents: tuple[str, ...],
    *,
    left: bool = False,
) -> str:
    """``lines`` in white, each centered on ``x0`` (or starting at it, when ``left``) at its middle
    in ``middles``; each word in ``accents`` in amber."""
    space = _width(" ", size, "SemiBold")
    drawing = []
    for line, y in zip(lines, middles, strict=True):
        words = line.split(" ")
        widths = [_width(word, size, "SemiBold") for word in words]
        x = x0 if left else x0 - (sum(widths) + space * (len(words) - 1)) / 2
        for word, width in zip(words, widths, strict=True):
            drawing.append(_text(word, x, y, size, grain.AMBER if word in accents else "#ffffff"))
            x += width + space
    return "".join(drawing)


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


BANNER_W, BANNER_H = 1600, 480  # shown about 830 px wide on GitHub, so sharp at twice that
# The README chart's 16-unit corner on its 960-unit width, at the banner's width: both are shown
# at the README's width, so their corners match.
BANNER_RADIUS = 27
BANNER_QUALITY = 65  # its base64 inside the SVG is about 290 KB, under MAX_BYTES
TAGLINE = "Accelerated, accurate, agentic causal discovery"
# The tagline's three A's appear in amber on the banner and social preview.
ACCENT = ("Accelerated,", "accurate,", "agentic")


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
# A post's cover is its art, 1600 x 640, under the post's title and on the blog's front page, where
# a small copy (``THUMB_W`` wide) marks each earlier post. Its link preview is the banner: on the
# left, over a navy fade, the wordmark, the title, and a short line (its release, or its kind); on
# the right, the art, at the card size (``W`` x ``H``) link previews expect. Each art (``ART``), its
# subject on the right, is stored already recolored to the brand's tones (``brand_tones``), so the
# file is what the page shows and cobalt and amber match across posts; a test fails on art stored
# otherwise. A post without art shows the stipple ground. An art may also come centered
# (``CENTERED``), its subject in the middle, for the cover and the thumbnail, which carry no text.
# A JPEG takes the first quality in ``COVER_QUALITIES`` under MAX_BYTES.
COVER_W, COVER_H = 1600, 640
COVER_QUALITIES = (75, 70, 65, 60, 55)
ART = {
    kind: BRAND / f"art-{kind}.jpg"
    for kind in (
        "pair",
        "lanes",
        "iceberg",
        "contour",
        "hourglass",
        "agentic",
        "start",
        "docs",
        "methods",
        "tide",
        "abacus",
        "punch-cards",
        "harmonograph",
        "rocket",
        "ceiling",
        "shelf",
    )
}
CENTERED = {kind: BRAND / f"art-{kind}-center.jpg" for kind in ("tide",)}
# Each art's description, and the alt text drawn from it (covers.py, which needs no Pillow).
_covers = runpy.run_path(str(HERE / "covers.py"))
COVERS, cover_alt = _covers["COVERS"], _covers["cover_alt"]
BANNER_LEFT = 104  # the text column's left edge, at the cover's height
BANNER_FADE = 0.55  # the navy fade behind the text ends at this share of the width
BANNER_FADE_PEAK = 0.6  # its opacity at the left edge
# The title's size, its widest line, and its most lines: large enough to read where a link preview
# shows the banner at a third of its size.
BANNER_TITLE = (88, 720, 2)
BANNER_LINE = 40  # the release or kind under the title
# The arts whose banner sets the title low, the line's baseline at BANNER_BOTTOM, to leave the art
# its height; the others center it under the wordmark.
BANNER_LOW = {"punch-cards"}
BANNER_BOTTOM = 552
# An earlier post's thumbnail on the blog's front page: the art, THUMB_W wide, sharp at three
# times the rows' 10 rem.
THUMB_W = 480
THUMB_QUALITY = 75
# Lightness and warmth of the brand's tones. LUMA weighs red, green, and blue into lightness.
LUMA = np.array([0.299, 0.587, 0.114])


def _rgb(color: str) -> np.ndarray:
    return np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], float)


@functools.cache
def _ground_lightness() -> tuple[float, float, float]:
    """The stipple ground's 10th, 50th, and 90th percentiles of lightness."""
    ground = np.asarray(grain.stipple(400, 160).convert("RGB"), float) @ LUMA
    return tuple(np.percentile(ground, [10, 50, 90]))


def _cool(light: np.ndarray) -> np.ndarray:
    """The blue of each lightness along black, navy, the palette's ramp, and white."""
    ramp = andrey.viz.LIGHT.ramp  # palest first
    colors = np.array(
        [np.zeros(3), _rgb(grain.NAVY), *(_rgb(c) for c in ramp[::-1][1:]), np.full(3, 255.0)]
    )
    return np.stack([np.interp(light, colors @ LUMA, colors[:, c]) for c in range(3)], -1)


def ground_tokens() -> dict[str, str]:
    """``grain.ground_tokens`` and ``--andrey-ground-art``: the color ``brand_tones`` gives an
    art's median pixel, its ground, so a page's ground meets the art with no seam."""
    art = _cool(np.array(_ground_lightness()[1])).round().astype(int)
    return {**grain.ground_tokens(), "--andrey-ground-art": "#{:02x}{:02x}{:02x}".format(*art)}


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
    cool = _cool(light)
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


# A release card's cover art, 8.5 x 11 at three times the cover's 288 px width: the art in the
# brand's tones, with a soft navy shade where the card's text sits, the masthead and title across
# the top and the version at the bottom left. A release without art, and every earlier release's
# tile, shows the stipple ground under the same shade (``release_ground_jpeg``).
RELEASE_W, RELEASE_H = 864, 1119
RELEASE_SHADE = (0.55, 0.5)  # the shade's depth at the top edge and at the bottom left corner


def release_art(image: Image.Image) -> Image.Image:
    """``image`` as a release card's cover: the brand's tones at the cover's size, shaded for its
    text."""
    art = np.asarray(
        brand_tones(image.convert("RGB").resize((RELEASE_W, RELEASE_H), Image.LANCZOS)), float
    )
    y = np.arange(RELEASE_H)[:, None] / RELEASE_H
    x = np.arange(RELEASE_W)[None, :] / RELEASE_W
    top = RELEASE_SHADE[0] * np.clip(1 - y / 0.36, 0, 1) ** 1.4
    corner = RELEASE_SHADE[1] * np.clip(1 - np.hypot(x / 0.62, (1 - y) / 0.34), 0, 1) ** 1.2
    shade = np.maximum(top, corner)[..., None]
    out = art * (1 - shade) + _rgb(grain.NAVY) * shade
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def release_ground_jpeg() -> bytes:
    """The stipple ground as a release cover (``release_art``), for a release without art."""
    out = io.BytesIO()
    release_art(grain.stipple(RELEASE_W, RELEASE_H).convert("RGB")).save(
        out, "JPEG", quality=60, subsampling=0, optimize=True
    )
    return out.getvalue()


def _art(kind: str, centered: bool = False) -> Image.Image:
    """The post's art as stored (already in the brand's tones), or the stipple ground; with
    ``centered``, its centered version when it has one (``CENTERED``)."""
    if kind not in COVERS:
        raise ValueError(f"unknown cover {kind!r}; expected one of {sorted(COVERS)}")
    if kind == "ground":
        return grain.stipple(COVER_W, COVER_H).convert("RGB")
    with Image.open(CENTERED[kind] if centered and kind in CENTERED else ART[kind]) as art:
        return art.convert("RGB").resize((COVER_W, COVER_H), Image.LANCZOS)


def _jpeg_href(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=92)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def preview_svg(kind: str, title: str, line: str, width: int = W, height: int = H) -> str:
    """The banner as SVG, ``width`` x ``height``: ``kind``'s art filling the height, its right edge
    aligned with the banner's. The wordmark, ``title``, and ``line`` sit over the left side, with
    an amber dot before ``line``, the title low for an art in ``BANNER_LOW``. Text sizes follow
    the height."""
    k = height / COVER_H
    size, line_width, most = BANNER_TITLE
    lines = _lines(title, size, line_width, most) or [title]
    step = round(size * 1.14)
    if kind in BANNER_LOW:
        top = BANNER_BOTTOM - 95 - (len(lines) - 1) * step
    else:
        top = 284 if len(lines) == most else 334
    title_text = "".join(
        f'<text x="{BANNER_LEFT}" y="{top + i * step}" font-family="Inter" font-weight="600" '
        f'font-size="{size}" letter-spacing="-2" fill="#ffffff">{html.escape(text)}</text>'
        for i, text in enumerate(lines)
    )
    rule_y = top + (len(lines) - 1) * step + 78
    text = (
        _placed(BRAND / "wordmark-dark.svg", BANNER_LEFT, 96, 56)
        + title_text
        + (
            f'<circle cx="{BANNER_LEFT + 9}" cy="{rule_y + 2.5}" r="9" fill="{grain.AMBER}"/>'
            f'<text x="{BANNER_LEFT + 34}" y="{rule_y + 17}" font-family="Inter" '
            f'font-weight="600" font-size="{BANNER_LINE}" fill="#ffffff" opacity="0.9">'
            f"{html.escape(line)}</text>"
            if line
            else ""
        )
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'{_art_ground(kind, width, height)}<g transform="scale({k:.4f})">{text}</g></svg>'
    )


def _art_ground(kind: str, width: int, height: int, *, left: bool = False) -> str:
    """``kind``'s art filling the height, its right edge at ``width`` (its left edge at 0, when
    ``left``), under a navy fade from the left that keeps white text readable."""
    stops = ((0, BANNER_FADE_PEAK), (BANNER_FADE * 0.55, BANNER_FADE_PEAK * 0.75), (BANNER_FADE, 0))
    fade = "".join(
        f'<stop offset="{offset:.2f}" stop-color="{grain.NAVY}" stop-opacity="{opacity:.2f}"/>'
        for offset, opacity in stops
    )
    art_w = round(height * COVER_W / COVER_H)
    return (
        f'<defs><linearGradient id="fade" x1="0" x2="1" y1="0" y2="0">{fade}</linearGradient>'
        f'</defs><image href="{_jpeg_href(_art(kind))}" x="{0 if left else width - art_w}" y="0" '
        f'width="{art_w}" height="{height}" preserveAspectRatio="none"/>'
        f'<rect width="{width}" height="{height}" fill="url(#fade)"/>'
    )


def _jpeg_under_cap(image: Image.Image) -> bytes:
    """``image`` as a JPEG at the first quality in ``COVER_QUALITIES`` under MAX_BYTES."""
    for quality in COVER_QUALITIES:
        out = io.BytesIO()
        image.save(out, "JPEG", quality=quality, subsampling=0, optimize=True)
        if out.tell() <= MAX_BYTES:
            break
    return out.getvalue()


def preview_jpeg(kind: str, title: str, line: str = "") -> bytes:
    """A post's link preview: the banner (``preview_svg``) at the card size."""
    return _jpeg_under_cap(_render(preview_svg(kind, title, line)).convert("RGB"))


def cover_jpeg(kind: str) -> bytes:
    """A post's cover: its art at the cover's size, centered when it has a centered version."""
    return _jpeg_under_cap(_art(kind, centered=True))


def thumb_jpeg(kind: str) -> bytes:
    """An earlier post's thumbnail on the blog's front page: its art, ``THUMB_W`` wide, centered
    when it has a centered version."""
    thumb = _art(kind, centered=True).resize((THUMB_W, THUMB_W * COVER_H // COVER_W), Image.LANCZOS)
    out = io.BytesIO()
    thumb.save(out, "JPEG", quality=THUMB_QUALITY, subsampling=0, optimize=True)
    return out.getvalue()


# ---- the brand kit -----------------------------------------------------------------------------
# Brand kit downloads (`docs/docs/brand.md`): SVG and PNG wordmarks and icons for light and dark
# grounds, a square avatar, a social preview, and a zip of all files. Each is drawn from the
# brand's own files (`site/wordmark.py`), keeping the kit consistent with the site.
KIT_WORDMARK_H = 256  # the wordmark PNGs' height
KIT_ICON = 512  # the icon PNGs' side
KIT_AVATAR = 1024  # the avatar's side
# The repository's social preview, at GitHub's recommended size, and the homepage's link preview.
SOCIAL_W, SOCIAL_H = 1280, 640
SOCIAL_TITLE = 84  # the tagline's size, on three lines left of the lanes' amber streak
SOCIAL_TEXT_W = 720
SOCIAL_LINK = 28  # the site's address at the bottom
SOCIAL_QUALITY = 85
KIT_NOTE = """Andrey brand kit

The wordmark and the icon, for a light and a dark ground, as SVG and PNG, a square avatar, and a
social preview for link cards. Use them as they are to refer or link to Andrey: no recoloring,
stretching, effects, or retyped lettering, and the A stays whole. The Apache License 2.0 covers
Andrey's code, not its name or marks. Guidelines: https://andrey.abel.ai/docs/brand.html
"""


def _icon_paths() -> tuple[str, str, str]:
    """The icon's viewBox, ink path, and amber stroke path."""
    icon = (BRAND / "favicon.svg").read_text()
    view_box = re.search(r'viewBox="([^"]+)"', icon)[1]
    ink, amber = re.findall(r'<path[^>]* d="([^"]+)"', icon)
    return view_box, ink, amber


def _kit_icon(ink: str, ground: str | None = None) -> str:
    """The A in ``ink`` with its amber stroke, at 62% on a square ``ground`` when given."""
    view_box, ink_d, amber_d = _icon_paths()
    x, y, side, _ = (float(v) for v in view_box.split())
    a = f'<path fill="{ink}" d="{ink_d}"/><path fill="{grain.AMBER}" d="{amber_d}"/>'
    if ground is None:
        return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}">{a}</svg>\n'
    pad = side * (1 / 0.62 - 1) / 2
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x - pad:g} {y - pad:g} '
        f'{side + 2 * pad:g} {side + 2 * pad:g}"><rect x="{x - pad:g}" y="{y - pad:g}" '
        f'width="{side + 2 * pad:g}" height="{side + 2 * pad:g}" fill="{ground}"/>{a}</svg>\n'
    )


def social_svg() -> str:
    """The repository's social preview: the wordmark, the tagline with ``ACCENT`` in amber, and
    the site's address at the bottom, over the lanes art set at its left edge, so the amber streak
    runs to the right of the tagline."""
    lines = _lines(TAGLINE, SOCIAL_TITLE, SOCIAL_TEXT_W, 3)
    step = round(SOCIAL_TITLE * 1.14)
    top = SOCIAL_H / 2 - (len(lines) - 1) * step / 2 + 10
    middles = [top + i * step for i in range(len(lines))]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SOCIAL_W}" height="{SOCIAL_H}">'
        + _art_ground("lanes", SOCIAL_W, SOCIAL_H, left=True)
        + _placed(BRAND / "wordmark-dark.svg", BANNER_LEFT, 80, 56)
        + _title(lines, SOCIAL_TITLE, middles, BANNER_LEFT, ACCENT, left=True)
        + f'<text x="{BANNER_LEFT}" y="{SOCIAL_H - 72}" font-family="Inter" font-weight="600" '
        f'font-size="{SOCIAL_LINK}" fill="#ffffff" opacity="0.72">andrey.abel.ai</text></svg>'
    )


def social_jpeg() -> bytes:
    """The social preview as a JPEG, under ``MAX_BYTES``."""
    return _jpeg(social_svg(), SOCIAL_QUALITY)


def _png(svg: str, **size: int) -> bytes:
    import resvg_py  # the docs group's renderer

    return bytes(resvg_py.svg_to_bytes(svg_string=svg, **size))


def brand_kit() -> dict[str, bytes]:
    """The kit's files by name, including ``andrey-brand-kit.zip`` with all files."""
    import zipfile

    light, dark = andrey.viz.LIGHT.ink, andrey.viz.DARK.ink
    files = {}
    for ground, theme in (("light", "light"), ("dark", "dark")):
        svg = (BRAND / f"wordmark-{theme}.svg").read_text()
        files[f"andrey-wordmark-on-{ground}.svg"] = svg.encode()
        files[f"andrey-wordmark-on-{ground}.png"] = _png(svg, height=KIT_WORDMARK_H)
        icon = _kit_icon(light if ground == "light" else dark)
        files[f"andrey-icon-on-{ground}.svg"] = icon.encode()
        files[f"andrey-icon-on-{ground}.png"] = _png(icon, width=KIT_ICON)
    files["andrey-avatar.png"] = _png(_kit_icon("#ffffff", grain.COBALT), width=KIT_AVATAR)
    files["andrey-social-preview.jpg"] = social_jpeg()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as kit:
        for name, data in [("README.txt", KIT_NOTE.encode()), *files.items()]:
            info = zipfile.ZipInfo(f"andrey-brand-kit/{name}", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            kit.writestr(info, data)  # a fixed date keeps zip bytes consistent across builds
    files["andrey-brand-kit.zip"] = out.getvalue()
    return files
