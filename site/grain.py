"""The stipple ground shared by the link-preview cards, the README banner, and the homepage hero.

A cobalt-to-amber gradient printed as frosted glass: random specks whose density follows the tone,
over a darker gradient, with light grain. Darker bands (``Band``) can sit behind text so white type
stays readable. The generator is seeded, so the same call draws the same pixels on every build.
"""

from __future__ import annotations

import io
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

import andrey.viz

NAVY = "#121d5e"  # the ground's darkest tone, below the palette's darkest cobalt
COBALT = andrey.viz.LIGHT.ramp[3]
COBALT_LIGHT = andrey.viz.LIGHT.ramp[2]
AMBER = andrey.viz.LIGHT.categorical[1]
SEED = 7
SCALE = 2  # drawn at twice the size, then downsampled, for clean speck edges


class Band(NamedTuple):
    """A darker ellipse behind text: its center, its half-width and half-height, and its depth."""

    x: float
    y: float
    rx: float
    ry: float
    depth: float = 0.8


def _rgb(color: str) -> np.ndarray:
    return np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], float)


def _blob(xs: np.ndarray, ys: np.ndarray, x: float, y: float, r: float) -> np.ndarray:
    return np.exp(-(((xs - x) / r) ** 2 + ((ys - y) / r) ** 2))


def stipple(width: int, height: int, bands: tuple[Band, ...] = ()) -> Image.Image:
    """The ground at ``width`` x ``height``: cobalt light from the top left, amber at the bottom
    right, each ``Band`` darkened. Positions scale from a 1200 x 630 layout."""
    rng = np.random.default_rng(SEED)
    sx, sy = width / 1200, height / 630
    ys, xs = np.mgrid[0 : height * SCALE, 0 : width * SCALE].astype(float) / SCALE
    xs, ys = xs / sx, ys / sy  # layout coordinates
    shade = sum(
        (np.exp(-(((xs - b.x) / b.rx) ** 2 + ((ys - b.y) / b.ry) ** 2)) * b.depth for b in bands),
        np.zeros_like(xs),
    )
    tone = np.clip(
        _blob(xs, ys, 220, 60, 700) * 0.95 + _blob(xs, ys, 640, 300, 560) * 0.45 - shade, 0, 1
    )
    amber = np.clip(_blob(xs, ys, 1200, 690, 380) * 1.1 - shade, 0, 1)
    img = _rgb(NAVY) * (1 - tone[..., None] * 0.5) + _rgb(COBALT) * tone[..., None] * 0.5
    speck = rng.random(tone.shape)
    img[speck < tone**1.4 * 0.75] = _rgb(COBALT_LIGHT)
    img[speck > 1 - amber**1.3 * 0.8] = _rgb(AMBER)
    img = img + rng.normal(0, 14, tone.shape)[..., None]
    img = img + rng.normal(0, 6, tone.shape)[..., None]  # a finer grain over the specks
    pixels = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels).resize((width, height), Image.LANCZOS)


def speck_tile(color: str, density: float, size: int = 256, seed: int = SEED) -> bytes:
    """A repeating tile of ``color`` specks on transparency, as a two-color PNG.

    The homepage lays these over CSS gradients and fades them with masks, which draws the cards'
    stipple at any width for a few kilobytes.
    """
    rng = np.random.default_rng(seed)
    specks = rng.random((size, size)) < density
    tile = Image.fromarray(specks.astype(np.uint8), "P")
    tile.putpalette([0, 0, 0, *_rgb(color).astype(int)])
    buffer = io.BytesIO()
    tile.save(buffer, "PNG", transparency=0, optimize=True)
    return buffer.getvalue()


def grain_tile(size: int = 256, seed: int = SEED + 1) -> bytes:
    """A repeating tile of faint light and dark grain, as a three-color PNG."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, (size, size))
    pixels = np.where(noise > 1.2, 2, np.where(noise < -1.2, 1, 0)).astype(np.uint8)
    tile = Image.fromarray(pixels, "P")
    tile.putpalette([0, 0, 0, 0, 0, 0, 255, 255, 255])
    buffer = io.BytesIO()
    tile.save(buffer, "PNG", transparency=bytes([0, 60, 60]), optimize=True)
    return buffer.getvalue()


def hero_tiles() -> tuple[tuple[str, bytes], ...]:
    """The homepage hero's tiles, by file name: cobalt specks, amber specks, and grain."""
    return (
        ("grain-cobalt.png", speck_tile(COBALT_LIGHT, 0.55)),
        ("grain-amber.png", speck_tile(AMBER, 0.6, seed=SEED + 2)),
        ("grain-noise.png", grain_tile()),
    )


def ground_tokens() -> dict[str, str]:
    """The ground's tones as CSS custom properties, the same in both themes: the page's CSS draws
    the hero with these and names no color itself."""
    return {
        "--andrey-ground-navy": NAVY,
        "--andrey-ground-cobalt": COBALT,
        "--andrey-ground-cobalt-light": COBALT_LIGHT,
        "--andrey-ground-amber": AMBER,
        "--andrey-ground-ink": "#ffffff",
        "--andrey-ground-muted": andrey.viz.LIGHT.ramp[0],
        # The dark palette's green, which reads on the navy ground in either theme.
        "--andrey-ground-green": andrey.viz.DARK.categorical[2],
    }


PORTRAIT_SEED = 11


class Portrait(NamedTuple):
    """A grayscale cut-out with alpha, and where its head is, in source pixels.

    ``eye`` is the midpoint between the eyes and ``crown`` the top of the head, so two portraits
    can be drawn with heads of one size at one eye height. ``turn`` rotates the photo about the
    eyes, in degrees counter-clockwise, to level them. ``floor``, ``gamma``, and ``fade`` shape
    the line screen: the darkness below ``floor`` draws no line, ``gamma`` bends the rest, and
    the lines fade out over that share of the width on the side toward the middle.
    """

    source: Image.Image
    eye: tuple[float, float]
    crown: float
    turn: float = 0.0
    floor: float = 0.12
    gamma: float = 0.8
    fade: float = 0.28


def pair(
    width: int,
    height: int,
    left: Portrait,
    right: Portrait,
    *,
    head: float = 0.30,
    eyes: float = 0.55,
    inset: float = 0.16,
) -> Image.Image:
    """Two portraits as navy line screens rising from light cobalt glows, amber below the middle.

    Both heads are ``head`` of the height from crown to eyes, with their eyes at ``eyes`` of the
    height and ``inset`` of the width from the nearer edge; each fades toward the middle. Seeded,
    so the same call draws the same pixels on every build.
    """
    rng = np.random.default_rng(PORTRAIT_SEED)
    ys, xs = np.mgrid[0 : height * SCALE, 0 : width * SCALE].astype(float) / SCALE
    ex, ey, radius = inset * width, eyes * height, height * 0.8
    glow = np.zeros_like(xs)
    for gx in (width - ex, ex):
        glow = np.maximum(
            glow, np.exp(-(((xs - gx) / radius) ** 2 + ((ys - ey) / (radius * 0.95)) ** 2))
        )
    tone = np.clip(glow * 0.95, 0, 1)
    amber = np.clip(
        np.exp(
            -(
                ((xs - 0.5 * width) / (0.3 * width)) ** 2
                + ((ys - 1.08 * height) / (0.35 * height)) ** 2
            )
        )
        * 0.9,
        0,
        1,
    )
    img = (
        _rgb(NAVY) * (1 - tone[..., None])
        + _rgb(COBALT_LIGHT) * tone[..., None] * 0.85
        + _rgb(COBALT) * tone[..., None] * 0.15
    )
    speck = rng.random(tone.shape)
    img[speck < tone**1.6 * 0.35] = _rgb(andrey.viz.LIGHT.ramp[1])
    img[speck > 1 - amber**1.3 * 0.8] = _rgb(AMBER)
    img += rng.normal(0, 12, tone.shape)[..., None]
    for portrait, x, toward in ((right, width - ex, "left"), (left, ex, "right")):
        _place(img, portrait, x, ey, head * height, toward)
    pixels = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels).resize((width, height), Image.LANCZOS)


def _place(
    img: np.ndarray, portrait: Portrait, x: float, y: float, head: float, toward: str
) -> None:
    """Draw ``portrait`` with its eyes at (``x``, ``y``) and ``head`` from crown to eyes."""
    source = portrait.source
    if portrait.turn:
        source = source.rotate(portrait.turn, resample=Image.BICUBIC, center=portrait.eye)
    scale = head / (portrait.eye[1] - portrait.crown)
    cx = x + (source.width / 2 - portrait.eye[0]) * scale
    cy = y + (source.height / 2 - portrait.eye[1]) * scale
    _line_screen(img, source, cx, cy, source.height * scale, portrait, toward)


def _line_screen(
    img: np.ndarray,
    source: Image.Image,
    cx: float,
    cy: float,
    size: float,
    shape: Portrait,
    toward: str,
) -> None:
    """Draw ``source`` into ``img`` as horizontal navy lines whose thickness follows its darkness,
    ``size`` tall and centered on (``cx``, ``cy``), fading toward ``toward`` and the top."""
    grey, alpha = source.convert("LA").split()
    grey = ImageOps.autocontrast(grey, cutoff=1)
    ph = int(size * SCALE)
    pw = int(grey.width * ph / grey.height)
    grey = grey.resize((pw, ph), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.2))
    keep = alpha.resize((pw, ph), Image.LANCZOS).filter(ImageFilter.GaussianBlur(2 * SCALE))
    dark = 1 - np.asarray(grey, float) / 255
    dark = np.clip((dark - shape.floor) / (1 - shape.floor), 0, 1) ** shape.gamma
    dark *= np.asarray(keep, float) / 255  # the cut-out's transparent ground draws no line
    yy, xx = np.mgrid[0:ph, 0:pw].astype(float)
    side = xx if toward == "left" else pw - 1 - xx
    dark *= np.clip(side / (pw * shape.fade), 0, 1) ** 1.3 * np.clip(yy / (ph * 0.05), 0, 1)
    x0, y0 = int(cx * SCALE - pw / 2), int(cy * SCALE - ph / 2)
    pitch = 7.0 * SCALE
    rows = np.arange(ph)
    centre = (np.floor(rows / pitch) + 0.5) * pitch
    band = centre.astype(int).clip(0, ph - 1)
    wobble = 0.6 * SCALE * np.sin(np.arange(pw)[None, :] / (9 * SCALE))
    on = np.abs(rows[:, None] - centre[:, None] + wobble) < dark[band, :] * pitch * 0.5
    shade = (rows / ph)[:, None, None]
    colour = _rgb(andrey.viz.LIGHT.ramp[4]) * (1 - shade) + _rgb(NAVY) * shade
    x_lo, y_lo = max(x0, 0), max(y0, 0)
    x_hi, y_hi = min(x0 + pw, img.shape[1]), min(y0 + ph, img.shape[0])
    inside = on[y_lo - y0 : y_hi - y0, x_lo - x0 : x_hi - x0]
    tint = np.broadcast_to(colour[y_lo - y0 : y_hi - y0], inside.shape + (3,))
    img[y_lo:y_hi, x_lo:x_hi][inside] = tint[inside]
