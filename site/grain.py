"""The stipple ground shared by the link-preview cards, the README banner, and the homepage hero.

A cobalt-to-amber gradient printed as frosted glass: random specks whose density follows the tone,
over a darker gradient, with light grain. Darker bands (``Band``) can sit behind text so white type
stays readable. The generator is seeded, so the same call draws the same pixels on every build.
"""

from __future__ import annotations

import io
from typing import NamedTuple

import numpy as np
from PIL import Image

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
