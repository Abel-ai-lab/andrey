"""Each art's description in a sentence fragment, and the alt text drawn from it. Kept apart from
``cards.py``, which draws with Pillow, so pages that only name an art need no image library."""

from __future__ import annotations

COVERS = {
    "pair": "portraits of Andrey Markov and Andrey Kolmogorov, side by side",
    "lanes": "lanes rising into the sky, one amber streak far ahead",
    "iceberg": "an iceberg whose two peaks join into one faint mass under the waterline",
    "contour": "two summits drawn in contour lines, one dotted trail ending on each",
    "hourglass": "an hourglass whose sand falls one way only",
    "agentic": "an orrery whose rings carry a graph, one amber arm steering it",
    "start": "a cloud of data points gathering into a small graph, one arrow amber",
    "docs": "the pages of a vast atlas fanning open, one edge amber like a bookmark",
    "methods": "a prism fanning one beam into many rays, one amber",
    "tide": "the moon over a swelling sea, its light an amber path on the waves",
    "abacus": "an abacus seen at an angle, one bead amber",
    "punch-cards": "a shuffled fan of punch cards, one card amber",
    "harmonograph": "a harmonograph tracing a figure of cosine curves, one curve amber",
    "balloon": "a high-altitude balloon above the curve of the Earth, the atmosphere's edge amber",
    "ground": "the stipple ground",
}


def cover_alt(kind: str) -> str:
    """The cover's alt text: its art, in a sentence."""
    if kind not in COVERS:
        raise ValueError(f"unknown cover {kind!r}; expected one of {sorted(COVERS)}")
    return f"Cover: {COVERS[kind]}, drawn in navy lines"
