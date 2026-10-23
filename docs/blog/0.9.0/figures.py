"""The 0.9.0 post's speed figure, saved as HTML beside this script for inclusion in the post.

``ranges.html``: each sped-up method against Andrey 0.8.0, grouped by family, as a bar on a
logarithmic axis that starts at 1x, as fast as 0.8.0: the range over the measured sizes of 0.8.0's
median fit time over 0.9.0's, with one worker. The release card has every size.

The numbers come from the 0.9.0 release card's summary (``SUMMARY``), its ``speed`` lane: one row
per method, regime, size, and worker count. Every color is an ``--andrey-*`` token. The script
runs with ``uv run python docs/blog/0.9.0/figures.py [summary.json]``.
"""

from __future__ import annotations

import json
import math
import sys
from html import escape
from pathlib import Path

from andrey.spec import list_specs

HERE = Path(__file__).parent
SUMMARY = HERE.parents[2] / "benchmarks" / "published" / "0.9.0" / "summary.json"
FAMILIES = {
    "constraint": "Constraint",
    "score": "Score",
    "permutation": "Permutation",
    "lingam": "LiNGAM",
    "pairwise": "Pairwise",
    "temporal": "Temporal",
    "latent": "Latent",
}
# Each method as the post names it.
NAMES = {
    "pc": "PC", "fci": "FCI", "gfci": "GFCI", "cdnod": "CD-NOD", "ges": "GES", "xges": "XGES",
    "exact_search": "Exact search", "boss": "BOSS", "grasp": "GRaSP",
    "direct_lingam": "DirectLiNGAM", "ica_lingam": "ICA-LiNGAM", "anm": "ANM", "pnl": "PNL",
    "lpcmci": "LPCMCI",
}  # fmt: skip
# The chart: a row's height, the method labels' width, and the bars' width; the axis's ticks.
ROW, LABEL, PLOT = 22, 110, 400
TICKS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)


def floored(value: float) -> str:
    """``value`` rounded down to two significant figures, so a speedup is never overstated."""
    digits = 1 - math.floor(math.log10(value))
    return f"{math.floor(value * 10**digits) / 10**digits:g}"


def ranges(summary: dict) -> list[tuple[str, list[tuple[str, float, float, bool]]]]:
    """Per family, each method's slowest and fastest speedup with one worker, and same output."""
    family = {s.name: s.family for s in list_specs()}
    speeds: dict[str, list[float]] = {}
    same: dict[str, bool] = {}
    for row in summary["speed"]:
        if row.get("workers", 1) != 1 or not row.get("speedup"):
            continue
        speeds.setdefault(row["method"], []).append(row["speedup"])
        same[row["method"]] = same.get(row["method"], True) and bool(row["same_output"])
    groups = [
        (label, [(m, min(speeds[m]), max(speeds[m]), same[m]) for m in speeds if family[m] == key])
        for key, label in FAMILIES.items()
    ]
    return [(label, methods) for label, methods in groups if methods]


def ranges_chart(summary: dict) -> str:
    """Each method's range of speedups as a bar on a log axis from 1x, grouped by family."""
    groups = ranges(summary)
    top = max(hi for _, methods in groups for _, _, hi, _ in methods)
    end = next(t for t in TICKS if t >= top)
    scale = PLOT / math.log10(end)

    def x(speed: float) -> float:
        return LABEL + scale * math.log10(max(speed, 1.0))

    parts, y = [], 8
    for label, methods in groups:
        parts.append(f'<text class="sr-family" x="0" y="{y + 12}">{escape(label)}</text>')
        y += ROW
        for m, lo, hi, _ in methods:
            tip = f"{NAMES.get(m, m)}: {floored(lo)}x to {hi:.2g}x faster than 0.8.0"
            parts.append(
                f'<text class="sr-method" x="{LABEL - 8}" y="{y + 14}" text-anchor="end">'
                f"{escape(NAMES.get(m, m))}</text>"
                f'<rect class="sr-bar" x="{x(lo):.1f}" y="{y + 5}" '
                f'width="{max(x(hi) - x(lo), 4):.1f}" height="{ROW - 10}" rx="6">'
                f"<title>{escape(tip)}</title></rect>"
                f'<text class="sr-value" x="{x(hi) + 6:.1f}" y="{y + 15}">{hi:.2g}x</text>'
            )
            y += ROW
        y += 6
    ticks = "".join(
        f'<line class="sr-grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="0" y2="{y}"/>'
        f'<text class="sr-tick" x="{x(t):.1f}" y="{y + 14}" text-anchor="middle">{t}x</text>'
        for t in TICKS
        if t <= end
    )
    width, height = LABEL + PLOT + 50, y + 22
    return (
        '<figure class="sr"><p class="sr-title">How much faster than Andrey 0.8.0</p>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Each method\'s speedup over '
        f'Andrey 0.8.0, from its slowest to its fastest size">{ticks}{"".join(parts)}</svg>'
        "<figcaption>Each bar runs from a method's smallest to its largest speedup over the "
        "measured sizes, with one worker. 1x is as fast as 0.8.0.</figcaption></figure>\n"
    )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SUMMARY
    summary = json.loads(path.read_text(encoding="utf-8"))
    (HERE / "ranges.html").write_text(ranges_chart(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
