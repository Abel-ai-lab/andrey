"""The 0.10.0 post's boundary figures, written next to it as HTML that the post includes.

``boundary-<family>.html``: for each method of a family, a heatmap over the number of variables
(columns) and the four conditions (rows: mean degree 2 or 4, 10 or 50 rows per variable). A cell's
color is the method's median errors as a share of the empty graph's, the deeper the fewer; a cell
where it does no better than the empty graph is gray; a mark ends a row where a fit stopped, on
time or memory, or where it failed. A line before the first size that does not beat the empty
graph or does not finish marks the row's boundary. Every color is an ``--andrey-*`` token.

The cells come from a feasibility file (``feasibility`` lane: one row per method, condition, and
size). Run with ``uv run python docs/blog/0.10.0/boundary.py <cells.json>``.
"""

from __future__ import annotations

import json
import math
import sys
from html import escape
from pathlib import Path

from andrey.spec import list_specs

HERE = Path(__file__).parent
# The graph methods' four conditions, in the order the rows show them.
CONDITIONS = {
    "deg2-n10d": "degree 2, 10 rows per variable",
    "deg2-n50d": "degree 2, 50 rows per variable",
    "deg4-n10d": "degree 4, 10 rows per variable",
    "deg4-n50d": "degree 4, 50 rows per variable",
}
FAMILIES = {
    "constraint": "Constraint-based",
    "score": "Score-based",
    "permutation": "Permutation-based",
    "lingam": "Linear non-Gaussian",
}
# Each method as the post names it.
NAMES = {
    "pc": "PC", "cdnod": "CD-NOD", "fci": "FCI", "gfci": "GFCI",
    "ges": "GES", "gies": "GIES", "xges": "XGES", "hc": "Hill climbing", "tabu": "Tabu search",
    "exact_search": "Exact search", "calm": "CALM", "boss": "BOSS", "grasp": "GRaSP",
    "direct_lingam": "DirectLiNGAM", "ica_lingam": "ICA-LiNGAM",
    "multi_group_direct_lingam": "MultiGroupDirectLiNGAM",
}  # fmt: skip
STOPS = {"time": ("×", "time cap"), "memory": ("■", "memory cap"), "error": ("!", "error")}
# A cell per measured size, centered on the logarithm of its number of variables (``DECADE`` pixels
# per tenfold), at most ``CELL`` wide.
ROW, LABEL, GAP, DECADE, CELL = 18, 250, 14, 220, 40


def place(d: int, low: int) -> float:
    """A size's center: its number of variables on a log scale from the family's smallest."""
    return LABEL + CELL / 2 + DECADE * math.log10(d / low)


def shade(share: float) -> str:
    """A cell's fill: the accent, deeper the fewer errors relative to the empty graph's."""
    strength = round(15 + 75 * (1 - min(max(share, 0.0), 1.0)))
    return f"color-mix(in srgb, var(--andrey-accent) {strength}%, var(--andrey-surface))"


def method_rows(cells: list[dict], low: int) -> tuple[str, int]:
    """One method's heatmap rows and their height: a row per condition it was measured in. A row's
    boundary line sits before the first size that does not beat the empty graph or stops."""
    out, y = [], 0
    for key, label in CONDITIONS.items():
        row = {c["d"]: c for c in cells if c["regime"] == key}
        if not row:
            continue
        out.append(
            f'<text class="bd-cond" x="{LABEL - 8}" y="{y + ROW / 2 + 4}" '
            f'text-anchor="end">{escape(label)}</text>'
        )
        edge, failed = None, False
        ds = sorted(row)
        for i, d in enumerate(ds):
            # As wide as CELL, or narrower where the next or previous size sits closer.
            near = [abs(place(o, low) - place(d, low)) for o in ds[max(i - 1, 0) : i + 2] if o != d]
            w = min([CELL, *near])
            x = place(d, low) - w / 2
            cell = row[d]
            share = cell.get("share_of_empty")
            good = share is not None and cell["beats_empty"] and cell["stop"] == "none"
            if not failed and not good:
                failed, edge = True, x
            elif not failed:
                edge = x + w
            tip = f"d = {d:,}, n = {cell['n']:,}"
            if share is not None:
                tip += f": {share:.0%} of the empty graph's errors"
            if cell["stop"] != "none":
                tip += f", stopped at the {STOPS[cell['stop']][1]}"
            if share is not None:
                fill = shade(share) if cell["beats_empty"] else "var(--andrey-line)"
                out.append(
                    f'<rect x="{x + 1:.1f}" y="{y + 1}" width="{w - 2:.1f}" height="{ROW - 2}" '
                    f'rx="3" fill="{fill}"><title>{escape(tip)}</title></rect>'
                )
            if cell["stop"] != "none":
                mark = STOPS[cell["stop"]][0]
                out.append(
                    f'<text class="bd-stop" x="{x + w / 2:.1f}" y="{y + ROW / 2 + 4}" '
                    f'text-anchor="middle">{mark}<title>{escape(tip)}</title></text>'
                )
        if edge is not None:
            out.append(
                f'<line class="bd-edge" x1="{edge:.1f}" x2="{edge:.1f}" y1="{y}" y2="{y + ROW}"/>'
            )
        y += ROW
    return "".join(out), y


def family_figure(family: str, cells: list[dict]) -> str:
    """A family's methods, one heatmap each, sharing the columns of variables."""
    names = {s.name: s.family for s in list_specs()}
    methods = [m for m in dict.fromkeys(c["method"] for c in cells) if names.get(m) == family]
    mine = [c for c in cells if c["method"] in methods and c["regime"] in CONDITIONS]
    sizes = sorted({c["d"] for c in mine})
    low = sizes[0]
    ticks = [t for t in (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if low <= t <= sizes[-1]]
    parts, y = [], 24
    parts.append(
        "".join(
            f'<text class="bd-size" x="{place(t, low):.1f}" y="14" text-anchor="middle">{t:,}'
            "</text>"
            for t in ticks
        )
    )
    for method in methods:
        rows, height = method_rows([c for c in mine if c["method"] == method], low)
        if not height:
            continue
        parts.append(
            f'<text class="bd-method" x="0" y="{y + 12}">{escape(NAMES.get(method, method))}'
            f'</text><g transform="translate(0 {y})">{rows}</g>'
        )
        y += height + GAP
    width = place(sizes[-1], low) + CELL / 2
    return (
        f'<figure class="bd"><p class="bd-title">{escape(FAMILIES[family])}</p>'
        f'<svg viewBox="0 0 {width} {y}" role="img" aria-label="{escape(FAMILIES[family])}: '
        f'errors relative to the empty graph by number of variables">{"".join(parts)}</svg>'
        "<figcaption>Columns: variables. Color: errors as a share of the empty graph's, deeper "
        "is fewer; gray does no better than the empty graph. × stopped at the time cap, ■ at "
        "the memory cap, ! failed. The line marks the boundary: before the first size that does "
        "not beat the empty graph or does not finish.</figcaption></figure>\n"
    )


def main() -> None:
    cells = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE
    for family in FAMILIES:
        (out / f"boundary-{family}.html").write_text(family_figure(family, cells), encoding="utf-8")


if __name__ == "__main__":
    main()
