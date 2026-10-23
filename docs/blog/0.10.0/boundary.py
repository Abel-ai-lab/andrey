"""The 0.10.0 post's boundary maps, written next to it as HTML that the post includes.

``boundary-<family>.html``: for each method of a family, two maps over the number of variables
(columns). On the size map, the rows are the number of rows of data, with a mean degree of 2; on the
density map, the rows are the mean degree, with 2,000 rows. A cell's color is the method's median
errors as a share of the empty graph's, from the cobalt ramp, the deeper the fewer; a cell where it
does no better than the empty graph is gray; a mark is a problem where a fit stopped, on time or
memory, or failed. The outline around the cells inside the boundary is the boundary: in each row,
the sizes before the first one that does not finish or does not beat the empty graph. Every color is
an ``--andrey-*`` token.

The cells come from a feasibility file (the ``feasibility`` lane of a release card's summary: one
row per method, regime, and size). Run with
``uv run python docs/blog/0.10.0/boundary.py <cells.json> [outdir]``.

The maps are a reference for the measured conditions, not a hard limit: another graph generator,
other data, other settings, or another machine move a boundary.
"""

from __future__ import annotations

import json
import re
import sys
from html import escape
from pathlib import Path

from andrey.spec import list_specs

HERE = Path(__file__).parent
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
}  # fmt: skip
STOPS = {"time": ("×", "time cap"), "memory": ("■", "memory cap"), "error": ("!", "failed")}
# A regime with a fixed number of rows, e.g. ``deg2-n2000``; ``deg2-n10d`` scales them with d.
FIXED = re.compile(r"deg(\d+)-n(\d+)$")
SIZE_DEGREE, DENSITY_ROWS = 2, 2000
# The maps' geometry: a cell's width and height, the row labels' width, the gap between the two
# maps, and the room for the title above and the column labels below.
CELL_W, CELL_H, LABEL, GAP, TOP, BOTTOM = 30, 22, 78, 28, 22, 30


def maps(cells: list[dict]) -> dict[str, dict[tuple[int, int], dict]]:
    """One method's cells on each map, keyed by (row value, variables)."""
    out: dict[str, dict[tuple[int, int], dict]] = {"size": {}, "density": {}}
    for cell in cells:
        fixed = FIXED.match(cell["regime"])
        if not fixed:
            continue
        degree, n = int(fixed[1]), int(fixed[2])
        if degree == SIZE_DEGREE:
            out["size"][n, cell["d"]] = cell
        if n == DENSITY_ROWS:
            out["density"][degree, cell["d"]] = cell
    return out


def good(cell: dict | None) -> bool:
    """Inside the boundary: finished every dataset, with fewer errors than the empty graph."""
    return bool(cell and cell["stop"] == "none" and cell.get("beats_empty"))


def shade(share: float) -> str:
    """A cell's fill from the cobalt ramp: the deeper, the fewer errors than the empty graph's."""
    return f"var(--andrey-ramp-{1 + min(4, int((1 - min(max(share, 0.0), 1.0)) * 5))})"


def grid(cells: dict[tuple[int, int], dict], ys: list[int], ds: list[int], x0: float, label) -> str:
    """One map: its cells, stop marks, and the outline of the cells inside the boundary."""
    out, inside = [], set()
    for row, y in enumerate(reversed(ys)):  # the largest value at the top
        top = TOP + row * CELL_H
        out.append(
            f'<text class="bnd-y" x="{x0 + LABEL - 6}" y="{top + CELL_H / 2 + 4}" '
            f'text-anchor="end">{escape(label(y))}</text>'
        )
        stopped = False
        for col, d in enumerate(ds):
            cell = cells.get((y, d))
            left = x0 + LABEL + col * CELL_W
            if cell is None:
                continue
            stopped = stopped or not good(cell)
            if not stopped:
                inside.add((row, col))
            share = cell.get("share_of_empty")
            tip = f"{d:,} variables, {label(y)}"
            if share is not None:
                tip += f": {share:.0%} of the empty graph's errors"
            if cell["stop"] != "none":
                tip += f"; stopped: {STOPS[cell['stop']][1]}"
            if share is not None:
                fill = shade(share) if cell.get("beats_empty") else "var(--andrey-line)"
                out.append(
                    f'<rect x="{left + 1}" y="{top + 1}" width="{CELL_W - 2}" '
                    f'height="{CELL_H - 2}" rx="2" fill="{fill}">'
                    f"<title>{escape(tip)}</title></rect>"
                )
            if cell["stop"] != "none":
                out.append(
                    f'<text class="bnd-stop" x="{left + CELL_W / 2}" y="{top + CELL_H / 2 + 4}" '
                    f'text-anchor="middle">{STOPS[cell["stop"]][0]}<title>{escape(tip)}</title>'
                    "</text>"
                )
    # The outline: every side of a cell inside the boundary whose neighbor is outside it.
    sides = []
    for row, col in inside:
        left, top = x0 + LABEL + col * CELL_W, TOP + row * CELL_H
        right, bottom = left + CELL_W, top + CELL_H
        if (row - 1, col) not in inside:
            sides.append(f"M{left} {top}H{right}")
        if (row + 1, col) not in inside:
            sides.append(f"M{left} {bottom}H{right}")
        if (row, col - 1) not in inside:
            sides.append(f"M{left} {top}V{bottom}")
        if (row, col + 1) not in inside:
            sides.append(f"M{right} {top}V{bottom}")
    if sides:
        out.append(f'<path class="bnd-edge" d="{"".join(sorted(sides))}"/>')
    bottom = TOP + len(ys) * CELL_H
    for col, d in enumerate(ds):
        out.append(
            f'<text class="bnd-x" x="{x0 + LABEL + col * CELL_W + CELL_W / 2}" y="{bottom + 14}" '
            f'text-anchor="middle">{d:,}</text>'
        )
    return "".join(out)


def panel(method: str, cells: list[dict]) -> str:
    """A method's two maps side by side, sharing the columns of variables."""
    both = maps(cells)
    ds = sorted({d for cells in both.values() for _, d in cells})
    if not ds:
        return ""
    parts, x0 = [], 0.0
    titles = {"size": "Rows, mean degree 2", "density": "Mean degree, 2,000 rows"}
    labels = {"size": lambda n: f"{n:,} rows", "density": lambda k: f"degree {k}"}
    for key in ("size", "density"):
        ys = sorted({y for y, _ in both[key]})
        if not ys:
            continue
        parts.append(f'<text class="bnd-map" x="{x0 + LABEL}" y="14">{titles[key]}</text>')
        parts.append(grid(both[key], ys, ds, x0, labels[key]))
        x0 += LABEL + len(ds) * CELL_W + GAP
    rows = max(len({y for y, _ in both[key]}) for key in both)
    width, height = x0 - GAP, TOP + rows * CELL_H + BOTTOM
    name = NAMES.get(method, method)
    return (
        f'<div class="bnd-method"><p>{escape(name)}</p>'
        f'<svg viewBox="0 0 {width:g} {height}" role="img" aria-label="{escape(name)}: errors '
        f'relative to the empty graph by variables, rows, and mean degree">{"".join(parts)}</svg>'
        "</div>"
    )


def family_figure(family: str, cells: list[dict]) -> str:
    """A family's methods, two maps each."""
    names = {s.name: s.family for s in list_specs()}
    methods = [m for m in dict.fromkeys(c["method"] for c in cells) if names.get(m) == family]
    panels = "".join(panel(m, [c for c in cells if c["method"] == m]) for m in methods)
    return (
        f'<figure class="bnd"><p class="bnd-title">{escape(FAMILIES[family])}</p>{panels}'
        "<figcaption>Columns: the number of variables. Color: errors as a share of the empty "
        "graph's, deeper is fewer; gray does no better than the empty graph. × stopped at the time "
        "cap, ■ at the memory cap, ! failed. The outline is the boundary: in each row, the sizes "
        "before the first that does not finish or does not beat the empty graph.</figcaption>"
        "</figure>\n"
    )


def main() -> None:
    cells = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE
    for family in FAMILIES:
        (out / f"boundary-{family}.html").write_text(family_figure(family, cells), encoding="utf-8")


if __name__ == "__main__":
    main()
