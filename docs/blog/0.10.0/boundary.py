"""The 0.10.0 post's boundary maps, written next to it as HTML that the post includes.

``boundary-<family>.html``: for each method of a family, two maps over the number of variables
(columns). On the size map, the rows are the number of rows of data, with a mean degree of 2; on the
density map, the rows are the mean degree, with 2,000 rows. A cell's color is the method's median
errors as a share of the empty graph's, from the cobalt ramp, the stronger the fewer; a cell where
it does no better than the empty graph is gray; a mark is a problem where a row stopped: on time or
memory, on a failed fit, or past the accuracy boundary, after two sizes in a row that do not beat
the empty graph. The outline around the cells inside the boundary is the boundary: in each row, the
sizes before the first one that does not finish or does not beat the empty graph. Every color is an
``--andrey-*`` token.

The cells come from the release card's summary (``SUMMARY``), its ``boundary`` lane: one ladder
per method and regime, each a list of rungs (one per number of variables) and how the ladder ended
(``end``). A rung that is not through, three or more of its fits cut, carries its cut as its mark;
the rung where a ladder ended past the accuracy boundary carries ``accuracy``. Run with
``uv run python docs/blog/0.10.0/boundary.py [summary.json] [outdir] [card_outdir]``. The release
card's ``ladders.html`` lists, as a table, the ladders the maps do not draw: the FCI and GFCI
diagonals, where the rows grow with the variables, and the latent-variable, time-series, and
pairwise methods.

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
SUMMARY = HERE.parents[2] / "benchmarks" / "published" / "0.10.0" / "summary.json"
# The release card's figures; it holds the ladders the maps do not draw as a table.
CARD = HERE.parents[1] / "docs" / "benchmarks" / "0.10.0"
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
    "multi_group_direct_lingam": "MultiGroupDirectLiNGAM", "rcd": "RCD",
    "bottom_up_parce_lingam": "BottomUpParceLiNGAM", "gin": "GIN", "anm": "ANM", "pnl": "PNL",
    "pcmci": "PCMCI", "pcmciplus": "PCMCI+", "lpcmci": "LPCMCI", "varma_lingam": "VARMA-LiNGAM",
    "longitudinal_lingam": "LongitudinalLiNGAM",
}  # fmt: skip
# The table's groups, by registry family, in the order the card lists them.
TABLE_FAMILIES = {
    "constraint": "Constraint-based",
    "lingam": "Linear non-Gaussian",
    "latent": "Latent variables",
    "temporal": "Time series",
    "pairwise": "Pairwise",
}
# How a ladder ended, in the table's words.
ENDS = {
    "time": "time cap",
    "memory": "memory cap",
    "error": "failed",
    "accuracy": "past the accuracy boundary",
    "ladder_end": "largest size measured",
    "open": "still running",
}
STOPS = {
    "time": ("×", "time cap"),
    "memory": ("■", "memory cap"),
    "error": ("!", "failed"),
    "accuracy": ("○", "past the accuracy boundary"),
}
# A regime with a fixed number of rows, e.g. ``deg2-n2000``; ``deg2-n10d`` scales them with d.
FIXED = re.compile(r"deg(\d+)-n(\d+)$")
SIZE_DEGREE, DENSITY_ROWS = 2, 2000
# A map's geometry: a cell's width and height, the row labels' width, and the room for the title
# above and the column labels below.
CELL_W, CELL_H, LABEL, TOP, BOTTOM = 30, 22, 78, 22, 30


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
    """A cell's fill from the cobalt ramp's top three steps, which stay apart from the gray of a
    cell that does no better than the empty graph in both themes: the stronger, the fewer errors."""
    return f"var(--andrey-ramp-{3 + min(2, int((1 - min(max(share, 0.0), 1.0)) * 3))})"


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
    """A method's two maps, each its own drawing at its own size, side by side where they fit and
    stacked on a narrow screen; both share the columns of variables."""
    both = maps(cells)
    ds = sorted({d for cells in both.values() for _, d in cells})
    if not ds:
        return ""
    name = NAMES.get(method, method)
    titles = {"size": "Rows, mean degree 2", "density": "Mean degree, 2,000 rows"}
    labels = {"size": lambda n: f"{n:,} rows", "density": lambda k: f"degree {k}"}
    drawings = []
    for key in ("size", "density"):
        ys = sorted({y for y, _ in both[key]})
        if not ys:
            continue
        width, height = LABEL + len(ds) * CELL_W, TOP + len(ys) * CELL_H + BOTTOM
        drawings.append(
            f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
            f'aria-label="{escape(name)}: errors relative to the empty graph, by variables and '
            f'{escape(titles[key].lower())}">'
            f'<text class="bnd-map" x="{LABEL}" y="14">{titles[key]}</text>'
            f"{grid(both[key], ys, ds, 0, labels[key])}</svg>"
        )
    return (
        f'<div class="bnd-method"><p>{escape(name)}</p>'
        f'<div class="bnd-maps">{"".join(drawings)}</div></div>'
    )


def family_figure(family: str, cells: list[dict]) -> str:
    """A family's methods, two maps each."""
    names = {s.name: s.family for s in list_specs()}
    methods = [m for m in dict.fromkeys(c["method"] for c in cells) if names.get(m) == family]
    panels = "".join(panel(m, [c for c in cells if c["method"] == m]) for m in methods)
    return (
        f'<figure class="bnd"><p class="bnd-title">{escape(FAMILIES[family])}</p>{panels}'
        "<figcaption>Columns: the number of variables. Color: errors as a share of the empty "
        "graph's, the stronger the fewer; gray does no better than the empty graph. × stopped at "
        "the time cap, ■ at the memory cap, ! failed, ○ past the accuracy boundary. The outline is "
        "the boundary: in each row, the sizes before the first that does not finish or does not "
        "beat the empty graph. A cell's tip gives its errors as a share of the empty graph's: on a "
        "denser graph the empty graph makes more errors, so beating it says less.</figcaption>"
        "</figure>\n"
    )


def cells(lane: dict) -> list[dict]:
    """The lane's rungs as map cells: a rung's method, regime, size, and errors, and its mark."""
    out = []
    for ladder in lane["ladders"]:
        end = ladder["end"]
        for rung in ladder["rungs"]:
            # A rung that is not through shows only its cut: its median, from the few fits that
            # finished, is not the rung's result.
            stop = "none" if rung["through"] else rung["stop"]
            if end["stop"] == "accuracy" and rung["d"] == end["d"]:
                stop = "accuracy"
            out.append(
                {
                    "method": ladder["method"],
                    "regime": ladder["regime"],
                    "d": rung["d"],
                    "share_of_empty": rung["share_of_empty"] if rung["through"] else None,
                    "beats_empty": rung["beats_empty"] if rung["through"] else None,
                    "stop": stop,
                }
            )
    return out


def mapped(ladder: dict) -> bool:
    """Whether a ladder is drawn on a map: a mapped family's fixed-row regime."""
    family = {s.name: s.family for s in list_specs()}[ladder["method"]]
    return family in FAMILIES and ladder["axis"] == "d" and bool(FIXED.match(ladder["regime"]))


def size(ladder: dict, value: int | None) -> str:
    """A ladder's size in its own unit: variables, rows, or latent variables."""
    if value is None:
        return "-"
    if ladder["axis"] == "n":
        return f"{value:,} rows"
    if ladder["axis"] == "latents":
        latents = next(r["latents"] for r in ladder["rungs"] if r["d"] == value)
        return f"{latents} latent variables"
    return f"{value:,} variables"


def ladders_table(lane: dict) -> str:
    """The ladders the maps do not draw, one row each, grouped by family: the largest size inside
    the boundary, and how and where the ladder ended."""
    family = {s.name: s.family for s in list_specs()}
    groups = {key: [] for key in TABLE_FAMILIES}
    for ladder in lane["ladders"]:
        if mapped(ladder):
            continue
        axis = "n" if ladder["axis"] == "n" else "d"
        inside = ladder[f"beats_empty_through_{axis}"]
        end = ladder["end"]
        where = end.get("latents") if ladder["axis"] == "latents" else end.get(axis)
        if end.get("reason", "").startswith("no fit attempted"):
            ended = f"no data could be generated at {size(ladder, where)}"
        elif ladder["axis"] == "latents" and where is not None:
            ended = f"{ENDS[end['stop']]}, {where} latent variables"
        else:
            ended = ENDS[end["stop"]] + (f", {size(ladder, where)}" if where is not None else "")
        groups[family[ladder["method"]]].append(
            "<tr>"
            f'<th scope="row">{escape(NAMES.get(ladder["method"], ladder["method"]))}'
            f"<small>{escape(ladder['label'])}</small></th>"
            f'<td class="ct-ours">{escape(size(ladder, inside))}</td>'
            f"<td>{escape(ended)}</td></tr>"
        )
    body = "".join(
        f'<tr><th class="ct-group" rowspan="{len(rows)}"><span>{escape(TABLE_FAMILIES[key])}</span>'
        f"</th>{rows[0][4:]}{''.join(rows[1:])}"
        for key, rows in groups.items()
        if rows
    )
    return (
        '<figure class="ct"><p class="ct-title">The other methods\' boundaries</p>'
        '<div class="ct-scroll"><table class="ct-table"><thead><tr><td colspan="2"></td>'
        '<th scope="col" class="ct-ours">Inside the boundary up to</th>'
        '<th scope="col">Ended</th></tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
        "<figcaption>Inside the boundary: the largest size up to which every size finished and "
        "beat the empty graph. For the latent-variable and LPCMCI ladders, the variables include "
        "the hidden ones.</figcaption></figure>\n"
    )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SUMMARY
    lane = json.loads(path.read_text(encoding="utf-8"))["boundary"]
    found = cells(lane)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE
    for family in FAMILIES:
        (out / f"boundary-{family}.html").write_text(family_figure(family, found), encoding="utf-8")
    card = Path(sys.argv[3]) if len(sys.argv) > 3 else CARD
    card.mkdir(exist_ok=True)
    (card / "ladders.html").write_text(ladders_table(lane), encoding="utf-8")


if __name__ == "__main__":
    main()
