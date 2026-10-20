"""The 0.8.0 post's figures, written next to it as HTML that the post includes.

- ``accuracy.html``: PC's errors on nonlinear data with each new test, its reference package's
  implementation, partial correlation, and the empty graph, by graph size.
- ``speed.html``: how much faster PC runs with each test than with its reference package, and on
  how many datasets it makes no more errors.

Every number comes from the 0.8.0 release card's summary (``SUMMARY``), so the figures change only
with it. Every color is an ``--andrey-*`` token, from the post's table classes. Run with
``uv run python docs/blog/0.8.0/figures.py [summary.json]``.
"""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

HERE = Path(__file__).parent
SUMMARY = HERE.parents[2] / "benchmarks" / "published" / "0.8.0" / "summary.json"
# The four tests in the post's order, by the indep_test name in each row's solution.
TESTS = {"kci": "KCI", "rcot": "RCoT", "cmiknn": "CMIknn", "gpdc": "GPDC"}


def table(title: str, columns: tuple[str, ...], groups: tuple, rank=None, note: str = "") -> str:
    """A comparison table in the post's one layout: rows grouped under a label set on its side,
    Andrey's column first and tinted, each row's best cell in bold.

    ``groups`` holds ``(label, rows)``, each row ``(name, detail, cells)`` with a cell per column:
    a number, a word, or None for a cell with no measurement. ``rank`` orders a row's cells,
    lowest best; it defaults to the number itself, and a cell it ranks None is never best.
    """
    rank = rank or (lambda cell: cell)
    ours = ' class="ct-ours"'
    head = "".join(
        f'<th scope="col"{ours if k == 0 else ""}>{escape(c)}</th>' for k, c in enumerate(columns)
    )

    def row(name: str, detail: str, cells: tuple) -> str:
        ranks = [rank(c) for c in cells if c is not None and rank(c) is not None]
        best = min(ranks) if ranks else None
        tds = []
        for k, c in enumerate(cells):
            cls = ["ct-ours"] if k == 0 else []
            if c is None:
                tds.append(f'<td class="{" ".join([*cls, "ct-na"])}">-</td>')
                continue
            if rank(c) is not None and rank(c) == best and len(set(ranks)) > 1:
                cls.append("ct-best")
            text = f"{c:,}" if isinstance(c, int) else escape(str(c))
            tds.append(f'<td class="{" ".join(cls)}">{text}</td>')
        small = f"<small>{escape(detail)}</small>" if detail else ""
        return f'<th scope="row">{escape(name)}{small}</th>{"".join(tds)}'

    body = "".join(
        "".join(
            "<tr>"
            + (
                f'<th class="ct-group" rowspan="{len(rows)}"><span>{escape(label)}</span></th>'
                if i == 0
                else ""
            )
            + row(*r)
            + "</tr>"
            for i, r in enumerate(rows)
        )
        for label, rows in groups
    )
    caption = f"<figcaption>{escape(note)}</figcaption>" if note else ""
    return (
        f'<figure class="ct"><p class="ct-title">{escape(title)}</p><div class="ct-scroll">'
        f'<table class="ct-table"><thead><tr><td colspan="2"></td>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>{caption}</figure>\n"
    )


def pc_rows(summary: dict) -> dict[str, dict]:
    """Each test's PC entry in the summary's ``methods``, keyed by its ``indep_test`` name."""
    found = {}
    for method in summary["methods"]:
        parts = method["solution"].split(".")
        if parts[:2] == ["andrey", "pc"] and parts[2] in TESTS and method["lane"] == "nonlinear":
            found[parts[2]] = method
    missing = set(TESTS) - set(found)
    if missing:
        raise ValueError(f"the summary has no PC row for {sorted(missing)}")
    return found


def shd(value: float | None) -> str | None:
    """A median SHD as the table shows it: whole, or with its half."""
    if value is None:
        return None
    return f"{value:g}"


def accuracy_table(summary: dict) -> str:
    # Median SHD per size, over the datasets both packages finished. The empty graph is a floor
    # every method should beat, so it is never the row's best.
    groups = []
    for key, name in TESTS.items():
        method = pc_rows(summary)[key]
        rows = tuple(
            (
                f"{r['d']} variables",
                f"{r['n']:,} rows",
                (
                    shd(r["andrey_shd"]),
                    shd(r["comparator_shd"]),
                    shd(r.get("fisherz_shd")),
                    f"{r['empty_shd']:g} *",
                ),
            )
            for r in method["rows"]
            if r["paired"]
        )
        groups.append((f"{name} vs {method['comparator']}", rows))
    return table(
        "PC's errors on nonlinear data (median SHD, lower is better)",
        ("Andrey", "Reference package", "Partial correlation", "Empty graph"),
        tuple(groups),
        rank=lambda cell: None if cell.endswith("*") else float(cell),
        note=(
            "Median structural Hamming distance between PC's equivalence class and the true one, "
            "over the datasets both packages finished. Partial correlation is Andrey's PC with "
            "fisherz on the same datasets. * The empty graph's errors, the floor every test "
            "should beat."
        ),
    )


def speed_table(summary: dict) -> str:
    # Speedup range over the sizes measured (a lower bound where the reference hit the time
    # cap), and the datasets where Andrey's graph had no more errors than the reference's.
    rows = []
    for key, name in TESTS.items():
        method = pc_rows(summary)[key]
        measured = [r for r in method["rows"] if r["paired"]]
        low = min(r["ratio"] for r in measured)
        high = max(r["ratio"] for r in measured)
        capped = any(r.get("ratio_is_lower_bound") for r in measured)
        paired = sum(r["paired"] for r in measured)
        no_more = paired - sum(r["comparator_fewer_errors"] for r in measured)
        rows.append(
            (
                name,
                f"vs {method['comparator']}",
                (
                    f"{low:.1f}-{high:.1f}x{' +' if capped else ''}",
                    f"{round(100 * no_more / paired)}% of {paired}",
                ),
            )
        )
    return table(
        "PC with each test, against its reference package",
        ("Faster by", "No more errors"),
        (("Nonlinear tests", tuple(rows)),),
        rank=lambda cell: None,
        note=(
            "Median fit time of the reference over Andrey's, per size; + where the reference hit "
            "the time cap, so the speedup is at least that. No more errors: the datasets where "
            "Andrey's graph had no more errors than the reference's."
        ),
    )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SUMMARY
    summary = json.loads(path.read_text(encoding="utf-8"))
    (HERE / "accuracy.html").write_text(accuracy_table(summary), encoding="utf-8")
    (HERE / "speed.html").write_text(speed_table(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
