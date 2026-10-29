"""The 0.11.0 release card's tables, saved as HTML beside this script.

``ebic.html``: per method, the median errors with the standard BIC (``ebic_gamma=0``) and with the
extended BIC (``ebic_gamma=1``), the mean change in errors per true edge with its 95% interval, and
the datasets on which the extended BIC does better, the same, and worse.

``missing.html``: for PC and FCI with each test, by number of variables, rows, and missing values,
the median errors with test-wise deletion, with listwise deletion (beside the rows it keeps), on
the complete data, and of the empty graph, and the mean difference between the two deletions. A
size where listwise deletion keeps no rows has no fit, never an empty graph's score.

``bootstrap.html``: per method and rows, by tenth of edge frequency, the edges and the share of
them that are true, for adjacencies and orientations, and the averaged graph's errors against a
single fit's.

The numbers come from the card's summary (``SUMMARY``), its ``ebic``, ``missing_values``, and
``bootstrap`` lanes. The script runs with
``uv run python docs/docs/benchmarks/0.11.0/figures.py [summary.json]``.
"""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

HERE = Path(__file__).parent
SUMMARY = HERE.parents[3] / "benchmarks" / "published" / "0.11.0" / "summary.json"
# Each method as the card names it.
NAMES = {
    "ges": "GES", "xges": "XGES", "gies": "GIES", "hc": "Hill climbing", "tabu": "Tabu search",
    "exact_search": "Exact search", "boss": "BOSS", "grasp": "GRaSP", "mmhc": "MMHC",
    "rsmax2": "RSMAX2", "gfci": "GFCI", "pc": "PC", "fci": "FCI", "direct_lingam": "DirectLiNGAM",
}  # fmt: skip
TESTS = {"fisherz": "Fisher's z", "student_t": "Student's t"}
PATTERNS = {"mcar": "completely at random", "mar": "at random"}


def number(value: float | None, digits: int = 1) -> str:
    """A median or mean, or a dash where no fit finished."""
    return "-" if value is None else f"{value:,.{digits}f}"


def interval(value: dict, digits: int = 2) -> str:
    """A mean with its 95% interval."""
    return (
        f"{value['mean']:+.{digits}f}"
        f"<small>[{value['low']:+.{digits}f}, {value['high']:+.{digits}f}]</small>"
    )


def share(cell: dict) -> str:
    """A frequency bin's share of true edges, or a dash where no edge fell in it."""
    return "-" if cell["edges"] == 0 else f"{cell['share_true']:.0%}"


def figure(title: str, head: str, body: str, caption: str, text: bool = False) -> str:
    """One comparison table in the cards' style."""
    kind = "ct-table ct-text" if text else "ct-table"
    return (
        f'<figure class="ct"><p class="ct-title">{escape(title)}</p><div class="ct-scroll">'
        f'<table class="{kind}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
        f"<figcaption>{caption}</figcaption></figure>\n"
    )


def ebic_table(lane: dict) -> str:
    """Per method, the errors with each penalty and the change the extended BIC makes."""
    body = ""
    for row in lane["rows"]:
        name = NAMES.get(row["method"], row["method"])
        if row["default_gamma"] == 0.0:
            name += "<small>keeps the standard BIC</small>"
        median = row["median_shd"]
        body += (
            f'<tr><th scope="row">{name}</th><td>{number(median["gamma0"])}</td>'
            f'<td class="ct-ours">{number(median["gamma1"])}</td>'
            f'<td class="ct-ours">{interval(row["mean_difference_per_true_edge"])}</td>'
            f"<td>{row['better']} / {row['equal']} / {row['worse']}</td></tr>"
        )
    head = (
        '<th scope="col">Method</th><th scope="col">Standard BIC</th>'
        '<th scope="col" class="ct-ours">Extended BIC</th>'
        '<th scope="col" class="ct-ours">Change per true edge</th>'
        '<th scope="col">Better / equal / worse</th>'
    )
    return figure(
        "Errors with the standard and the extended BIC",
        head,
        body,
        "Median errors (SHD) over every dataset with each penalty; the change is the mean of the "
        "extended BIC's errors minus the standard BIC's, per true edge, with its 95% interval; "
        "better, equal, and worse count the datasets.",
    )


def missing_tables(lane: dict) -> str:
    """For each method and test, by size and missing values, the errors of each way to fit."""
    out = ""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in lane["rows"]:
        groups.setdefault((row["method"], row["indep_test"]), []).append(row)
    for (method, test), rows in groups.items():
        sizes: dict[tuple[int, int], list[dict]] = {}
        for row in rows:
            sizes.setdefault((row["d"], row["n"]), []).append(row)
        body = ""
        for (d, n), cells in sorted(sizes.items()):
            cells = sorted(cells, key=lambda r: (r["pattern"] != "mcar", r["rate"]))
            for k, row in enumerate(cells):
                arms = row["arms"]
                group = (
                    f'<th class="ct-group" rowspan="{len(cells)}"><span>{d} variables, '
                    f"{n:,} rows</span></th>"
                    if k == 0
                    else ""
                )
                missing = f"{row['rate']:.0%}" + (
                    "<small>at random</small>" if row["pattern"] == "mar" else ""
                )
                kept = row["rows_kept_listwise"]["median"]
                listwise = number(arms["listwise"]["median_shd"]) + (
                    f"<small>{kept:,.0f} row{'' if round(kept) == 1 else 's'}</small>"
                    if row["rate"]
                    else ""
                )
                diff = row.get("testwise_minus_listwise")
                body += (
                    f'<tr>{group}<th scope="row">{missing}</th>'
                    f'<td class="ct-ours">{number(arms["testwise"]["median_shd"])}</td>'
                    f"<td>{listwise}</td><td>{number(arms['complete']['median_shd'])}</td>"
                    f"<td>{number(arms['empty']['median_shd'], 0)}</td>"
                    f'<td class="ct-ours">'
                    f"{interval(diff, 1) if row['rate'] and diff and diff.get('pairs') else '-'}"
                    "</td></tr>"
                )
        head = (
            '<td colspan="2"></td><th scope="col" class="ct-ours">Test-wise</th>'
            '<th scope="col">Listwise</th><th scope="col">Complete data</th>'
            '<th scope="col">Empty graph</th>'
            '<th scope="col" class="ct-ours">Test-wise minus listwise</th>'
        )
        out += figure(
            f"{NAMES.get(method, method)} with {TESTS.get(test, test)}, missing values",
            head,
            body,
            "Median errors over the seeds. The missing share is of every value, completely at "
            "random unless marked; at random, the odd columns miss twice as often. Listwise "
            "deletion fits the complete rows only, whose median count is under its errors; a dash "
            "is a size where no fit finished, such as no complete rows left, never scored as an "
            "empty graph. The difference is the mean over the seeds, with its 95% interval.",
        )
    return out


def bootstrap_tables(lane: dict) -> str:
    """Per method, by tenth of edge frequency, the edges and the share that are true."""
    out = ""
    methods: dict[str, list[dict]] = {}
    for row in lane["rows"]:
        methods.setdefault(row["method"], []).append(row)
    for method, rows in methods.items():
        body, notes = "", []
        for row in sorted(rows, key=lambda r: r["n"]):
            adjacency, orientation = row["adjacency"]["bins"], row["orientation"]["bins"]
            for k, (a, o) in enumerate(zip(reversed(adjacency), reversed(orientation))):
                group = (
                    f'<th class="ct-group" rowspan="{len(adjacency)}"><span>{row["n"]:,} rows'
                    "</span></th>"
                    if k == 0
                    else ""
                )
                body += (
                    f'<tr>{group}<th scope="row">{a["lo"]:.1f} to {a["hi"]:.1f}</th>'
                    f'<td>{a["edges"]:,}</td><td class="ct-ours">{share(a)}</td>'
                    f'<td>{o["edges"]:,}</td><td class="ct-ours">{share(o)}</td></tr>'
                )
            averaged, single = row["averaged_graph"], row["single_fit"]
            notes.append(
                f"{row['n']:,} rows: the averaged graph's median errors "
                f"{number(averaged['median_shd'])}, a single fit's {number(single['median_shd'])}"
            )
        head = (
            '<td colspan="2"></td><th scope="col">Adjacencies</th>'
            '<th scope="col" class="ct-ours">True</th><th scope="col">Orientations</th>'
            '<th scope="col" class="ct-ours">True</th>'
        )
        out += figure(
            f"{NAMES.get(method, method)}: how often an edge appears, and how often it is true",
            head,
            body,
            "Each tenth of frequency counts the edges that appeared in that share of the "
            f"bootstrap fits, over {rows[0]['datasets']} datasets of {rows[0]['d']} variables, "
            "and the share of them in the true graph. " + "; ".join(notes) + ".",
        )
    return out


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SUMMARY
    summary = json.loads(path.read_text(encoding="utf-8"))
    (HERE / "ebic.html").write_text(ebic_table(summary["ebic"]), encoding="utf-8")
    (HERE / "missing.html").write_text(missing_tables(summary["missing_values"]), encoding="utf-8")
    (HERE / "bootstrap.html").write_text(bootstrap_tables(summary["bootstrap"]), encoding="utf-8")


if __name__ == "__main__":
    main()
