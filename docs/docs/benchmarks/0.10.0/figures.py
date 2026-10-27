"""The 0.10.0 release card's comparison with Tetrad, saved as HTML beside this script.

``tetrad.html``: for PC, GES, BOSS, GRaSP, and FCI, one table each over the measured sizes: both
packages' median fit time on the same datasets, the speedup, and both packages' errors. Where a
diagnosis rerun covered more datasets (``matched``), the errors are its means over them, at the
row's own settings; otherwise they are the medians over the row's datasets.

The rows come from the card's summary (``SUMMARY``), the ``comparisons`` of its ``speed`` lane, with
one worker. The script runs with
``uv run python docs/docs/benchmarks/0.10.0/figures.py [summary.json]``.
"""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

HERE = Path(__file__).parent
SUMMARY = HERE.parents[3] / "benchmarks" / "published" / "0.10.0" / "summary.json"
# Each method as the card names it, and the Tetrad search it runs against.
METHODS = {
    "pc": ("PC", "PC"),
    "ges": ("GES", "FGES"),
    "boss": ("BOSS", "BOSS"),
    "grasp": ("GRaSP", "GRaSP"),
    "fci": ("FCI", "FCI"),
}


def figure(text: float) -> str:
    """A number with three significant figures, as the card's other tables give them."""
    return f"{text:#.3g}".rstrip(".")


def errors(row: dict) -> tuple[str, str, int]:
    """Both packages' errors and the datasets behind them: the matched means where there are any."""
    if row.get("matched"):
        same = row["matched"][0]
        return f"{same['errors_mean_090']:.1f}", f"{same['errors_mean_ref']:.1f}", same["datasets"]
    return f"{row['errors_090']:.0f}", f"{row['errors_ref']:.0f}", row["datasets"]


def method_table(method: str, rows: list[dict]) -> str:
    """One method's sizes: both times, the speedup, and both packages' errors."""
    name, theirs = METHODS[method]
    body = ""
    for row in sorted(rows, key=lambda r: r["d"]):
        ours_errors, their_errors, datasets = errors(row)
        body += (
            f'<tr><th scope="row">{row["d"]:,}<small>{row["n"]:,} rows</small></th>'
            f'<td class="ct-ours">{figure(row["seconds_090"])}</td>'
            f"<td>{figure(row['seconds_ref'])}</td>"
            f'<td class="ct-ours">{figure(row["speedup"])}x</td>'
            f'<td class="ct-ours">{ours_errors}</td><td>{their_errors}</td>'
            f"<td>{datasets}</td></tr>"
        )
    return (
        f'<figure class="ct"><p class="ct-title">{escape(name)} against Tetrad\'s '
        f"{escape(theirs)}</p>"
        '<div class="ct-scroll"><table class="ct-table"><thead><tr><th scope="col">Variables</th>'
        '<th scope="col" class="ct-ours">Andrey (s)</th><th scope="col">Tetrad (s)</th>'
        '<th scope="col" class="ct-ours">Faster</th><th scope="col" class="ct-ours">'
        'Andrey errors</th><th scope="col">Tetrad errors</th><th scope="col">Datasets</th>'
        f"</tr></thead><tbody>{body}</tbody></table></div></figure>\n"
    )


def tetrad_tables(summary: dict) -> str:
    """A table per method, in the card's order, from the one-worker comparisons with Tetrad."""
    rows = [
        r
        for r in summary["speed"]["comparisons"]
        if r["comparator"] == "Tetrad" and r.get("workers", 1) == 1
    ]
    return "".join(
        method_table(m, [r for r in rows if r["method"] == m])
        for m in METHODS
        if any(r["method"] == m for r in rows)
    )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SUMMARY
    summary = json.loads(path.read_text(encoding="utf-8"))
    (HERE / "tetrad.html").write_text(tetrad_tables(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
