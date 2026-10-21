"""The 0.8.0 post's figures, written next to it as HTML that the post includes.

- ``dependence.html``: three relationships, linear, cos, and square, each with the p-value partial
  correlation gives it and the one KCI gives it.
- ``graphs.html``: the post's example, the true graph beside the graph PC finds with partial
  correlation and with each new test.
- ``accuracy.html``: PC's errors on nonlinear data with each new test, its reference package's
  implementation, partial correlation, and the empty graph, by graph size.
- ``speed.html``: how much faster PC runs with each test than with its reference package, and on
  how many datasets it makes no more errors.

The tables' numbers come from the 0.8.0 release card's summary (``SUMMARY``), so they change only
with it; the other two figures are fitted here, on fixed seeds. Every color is an ``--andrey-*``
token. Run with ``uv run python docs/blog/0.8.0/figures.py [summary.json]``.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import warnings
from html import escape
from pathlib import Path

import numpy as np

import andrey
from andrey import GraphStructure
from andrey.core.ci import make_indep_test
from andrey.data import sample_scm
from andrey.viz import LIGHT, Palette, draw

HERE = Path(__file__).parent
SUMMARY = HERE.parents[2] / "benchmarks" / "published" / "0.8.0" / "summary.json"
# The four tests in the post's order, by the indep_test name in each row's solution.
TESTS = {"kci": "KCI", "rcot": "RCoT", "cmiknn": "CMIknn", "gpdc": "GPDC"}
PALETTE = Palette(
    **{
        field.name: (
            getattr(LIGHT, field.name)
            if isinstance(getattr(LIGHT, field.name), tuple)
            else f"var(--andrey-{field.name.replace('_', '-')})"
        )
        for field in dataclasses.fields(LIGHT)
    }
)


def listed(names: list[str]) -> str:
    """Names joined as a sentence lists them: A, B, and C."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f", and {names[-1]}"


def p_text(p: float) -> str:
    """A p-value as the figures print it."""
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def dependence_figure() -> str:
    # 400 draws of x, and y as a line, a cos, and a square of x with the same noise: each
    # relationship's points on the left of its row, and on the right whether each test finds it.
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, 400)
    noise = rng.normal(0, 0.3, 400)
    shapes = (
        ("y = x", 0.5 * x),
        ("y = cos(x)", np.cos(x)),
        ("y = x²", x**2 / 3 - 1),
    )
    warnings.simplefilter("ignore", andrey.ExperimentalWarning)
    tests = (("fisherz", "Partial correlation"), *TESTS.items())
    w, h, pad = 180, 110, 8
    rows, results = [], []
    for name, signal in shapes:
        y = signal + noise
        data = np.column_stack([x, y])
        found = {test: make_indep_test(test, data)(0, 1, ()) for test in dict(tests)}
        results.append(found)
        sx = lambda v: pad + (v + 3) / 6 * (w - 2 * pad)  # noqa: E731
        lo, hi = float(y.min()), float(y.max())
        sy = lambda v: h - pad - (v - lo) / (hi - lo) * (h - 2 * pad)  # noqa: E731
        dots = "".join(
            f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="1.6"/>'
            for a, b in zip(x, y, strict=True)
        )
        cells = "".join(
            f'<td class="{"dp-seen" if found[t] < 0.05 else "dp-missed"}">'
            f'<b aria-label="{"finds it" if found[t] < 0.05 else "misses it"}">'
            f"{'✓' if found[t] < 0.05 else '✗'}</b>"
            f"<small>{escape(p_text(found[t]))}</small></td>"
            for t in dict(tests)
        )
        rows.append(
            f'<tr><th scope="row"><svg viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="{escape(name)}: 400 points"><g class="dp-dots">{dots}</g></svg>'
            f'<span class="dp-name">{escape(name)}</span></th>{cells}</tr>'
        )
    head = "".join(f'<th scope="col">{label}</th>' for _, label in tests)
    every = [label for t, label in tests[1:] if all(r[t] < 0.05 for r in results)]
    return (
        '<figure class="dp"><table class="dp-table"><thead><tr><td></td>'
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "<figcaption>Three ways y depends on x, each with the same noise; ✓ finds the dependence "
        "at alpha 0.05. Partial correlation measures only a straight-line trend, so it misses "
        f"the cos and the square; {listed(every)} find all three.</figcaption></figure>\n"
    )


def graphs_figure() -> str:
    # The post's example: 500 rows of 8 variables, each a weighted sum of the cos of its causes.
    warnings.simplefilter("ignore", andrey.ExperimentalWarning)
    dataset = sample_scm(
        d=8, n=500, functional="nonmonotone", density=2.0, scale="standardize", seed=0
    )
    labels = tuple("ABCDEFGH")
    truth = dataset.graph.adjacency
    true_graph = GraphStructure.from_numpy(truth, kind="dag", labels=labels)
    # The nodes on a circle, so no edge runs through another node at this size.
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(labels), endpoint=False)
    positions = {
        i: (0.5 + 0.42 * float(np.cos(a)), 0.5 - 0.42 * float(np.sin(a)))
        for i, a in enumerate(angles)
    }
    edges = truth != 0
    edges |= edges.T

    def picture(graph: GraphStructure) -> str:
        drawn = draw(graph, palette=PALETTE, positions=positions, background=False,
                     width=200, height=200)  # fmt: skip
        return str(drawn).split("?>", 1)[-1].strip()

    cells = [("True graph", "8 links", picture(true_graph))]
    for test, label in (("fisherz", "Partial correlation"), *TESTS.items()):
        fit = andrey.pc(dataset.data, alpha=0.05, indep_test=test).structure
        found = fit.adjacency != 0
        found |= found.T
        hits, extra = (found & edges).sum() // 2, (found & ~edges).sum() // 2
        graph = GraphStructure.from_numpy(fit.adjacency, kind="cpdag", labels=labels)
        cells.append((label, f"{hits} of 8 found, {extra} extra", picture(graph)))
    body = "".join(
        f'<div class="gf-cell"><p class="gf-name">{escape(name)}</p>{svg}'
        f'<p class="gf-count">{escape(count)}</p></div>'
        for name, count, svg in cells
    )
    return (
        f'<figure class="gf"><div class="gf-grid">{body}</div><figcaption>The example\'s true '
        "graph, and the graph PC finds on its 500 rows with each test.</figcaption></figure>\n"
    )


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
    (HERE / "dependence.html").write_text(dependence_figure(), encoding="utf-8")
    (HERE / "graphs.html").write_text(graphs_figure(), encoding="utf-8")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SUMMARY
    if path.exists():  # the release card's summary lands with the card
        summary = json.loads(path.read_text(encoding="utf-8"))
        (HERE / "accuracy.html").write_text(accuracy_table(summary), encoding="utf-8")
        (HERE / "speed.html").write_text(speed_table(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
