"""The 0.7.0 post's figures, written next to it as HTML that the post includes.

- ``order.html``: PC on five variables under all 120 orders of the same columns, with the
  column-order rule and with 0.7.0's default. It holds the true graph, each graph the column-order
  rule returns, and the one graph the default returns, drawn by ``andrey.viz.draw`` over the same
  node positions, and a map from each order to its graph. ``docs/_static/column-order.js``
  shuffles the columns in the post.
- ``generator.html``: a scale-free graph numbered by column, as the data generators wrote it before
  0.7.0 and as they shuffle it now, beside how that order moved PC's comparison.
- ``reorder.html``: the post's counts of datasets where a reordering changed the graph.

The figures are committed, so the build shows the fits of the machine that ran this script.
Every color is an ``--andrey-*`` token. Run with ``uv run python docs/blog/0.7.0/figures.py``.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from html import escape
from pathlib import Path

import numpy as np

import andrey
from andrey import GraphStructure
from andrey.constraint.pc import pc as pc_trace
from andrey.core import ARROW
from andrey.data import sample_scm
from andrey.viz import LIGHT, Palette, draw, layout

HERE = Path(__file__).parent
LABELS = ("A", "B", "C", "D", "E")
# The drawn columns, in causal order: A causes B, C, and D, which cause E.
CAUSAL = [4, 0, 1, 3, 2]
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


def fit(data: np.ndarray, order: tuple[int, ...], rule: str) -> np.ndarray:
    """PC's endpoint matrix for the columns in ``order``, put back in causal order."""
    columns = np.array(order)
    back = np.argsort(columns)
    adjacency = andrey.pc(data[:, columns], alpha=0.05, collider_order=rule).structure.adjacency
    return adjacency[np.ix_(back, back)]


def svg(graph: GraphStructure, positions: dict, highlight: list, size: tuple[int, int]) -> str:
    """One graph over fixed node positions, without the XML declaration."""
    width, height = size
    drawn = draw(
        graph,
        palette=PALETTE,
        positions=positions,
        background=False,
        highlight=highlight,
        width=width,
        height=height,
    )
    return str(drawn).split("?>", 1)[-1].strip()


def collider_steps(data: np.ndarray, order: tuple[int, ...], rule: str) -> list[tuple[str, bool]]:
    """The colliders PC proposes for the columns in ``order``, in the order it decides them: each
    as ``"x y z"`` in causal-order labels, ends sorted, and whether PC oriented it."""
    trace: list[dict] = []
    pc_trace(data[:, np.array(order)], alpha=0.05, collider_order=rule, _trace=trace)
    steps = []
    for step in trace:
        if step["step"] == "collider" and step["triple"][1] not in step["sepset"]:
            x, y, z = (LABELS[order[k]] for k in step["triple"])
            steps.append((" ".join([min(x, z), y, max(x, z)]), bool(step["oriented"])))
    return steps


def order_figure() -> str:
    dataset = sample_scm(d=5, n=300, density=2.5, scale="standardize", seed=13)
    data = dataset.data[:, CAUSAL]
    truth = dataset.graph.adjacency[np.ix_(CAUSAL, CAUSAL)]
    positions = layout(GraphStructure.from_numpy(truth, kind="dag", labels=LABELS), "builtin")

    def draw_one(matrix: np.ndarray, kind: str, highlight: list) -> str:
        graph = GraphStructure.from_numpy(matrix, kind=kind, labels=LABELS)
        return svg(graph, positions, highlight, (240, 130))

    def draw_collider(key: str) -> str:
        """A collider x -> y <- z as a line of text, its middle node marked."""
        x, y, z = key.split()
        return (
            f'<span class="co-node">{x}</span><span class="co-arrow" aria-label="causes">'
            f'-&gt;</span><span class="co-node co-middle">{y}</span><span class="co-arrow" '
            f'aria-label="is caused by">&lt;-</span><span class="co-node">{z}</span>'
        )

    orders = list(itertools.permutations(range(5)))
    default = {fit(data, order, "p_value").tobytes() for order in orders}
    assert len(default) == 1, "the default rule gives one graph"
    cpdag = np.frombuffer(default.pop(), dtype=truth.dtype).reshape(5, 5)
    ranked = collider_steps(data, tuple(range(5)), "p_value")
    # Colliders of equal p-value keep their column order, but every order orients the same ones.
    assert all(sorted(collider_steps(data, o, "p_value")) == sorted(ranked) for o in orders)
    trace: list[dict] = []
    pc_trace(data, alpha=0.05, collider_order="p_value", _trace=trace)
    pvalue = {}
    for step in trace:
        if step["step"] == "collider":
            x, y, z = (LABELS[k] for k in step["triple"])
            pvalue[" ".join([min(x, z), y, max(x, z)])] = step["p"]

    graphs: list[np.ndarray] = []
    states: dict[str, str] = {}
    for order in orders:
        matrix = fit(data, order, "column")
        index = next((k for k, g in enumerate(graphs) if np.array_equal(g, matrix)), None)
        if index is None:
            index, graphs = len(graphs), [*graphs, matrix]
        steps = collider_steps(data, order, "column")
        # The graph's index, then each collider as its key and "+" (oriented) or "-" (skipped).
        states["".join(LABELS[k] for k in order)] = f"{index}|" + ",".join(
            f"{key}{'+' if oriented else '-'}" for key, oriented in steps
        )
    counts = [sum(v.startswith(f"{k}|") for v in states.values()) for k in range(len(graphs))]
    right = sum(c for k, c in enumerate(counts) if np.array_equal(graphs[k], cpdag))
    print(f"order.html: {len(graphs)} graphs from 120 orders, the true one from {right}")
    causal = "".join(LABELS)
    first = int(states[causal].split("|")[0])

    def is_true(key: str) -> bool:
        x, y, z = (LABELS.index(c) for c in key.split())
        return truth[y, x] == ARROW and truth[y, z] == ARROW

    def verdicts(steps: list[tuple[str, bool]]) -> list[tuple[str, bool, list[str]]]:
        """Each collider, whether it was oriented, and the edges an earlier one had claimed."""
        heads: set[tuple[str, str]] = set()
        out = []
        for key, oriented in steps:
            x, y, z = key.split()
            taken = ["-".join(sorted((a, y))) for a in (x, z) if (y, a) in heads]
            if oriented:
                heads.update({(x, y), (z, y)})
            out.append((key, oriented, taken))
        return out

    def card(key: str, step: int, oriented: bool, taken: list[str], bar: str = "") -> str:
        x, y, z = key.split()
        verdict = "orients" if oriented else f"skipped: <code>{', '.join(taken)}</code> taken"
        return (
            f'<li class="co-card{"" if is_true(key) else " is-false"}'
            f'{"" if oriented else " is-skipped"}" data-collider="{key}">'
            f'<span class="co-step">{step}</span><span class="co-triple">'
            f'<span class="co-node">{x}</span><span class="co-arrow" aria-label="causes">-&gt;'
            f'</span><span class="co-node co-middle">{y}</span><span class="co-arrow" '
            f'aria-label="is caused by">&lt;-</span><span class="co-node">{z}</span></span>{bar}'
            f'<span class="co-p">p = {pvalue[key]:.3f}</span>'
            f'<span class="co-verdict">{verdict}</span></li>'
        )

    # The column-order rule's board: a column per data column, each collider under its middle.
    old = verdicts([(c[:-1], c[-1] == "+") for c in states[causal].split("|")[1].split(",")])
    step_of = {key: n for n, (key, _, _) in enumerate(old, 1)}
    board = "".join(
        f'<li class="co-col" data-column="{c}"><button type="button" class="co-chip" '
        f'aria-label="Move column {c} first">{c}</button><ol class="co-cards">'
        + "".join(card(k, step_of[k], o, t) for k, o, t in old if k.split()[1] == c)
        + "</ol></li>"
        for c in LABELS
    )

    def why(order: str) -> str:
        """Why an order gives its graph: which of C and D sit left of E."""
        c, d = (order.index(v) < order.index("E") for v in "CD")
        if c and d:
            return "C and D sit left of E, so their false colliders take C-E and D-E first."
        if c:
            return "C sits left of E, so its false collider takes C-E first. D-E is right."
        if d:
            return "D sits left of E, so its false collider takes D-E first. C-E is right."
        return "E sits left of C and D, so its true colliders take both edges first."

    def wrong(g: np.ndarray) -> list[tuple[int, int]]:
        """The arrowheads of ``g`` that the true graph does not have."""
        return [tuple(map(int, p)) for p in np.argwhere((g == ARROW) & (cpdag != ARROW))]

    example = {int(v.split("|")[0]): order for order, v in states.items()}
    variants = "".join(
        f'<div class="co-graph" data-graph="{k}" data-count="{counts[k]}" '
        f'data-why="{escape(why(example[k]))}"{"" if k == first else " hidden"}>'
        f"{draw_one(g, 'cpdag', wrong(g))}</div>"
        for k, g in enumerate(graphs)
    )
    # The p-value rule's list, with a bar as long as each p-value and the significance level.
    rank = "".join(
        card(
            k,
            n,
            o,
            t,
            f'<span class="co-bar"><span class="co-track"><span class="co-fill" '
            f'style="--w: {pvalue[k] * 100:.1f}%"></span></span></span>',
        )
        for n, (k, o, t) in enumerate(verdicts(ranked), 1)
    )
    axis = (
        '<li class="co-card co-axis-row" aria-hidden="true"><span class="co-step"></span>'
        '<span class="co-triple"></span><span class="co-axis"><span class="at-alpha">0.05</span>'
        '<span class="at-one">p = 1</span></span><span class="co-p"></span>'
        '<span class="co-verdict"></span></li>'
    )
    caption = "PC on the same 300 rows. A dashed card is a collider the true graph does not have."
    states_json = escape(json.dumps(states, separators=(",", ":")))
    return (
        f'<figure class="co" data-states="{states_json}">'
        '<div class="co-setup"><div class="co-graph">'
        f'{draw_one(truth, "dag", [])}</div><p class="co-note"><b>The data.</b> A causes B, '
        "C, and D, and each of them causes E. PC proposes five colliders; two are false.</p></div>"
        '<section class="co-lane co-old"><div class="co-head">'
        '<p class="co-title">0.6.0: column-order rule</p><div class="co-actions">'
        '<button type="button" class="co-shuffle" hidden>Shuffle the columns</button>'
        '<button type="button" class="co-reset" hidden>Original order</button></div></div>'
        '<p class="co-rule">Visits the colliders <b>column by column</b>, left to right: each '
        'waits under the column it points into. <span class="co-hint" hidden>Click a column to '
        "move it first.</span></p>"
        f'<div class="co-body"><ol class="co-board" aria-label="Column order">{board}</ol>'
        f'<div class="co-side">{variants}'
        f'<p class="co-note co-why" aria-live="polite">{escape(why(causal))}</p>'
        '<p class="co-note">This graph comes from <b class="co-count">'
        f'{counts[first]}</b> of 120 orders. Graphs seen: <b class="co-seen">1</b> of '
        f"{len(graphs)}.</p></div></div></section>"
        '<section class="co-lane co-new"><div class="co-head">'
        '<p class="co-title">0.7.0: p-value rule</p></div>'
        '<p class="co-rule">Visits the colliders <b>by p-value, highest first</b>: the most '
        "clearly separated pair goes first, wherever its columns sit.</p>"
        f'<div class="co-body"><ol class="co-rank" aria-label="Colliders by p-value">{rank}'
        f'{axis}</ol><div class="co-side"><div class="co-graph">'
        f"{draw_one(cpdag, 'cpdag', [])}</div>"
        '<p class="co-note">The true graph, from all 120 orders.</p></div></div></section>'
        f"<figcaption>{escape(caption)}</figcaption></figure>\n"
    )


def generator_figure() -> str:
    dataset = sample_scm(graph="scale_free", d=10, n=10, density=2.0, seed=1, shuffle=False)
    truth = dataset.graph.adjacency
    degree = (truth != 0).sum(axis=1)
    assert degree[0] == degree.max(), "the generator writes the biggest hub first"
    shuffled = np.random.default_rng(7).permutation(10)

    def draw_numbered(columns: np.ndarray) -> str:
        labels = tuple(str(int(c) + 1) for c in columns)
        graph = GraphStructure.from_numpy(truth, kind="dag", labels=labels)
        return svg(graph, positions, [], (300, 210))

    positions = layout(GraphStructure.from_numpy(truth, kind="dag"), "builtin")
    # How much the p-value rule's mean endpoint errors differ from the column-order rule's on
    # scale-free graphs, at the two significance levels: negative is fewer errors.
    rows = (("Columns as generated", (1.0, 4.8)), ("Columns shuffled", (-1.0, -3.1)))
    scale = 6.5
    bars = "".join(
        f'<div class="cg-row"><p class="cg-name">{name}</p><div class="cg-pair">'
        + "".join(
            f'<div class="cg-track"><span class="cg-bar {"is-worse" if v > 0 else "is-better"}" '
            f'style="--cg-w: {abs(v) / scale * 50:.1f}%"></span>'
            f'<b class="cg-v">{v:+.1f}</b></div>'
            for v in values
        )
        + "</div></div>"
        for name, values in rows
    )
    caption = "Each node is numbered by its column."
    return (
        '<figure class="cg"><div class="cg-panels">'
        '<div class="cg-panel"><p class="co-title">Before 0.7.0: column 1 is the hub</p>'
        f"{draw_numbered(np.arange(10))}</div>"
        '<div class="cg-panel"><p class="co-title">0.7.0: columns shuffled</p>'
        f"{draw_numbered(shuffled)}</div>"
        '<div class="cg-panel cg-chart"><p class="co-title">p-value rule minus column-order rule, '
        "endpoint errors</p>"
        f'{bars}<p class="cg-axis"><span>fewer errors</span><span>more errors</span></p></div>'
        f"</div><figcaption>{escape(caption)}</figcaption></figure>\n"
    )


def table(title: str, columns: tuple[str, ...], groups: tuple, rank=None, note: str = "") -> str:
    """A comparison table in the post's one layout: rows grouped under a label set on its side,
    Andrey's column first and tinted, each row's best cell in bold.

    ``groups`` holds ``(label, rows)``, each row ``(name, detail, cells)`` with a cell per column:
    a number, a word, or None for a method a package does not have. ``rank`` orders a row's cells,
    lowest best; it defaults to the number itself.
    """
    rank = rank or (lambda cell: cell)
    ours = ' class="ct-ours"'
    head = "".join(
        f'<th scope="col"{ours if k == 0 else ""}>{escape(c)}</th>' for k, c in enumerate(columns)
    )

    def row(name: str, detail: str, cells: tuple) -> str:
        ranks = [rank(c) for c in cells if c is not None]
        best = min(ranks) if ranks else None
        tds = []
        for k, c in enumerate(cells):
            cls = ["ct-ours"] if k == 0 else []
            if c is None:
                tds.append(f'<td class="{" ".join([*cls, "ct-na"])}">-</td>')
                continue
            if rank(c) == best and len(set(ranks)) > 1:
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


def reorder_chart() -> str:
    # Datasets where a reordering of the columns changed the graph: the counts the post reports.
    return table(
        "Datasets where reordering the columns changed the graph (lower is better)",
        ("Andrey 0.7.0", "Andrey 0.6.0", "causal-learn"),
        (
            (
                "Score-based",
                (
                    ("Hill climbing, BDeu", "150 datasets", (0, 145, None)),
                    ("Hill climbing, BIC", "150 datasets", (0, 147, None)),
                    ("Tabu search, BDeu", "150 datasets", (0, 112, None)),
                    ("Tabu search, BIC", "150 datasets", (0, 118, None)),
                ),
            ),
            (
                "Permutation",
                (
                    ("BOSS", "150 datasets", (0, 146, 137)),
                    ("GRaSP", "150 datasets", (0, 149, 150)),
                ),
            ),
            ("Hidden causes", (("GIN", "15 datasets", (0, 11, None)),)),
        ),
    )


def compare_table() -> str:
    # Whether a reordering of the columns changed each package's graph, at its defaults: three
    # small datasets per method, each in nine column orders. "By name": the graph follows the
    # variables' names; "By run": it varies between runs, not with the order.
    order = {"Same": 0, "By name": 1, "By run": 1, "Changed": 2, "Changed*": 2}
    same, changed, name = "Same", "Changed", "By name"
    return table(
        "Does reordering the columns change the graph?",
        (
            "Andrey 0.7.0",
            "causal-learn 0.1.4.8",
            "pcalg 2.7-12",
            "bnlearn 5.2.1",
            "pgmpy 1.1.2",
            "Tetrad 7.6.10",
        ),
        (
            (
                "Constraint",
                (
                    ("PC", "", (same, changed, changed, changed, name, name)),
                    ("FCI", "", (same, changed, changed, None, None, name)),
                ),
            ),
            (
                "Score-based",
                (
                    ("Hill climbing", "", (same, None, None, changed, "By run", None)),
                    ("Tabu search", "", (same, None, None, changed, None, None)),
                ),
            ),
            (
                "Permutation",
                (
                    ("BOSS", "", (same, changed, None, None, None, changed)),
                    ("GRaSP", "", (same, changed, None, None, None, changed)),
                ),
            ),
            ("Hidden causes", (("GIN", "", (same, "Changed*", None, None, None, changed)),)),
        ),
        rank=order.get,
        note=(
            "Each package at its defaults, on three small datasets per method, each in nine "
            "column orders; causal-learn's and Tetrad's BOSS and GRaSP ran without their own "
            "randomness. By name: the graph follows the variables' names, so it can change when "
            "the names follow the positions, as a bare array's do. By run: pgmpy's hill climbing "
            "breaks ties in Python's hash order, which changes between runs. * causal-learn's GIN "
            "also tests columns by position, a known bug (py-why/causal-learn#271)."
        ),
    )


def fci_tables() -> str:
    # FCI on six real tables: the orders, of 200 (20 for the full superconductivity table), that
    # changed the result, with the default and with possible_dsep="fast".
    pending = "[PENDING]"
    return table(
        "Column orders that changed FCI's result (lower is better)",
        ("Default", 'possible_dsep="fast"'),
        (
            (
                "Real tables",
                (
                    ("Sachs protein signaling", "11 variables, 853 rows, 200 orders", (pending, 0)),
                    ("Wine", "13 variables, 178 rows, 200 orders", (pending, 0)),
                    ("Diabetes", "10 variables, 442 rows, 200 orders", (pending, 150)),
                    ("Breast cancer", "30 variables, 569 rows, 200 orders", (pending, 156)),
                    (
                        "Superconductivity, 1,000-row sample",
                        "81 variables, 1,000 rows, 200 orders",
                        (pending, 129),
                    ),
                    ("Superconductivity", "81 variables, 21,263 rows, 20 orders", (pending, 20)),
                ),
            ),
        ),
        rank=lambda c: c if isinstance(c, int) else -1,
    )


def main() -> None:
    (HERE / "order.html").write_text(order_figure(), encoding="utf-8")
    (HERE / "generator.html").write_text(generator_figure(), encoding="utf-8")
    (HERE / "reorder.html").write_text(reorder_chart(), encoding="utf-8")
    (HERE / "compare.html").write_text(compare_table(), encoding="utf-8")
    (HERE / "fci.html").write_text(fci_tables(), encoding="utf-8")


if __name__ == "__main__":
    main()
