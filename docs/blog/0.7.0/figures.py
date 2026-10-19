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
    start = states["".join(LABELS)]
    first = int(start.split("|")[0])

    def lane(steps: list[tuple[str, bool]]) -> str:
        return "".join(
            f'<li data-collider="{key}" class="{"" if oriented else "is-skipped"}">'
            f'<span class="co-triple">{draw_collider(key)}</span>'
            f'<span class="co-p">p = {pvalue[key]:.3f}</span>'
            f'<span class="co-mark">{"oriented" if oriented else "skipped"}</span></li>'
            for key, oriented in steps
        )

    old_steps = [(c[:-1], c[-1] == "+") for c in start.split("|")[1].split(",")]

    def wrong(g: np.ndarray) -> list[tuple[int, int]]:
        """The arrowheads of ``g`` that the true graph does not have."""
        return [tuple(map(int, p)) for p in np.argwhere((g == ARROW) & (cpdag != ARROW))]

    variants = "".join(
        f'<div class="co-graph" data-graph="{k}" data-count="{counts[k]}"'
        f"{'' if k == first else ' hidden'}>{draw_one(g, 'cpdag', wrong(g))}</div>"
        for k, g in enumerate(graphs)
    )
    chips = "".join(f'<li data-column="{c}">{c}</li>' for c in LABELS)
    caption = "PC on the same 300 rows. Rings mark arrowheads the true graph does not have."
    states_json = escape(json.dumps(states, separators=(",", ":")))
    return (
        f'<figure class="co" data-states="{states_json}">'
        '<div class="co-bar"><p class="co-label">Column order</p>'
        f'<ol class="co-columns" aria-label="Column order">{chips}</ol>'
        '<div class="co-actions"><button type="button" class="co-shuffle" hidden>'
        'Shuffle the columns</button><button type="button" class="co-reset" hidden>'
        "Causal order</button></div></div>"
        '<div class="co-lanes">'
        '<div class="co-lane co-old"><p class="co-title">0.6.0: column-order rule</p>'
        '<p class="co-rule">Visits each collider by its middle column, left to right. The '
        "first to claim an edge keeps it.</p>"
        f'<ol class="co-queue">{lane(old_steps)}</ol>{variants}'
        '<p class="co-note" aria-live="polite">This graph comes from <b class="co-count">'
        f'{counts[first]}</b> of 120 orders. Graphs seen: <b class="co-seen">1</b>.</p></div>'
        '<div class="co-lane co-new"><p class="co-title">0.7.0: p-value rule</p>'
        '<p class="co-rule">Visits the colliders whose pair was most clearly separated, the '
        "highest p-value, first. Column order plays no part.</p>"
        f'<ol class="co-queue">{lane(ranked)}</ol>'
        f'<div class="co-graph">{draw_one(cpdag, "cpdag", [])}</div>'
        '<p class="co-note">The true graph, from all 120 orders.</p></div>'
        '<div class="co-lane co-truth"><p class="co-title">How the data were made</p>'
        f'<div class="co-graph">{draw_one(truth, "dag", [])}</div>'
        '<p class="co-note">A causes B, C, and D, and each of them causes E. 300 rows.</p></div>'
        f"</div><figcaption>{escape(caption)}</figcaption></figure>\n"
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
        '<div class="cg-panel cg-chart"><p class="co-title">New PC rule minus old, errors</p>'
        f'{bars}<p class="cg-axis"><span>fewer errors</span><span>more errors</span></p></div>'
        f"</div><figcaption>{escape(caption)}</figcaption></figure>\n"
    )


def reorder_chart() -> str:
    # Datasets where a reordering of the columns changed the graph: (method, denominator,
    # [(series, count)]), the counts the post's tables report.
    groups = (
        ("Hill climbing, BDeu", 150, (("old", 145), ("new", 0))),
        ("Hill climbing, BIC", 150, (("old", 147), ("new", 0))),
        ("Tabu search, BDeu", 150, (("old", 112), ("new", 0))),
        ("Tabu search, BIC", 150, (("old", 118), ("new", 0))),
        ("BOSS", 150, (("cl", 137), ("old", 146), ("new", 0))),
        ("GRaSP", 150, (("cl", 150), ("old", 149), ("new", 0))),
        ("GIN", 15, (("first", 11), ("new", 0))),
    )
    names = {
        "cl": "causal-learn",
        "old": "Andrey 0.6.0",
        "first": "first-column root test",
        "new": "Andrey 0.7.0",
    }
    rows = "".join(
        f'<div class="rc-group"><p class="rc-method">{method}</p>'
        + "".join(
            f'<div class="rc-row rc-{key}"><span class="rc-series">{names[key]}</span>'
            f'<span class="rc-track"><span class="rc-bar" '
            f'style="--rc-w: {count / total * 100:.1f}%"></span></span>'
            f'<span class="rc-count"><b>{count}</b> of {total}</span></div>'
            for key, count in series
        )
        + "</div>"
        for method, total, series in groups
    )
    return (
        '<figure class="rc"><p class="co-title">Datasets where reordering the columns changed the '
        f'graph</p><div class="rc-groups">{rows}</div><figcaption>Each dataset fit in its own '
        "column order and five reorderings, ten for GIN.</figcaption></figure>\n"
    )


def main() -> None:
    (HERE / "order.html").write_text(order_figure(), encoding="utf-8")
    (HERE / "generator.html").write_text(generator_figure(), encoding="utf-8")
    (HERE / "reorder.html").write_text(reorder_chart(), encoding="utf-8")


if __name__ == "__main__":
    main()
