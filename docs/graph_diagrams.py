"""Build graph-guide illustrations from the public structure and SVG APIs.

Run by Sphinx before reading the guides, or directly with the docs environment.
Generated HTML uses the site's palette tokens and works without JavaScript; the comparison
controls progressively select a single example and highlight corresponding matrix cells.
"""

from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path

import numpy as np

from andrey import GraphStructure, TemporalStructure
from andrey.core import ARROW, CIRCLE, TAIL
from andrey.viz import LIGHT, Palette, draw

STATIC = Path(__file__).parent / "_static"
MARKS = {0: "NULL", TAIL: "TAIL", ARROW: "ARROW", CIRCLE: "CIRCLE"}
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


def graph(matrix: list[list[int]], kind: str, labels: tuple[str, ...]) -> GraphStructure:
    """Construct and validate the exact graph used by a diagram and its matrix."""
    result = GraphStructure.from_numpy(np.array(matrix, dtype=np.int8), kind=kind, labels=labels)
    result.validate()
    return result


def svg(g: GraphStructure, description: str, *, width: int = 340, height: int = 180) -> str:
    """Render a fluid SVG with an accessible description and theme-aware colors."""
    markup = str(draw(g, palette=PALETTE, width=width, height=height, background=False))
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    root = ET.fromstring(markup)
    root.set("role", "img")
    root.set("aria-label", description)
    return ET.tostring(root, encoding="unicode")


def matrix(values: np.ndarray, labels: tuple[str, ...], caption: str) -> str:
    """Render a labeled matrix; coordinates link cells across the comparison tables."""
    cells = [f'<table class="graph-matrix"><caption>{escape(caption)}</caption>']
    cells.append("<thead><tr><td></td>")
    cells.extend(f'<th scope="col">{escape(label)}</th>' for label in labels)
    cells.append("</tr></thead><tbody>")
    for i, row in enumerate(values):
        cells.append(f'<tr><th scope="row">{escape(labels[i])}</th>')
        for j, value in enumerate(row):
            label = f"row {labels[i]}, column {labels[j]}: {value:g}"
            cells.append(
                f'<td data-row="{i}" data-col="{j}" aria-label="{escape(label)}">{value:g}</td>'
            )
        cells.append("</tr>")
    cells.append("</tbody></table>")
    return "".join(cells)


def figure(g: GraphStructure, title: str, description: str) -> str:
    """Pair a causal graph with its actual dense endpoint matrix."""
    return (
        '<figure class="graph-figure">'
        f"<figcaption><strong>{escape(title)}</strong> {escape(description)}</figcaption>"
        '<div class="graph-figure-body">'
        f"{svg(g, description)}{matrix(g.to_numpy(), g.labels, 'Endpoint matrix M')}"
        "</div></figure>\n"
    )


def comparison() -> str:
    """Compare four matrix conventions on the same two-node graph."""
    cases = [
        ("directed", "X -> Y", TAIL, ARROW, "dag", "Tail at X, arrowhead at Y."),
        ("reversed", "X <- Y", ARROW, TAIL, "dag", "Arrowhead at X, tail at Y."),
        ("undirected", "X -- Y", TAIL, TAIL, "cpdag", "Direction is unresolved in this CPDAG."),
        ("bidirected", "X <-> Y", ARROW, ARROW, "pag", "Two arrowheads in a PAG; not feedback."),
        ("partial", "X o-> Y", CIRCLE, ARROW, "pag", "The endpoint at X is unresolved."),
        ("circles", "X o-o Y", CIRCLE, CIRCLE, "pag", "Both endpoints are unresolved."),
    ]
    panels = ['<div class="graph-explorer" data-graph-explorer>']
    panels.append('<fieldset class="graph-controls" hidden><legend>Choose an edge</legend>')
    for index, (key, label, *_, description) in enumerate(cases):
        checked = " checked" if index == 0 else ""
        panels.append(
            f'<label><input type="radio" name="graph-edge" value="{key}"{checked} '
            f'aria-label="{escape(label + ": " + description)}">'
            f"<span>{escape(label)}</span></label>"
        )
    panels.append("</fieldset>")
    for key, label, left, right, kind, description in cases:
        g = graph([[0, left], [right, 0]], kind, ("X", "Y"))
        marks = g.to_numpy()
        signed = np.array([0, -1, 1, 2])[marks]
        panels.append(
            f'<section class="graph-example" data-example="{key}" aria-label="{escape(label)}">'
            '<div class="graph-pair">'
            f"{svg(g, label, width=280, height=150)}<div>"
            f'<p class="graph-kind">kind="{kind}"</p><p><strong>{escape(description)}</strong></p>'
            f"<p><code>M[X, Y] = {left}</code> ({MARKS[left]} at X)<br>"
            f"<code>M[Y, X] = {right}</code> ({MARKS[right]} at Y)</p></div></div>"
            '<div class="graph-encodings">'
        )
        entries = [
            (
                "Andrey",
                marks,
                "M: endpoint at the row node",
                "0 absent / 1 tail / 2 arrow / 3 circle",
            ),
            (
                "causal-learn",
                signed,
                "G.graph: endpoint at the row node",
                "0 absent / -1 tail / 1 arrow / 2 circle",
            ),
        ]
        if kind == "dag":
            binary = ((marks == TAIL) & (marks.T == ARROW)).astype(int)
            entries.extend(
                [
                    (
                        "gCastle DAG",
                        binary,
                        "A: row is the source",
                        "1 records a directed edge; no coefficient.",
                    ),
                    (
                        "LiNGAM",
                        0.8 * binary.T,
                        "B: row is the effect",
                        "Illustrative coefficient: 0.8. Andrey weights use B.T.",
                    ),
                ]
            )
        else:
            entries.extend(
                [
                    (
                        "gCastle DAG",
                        None,
                        "Binary directed adjacency",
                        "Needs an extra convention to retain these endpoint marks.",
                    ),
                    (
                        "LiNGAM",
                        None,
                        "Directed coefficients",
                        "A coefficient alone does not encode this endpoint uncertainty.",
                    ),
                ]
            )
        for name, values, caption, note in entries:
            panels.append(f'<div class="graph-encoding"><h3>{name}</h3>')
            if values is None:
                panels.append(f'<p class="graph-unavailable">{caption}</p>')
            else:
                panels.append(matrix(values, ("X", "Y"), caption))
            panels.append(f'<p class="graph-note">{note}</p></div>')
        panels.append("</div></section>")
    panels.append(
        '<p class="graph-help" hidden>Focus or hover over a matrix cell to follow the same node '
        "pair across representations. Matrix rows and columns are labeled X and Y.</p></div>\n"
    )
    return "".join(panels)


def write_diagrams() -> None:
    """Write only changed fragments so live documentation builds settle."""
    dag = graph([[0, 1, 0], [2, 0, 1], [0, 2, 0]], "dag", ("X", "Y", "Z"))
    cpdag = graph([[0, 1, 0], [1, 0, 1], [0, 1, 0]], "cpdag", ("X", "Y", "Z"))
    pag = graph([[0, 3, 0], [2, 0, 2], [0, 3, 0]], "pag", ("X", "Y", "Z"))
    pair = [[0, 2], [2, 0]]
    lag1 = graph([[0, 1], [2, 0]], "dag", ("X", "Y"))
    lag2 = graph([[0, 2], [1, 0]], "dag", ("X", "Y"))
    temporal = TemporalStructure.from_lag_graphs([lag1, lag2], lags=[1, 2], labels=("X", "Y"))
    fragments = {
        "graph-comparison.html": comparison(),
        "graph-types-dag.html": figure(dag, "DAG", "A directed chain: X causes Y, which causes Z."),
        "graph-types-cpdag.html": figure(cpdag, "CPDAG", "The chain's directions are unresolved."),
        "graph-types-pag.html": figure(
            pag, "PAG", "X o-> Y <-o Z: arrowheads at Y, circles at X and Z."
        ),
        "graph-types-feedback.html": (
            figure(graph(pair, "pag", ("X", "Y")), "PAG edge fragment", "One bidirected edge.")
            + figure(
                graph(pair, "digraph", ("X", "Y")),
                "Directed graph",
                "Two directed edges forming a cycle.",
            )
        ),
        "graph-types-temporal.html": (
            figure(lag1, "Lag 1", "X(t-1) -> Y(t). Each edge still uses two endpoint marks.")
            + figure(lag2, "Lag 2", "Y(t-2) -> X(t). The reverse cell completes the mark pair.")
            + figure(
                temporal.summary_graph(),
                "Summary",
                "Time is collapsed; both directed edges remain.",
            )
        ),
    }
    for name, content in fragments.items():
        path = STATIC / name
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    write_diagrams()
