"""Compare Andrey and causal-learn on one simulated dataset, edge by edge.

``compare_page()`` returns a standalone page with the true graph, Andrey's graph, and causal-learn's
graph for PC and GES, every edge colored by how it compares with the truth. Andrey runs when the
page is generated; causal-learn's graphs come from ``compare-causal-learn.json``, recorded once with
``python apps/compare.py --record``, so the docs build does not need causal-learn.
"""

# No `from __future__ import annotations`: string annotations make the dataclass below look its
# module up in sys.modules, which fails when a build loads this file by path.
import argparse
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import re
import sys
from html import escape
from pathlib import Path

import numpy as np

import andrey
import andrey.viz
from andrey import GraphStructure
from andrey.core.structure import ARROW, CIRCLE, TAIL
from andrey.data import SCM, functional, noise
from andrey.data.graphs import DAGDraw, GraphSpec, dag_truth
from andrey.metrics import score, to_cpdag
from andrey.viz import _svg
from andrey.viz.theme import DARK, LIGHT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "live-discovery"))

from learn import FRAME_SCRIPT, TOKEN_PALETTE, palette_css  # noqa: E402
from runners import signed_marks  # noqa: E402

RECORDING = Path(__file__).with_name("compare-causal-learn.json")
CAUSAL_LEARN = "0.1.4.8"

# Node order is topological and sets each layout column's top-to-bottom order.
LABELS = (
    "sunlight",
    "rain",
    "irrigation",
    "fertilizer",
    "temperature",
    "nitrogen",
    "pests",
    "moisture",
    "growth",
    "yield",
)
# Edge order fixes which weight each edge draws from the seed.
EDGES = (
    ("rain", "moisture"),
    ("irrigation", "moisture"),
    ("sunlight", "temperature"),
    ("temperature", "moisture"),
    ("fertilizer", "nitrogen"),
    ("moisture", "growth"),
    ("nitrogen", "growth"),
    ("sunlight", "growth"),
    ("temperature", "pests"),
    ("pests", "yield"),
    ("growth", "yield"),
)
N_SAMPLES = 1_000
SEED = 4
ALPHA = 0.05
METHODS = ("PC", "GES")
CAUSAL_LEARN_PARAMETERS = {
    "PC": {"alpha": ALPHA, "indep_test": "fisherz", "stable": True},
    # Andrey's GES caps parents at half the variables, and causal-learn's lambda 0.5 is the same
    # BIC penalty as Andrey's default 1.0.
    "GES": {"score_func": "local_score_BIC", "maxP": len(LABELS) // 2, "lambda_value": 0.5},
}

STATUSES = {
    "correct": "Correct",
    "missing": "Missing",
    "extra": "Extra",
    "wrong": "Wrong direction",
}
METRICS = {"shd": "SHD", "skeleton_f1": "Skeleton F1", "arrow_f1": "Arrow F1"}

# Drawing units: a panel renders near this width, so these sizes are close to screen pixels.
# The left pad fits the widest first-column name; the last column holds a short one.
WIDTH, HEIGHT = 460, 345
PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 48, 28, 22, 44
FONT = 16


@dataclasses.dataclass(frozen=True, kw_only=True)
class FixedDAG(GraphSpec):
    """A hand-written DAG listed in topological order, so sampling draws no graph."""

    name: str = "fixed"
    d: int
    edges: tuple[tuple[int, int], ...]

    @property
    def density(self):
        return 2 * len(self.edges) / self.d

    def materialize(self, rng):
        parents, children = (np.array(side, dtype=np.int64) for side in zip(*self.edges))
        if not (parents < children).all():
            raise ValueError("List every parent before its children.")
        draw = DAGDraw(d=self.d, parents=parents, children=children, topo_order=np.arange(self.d))
        return dag_truth(parents, children, self.d), draw


MODEL = SCM(
    graph=FixedDAG(
        d=len(LABELS), edges=tuple((LABELS.index(a), LABELS.index(b)) for a, b in EDGES)
    ),
    functional=functional.linear(),
    noise=noise.gaussian(),
)


def labelled(marks, kind="cpdag"):
    """Attach the variable names to an endpoint-mark matrix."""
    return GraphStructure.from_numpy(np.asarray(marks, dtype=np.int8), kind=kind, labels=LABELS)


@functools.cache
def dataset():
    """Return the standardized, rounded observations (read-only) and the true DAG."""
    sample = MODEL.sample(n=N_SAMPLES, seed=SEED, scale="standardize")
    # The generator's matrix products differ in the last bit between BLAS kernels; six decimals,
    # as a CSV would store them, give every machine the same bytes.
    data = np.round(sample.data, 6)
    data.flags.writeable = False
    return data, labelled(sample.graph.to_numpy(), kind="dag")


def fingerprint(data):
    """SHA-256 of the exact observation bytes."""
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


@functools.cache
def andrey_graphs():
    """Fit Andrey's PC and GES at their defaults."""
    data, _ = dataset()
    return {
        "PC": labelled(andrey.pc(data, alpha=ALPHA, indep_test="fisherz").structure.to_numpy()),
        "GES": labelled(andrey.ges(data).structure.to_numpy()),
    }


def record(path=RECORDING):
    """Run causal-learn on the dataset and write its graphs with the data's fingerprint."""
    version = importlib.metadata.version("causal-learn")
    if version != CAUSAL_LEARN:
        raise RuntimeError(f"Record with causal-learn {CAUSAL_LEARN}; found {version}.")
    if not hasattr(np, "mat"):
        np.mat = np.asmatrix  # causal-learn still calls np.mat, which NumPy 2 removed
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.ScoreBased.GES import ges

    data, _ = dataset()
    fits = {
        "PC": pc(np.array(data), show_progress=False, **CAUSAL_LEARN_PARAMETERS["PC"]).G,
        "GES": ges(np.array(data), **CAUSAL_LEARN_PARAMETERS["GES"])["G"],
    }
    doc = {
        "causal-learn": version,
        "data": {
            "labels": list(LABELS),
            "shape": list(data.shape),
            "seed": SEED,
            "sha256": fingerprint(data),
        },
        "graph_format": "causal-learn endpoint codes; row i, column j is the mark at variable i "
        "on its edge with j: -1 tail, 1 arrowhead, 2 circle, 0 no edge",
        "methods": {
            method: {
                "parameters": CAUSAL_LEARN_PARAMETERS[method],
                "graph": np.asarray(fit.graph, dtype=int).tolist(),
            }
            for method, fit in fits.items()
        },
    }
    text = json.dumps(doc, indent=2)
    # One matrix row per line keeps the recording short and its diffs readable.
    text = re.sub(
        r"\[\s+(-?\d+(?:,\s+-?\d+)*)\s+\]",
        lambda match: "[" + ", ".join(match.group(1).replace(",", " ").split()) + "]",
        text,
    )
    path.write_text(text + "\n")
    causal_learn_graphs(json.loads(path.read_text()))


def causal_learn_graphs(recording=None):
    """Return causal-learn's recorded graphs; refuse a recording made from other data."""
    if recording is None:
        recording = json.loads(RECORDING.read_text())
    data, _ = dataset()
    expected = {
        "labels": list(LABELS),
        "shape": list(data.shape),
        "seed": SEED,
        "sha256": fingerprint(data),
    }
    stale = [key for key, value in expected.items() if recording["data"].get(key) != value]
    if stale:
        raise ValueError(
            f"{RECORDING.name} does not match the current dataset ({', '.join(stale)} changed). "
            f"Re-record it where causal-learn {CAUSAL_LEARN} is installed: "
            "python apps/compare.py --record"
        )
    return {
        method: labelled(signed_marks(recording["methods"][method]["graph"])) for method in METHODS
    }


def truth_cpdag():
    """The true DAG's CPDAG: what observational data can identify."""
    _, dag = dataset()
    return labelled(to_cpdag(dag).to_numpy())


def edge_statuses(estimate, truth):
    """Classify every pair adjacent in either graph by how the estimate compares with the truth.

    ``wrong`` means both graphs join the pair with different endpoint marks, including a directed
    edge in one and an undirected edge in the other.
    """
    est, true = estimate.to_numpy(), truth.to_numpy()
    rows = []
    for i, j in zip(*np.triu_indices(len(est), 1)):
        in_est, in_true = bool(est[i, j] or est[j, i]), bool(true[i, j] or true[j, i])
        if not in_est and not in_true:
            continue
        if not in_est:
            status = "missing"
        elif not in_true:
            status = "extra"
        elif (est[i, j], est[j, i]) == (true[i, j], true[j, i]):
            status = "correct"
        else:
            status = "wrong"
        rows.append((int(i), int(j), status))
    return rows


def metrics(estimate, truth):
    """SHD, skeleton F1, and arrowhead F1 of a CPDAG estimate."""
    result = score(estimate, truth)
    return {
        "shd": result["shd"],
        "skeleton_f1": result["skeleton_f1"],
        "arrow_f1": result["arrowhead_f1"],
    }


def number(key, value):
    """Format a metric for display."""
    return str(int(value)) if key == "shd" else f"{value:.2f}"


def edge_text(marks, i, j):
    """Write an edge as, for example, ``rain --> moisture`` or ``pests --- temperature``."""
    if (marks[i, j], marks[j, i]) == (ARROW, TAIL):
        i, j = j, i
    left = {TAIL: "-", ARROW: "<", CIRCLE: "o"}[int(marks[i, j])]
    right = {TAIL: "-", ARROW: ">", CIRCLE: "o"}[int(marks[j, i])]
    return f"{LABELS[i]} {left}-{right} {LABELS[j]}"


def describe(status, estimate, truth, i, j):
    """Name an edge's status and its endpoints in words."""
    est, true = estimate.to_numpy(), truth.to_numpy()
    if status == "missing":
        return f"Missing: the truth has {edge_text(true, i, j)}"
    if status == "extra":
        return f"Extra: {edge_text(est, i, j)} is not in the truth"
    if status == "wrong":
        return f"Wrong direction: {edge_text(est, i, j)} here, {edge_text(true, i, j)} in the truth"
    return f"Correct: {edge_text(est, i, j)}"


# The page resolves colors from its light/dark tokens; the thumbnail passes a concrete Palette.
PAGE_PALETTE = dataclasses.replace(
    TOKEN_PALETTE,
    categorical=tuple(f"var(--andrey-categorical-{k})" for k in range(len(LIGHT.categorical))),
)


def status_color(palette, status):
    """Graphite for a correct edge; mistakes avoid the colors that mean an edge type."""
    return {
        "correct": palette.edge_directed,
        "missing": palette.muted,
        "extra": palette.categorical[3],  # vermilion
        "wrong": palette.categorical[5],  # sky
    }[status]


def categorical_css():
    """Light and dark tokens for the categorical hues, following the host theme like the rest."""

    def block(palette):
        return ";".join(f"--andrey-categorical-{k}:{c}" for k, c in enumerate(palette.categorical))

    light, dark = block(LIGHT), block(DARK)
    return (
        f":root{{{light}}}@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{{dark}}}}}"
        f":root[data-theme=dark]{{{dark}}}"
    )


def panel_svg(graph, positions, palette, *, label, truth=None, ground=None, size=None, font=FONT):
    """Draw ``graph`` over fixed node positions; with ``truth``, color each edge by its status.

    ``ground`` paints a background and outlines labels in the same color; ``size`` sets the SVG's
    width and height attributes, which an ``<img>`` needs; ``font`` is the label size.
    """
    n = len(LABELS)
    pos = {
        k: (
            PAD_LEFT + x * (WIDTH - PAD_LEFT - PAD_RIGHT),
            PAD_TOP + y * (HEIGHT - PAD_TOP - PAD_BOTTOM),
        )
        for k, (x, y) in positions.items()
    }
    marks = graph.to_numpy()
    if truth is None:
        edges = [(i, j, "correct") for i, j in zip(*np.nonzero(np.triu(marks | marks.T)))]
    else:
        order = list(STATUSES)
        edges = sorted(edge_statuses(graph, truth), key=lambda edge: order.index(edge[2]))
    halo = ground or palette.surface
    dimensions = f' width="{size[0]}" height="{size[1]}"' if size else ' width="100%"'
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}"{dimensions} '
        f'role="img" aria-label="{escape(label)}" font-family="Inter, system-ui, sans-serif">'
    ]
    if ground:
        parts.append(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{ground}"/>')
    for i, j, status in edges:
        i, j = int(i), int(j)
        (ax, ay), (bx, by) = pos[i], pos[j]
        bend = _svg._route(ax, ay, bx, by, [pos[k] for k in range(n) if k not in (i, j)])
        mx, my = _svg._control(ax, ay, bx, by, bend)
        ends = (TAIL, TAIL) if status == "missing" else (int(marks[i, j]), int(marks[j, i]))
        color = status_color(palette, status)
        line = _svg._edge(ax, ay, bx, by, *ends, color, palette, bend=bend)
        dash = ' stroke-dasharray="6 5"' if status == "missing" else ""
        if truth is None:
            title = f"True edge: {edge_text(marks, i, j)}"
        else:
            title = describe(status, graph, truth, i, j)
        # A wide transparent stroke gives the thin edge a usable hover target for its title.
        parts.append(
            f'<g data-status="{status}"><title>{escape(title)}</title>'
            f'<path d="M {ax:.1f} {ay:.1f} Q {mx:.1f} {my:.1f} {bx:.1f} {by:.1f}" fill="none" '
            f'stroke="transparent" stroke-width="12"/><g{dash}>{line}</g></g>'
        )
    for k, (x, y) in pos.items():
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{_svg.NODE_R}" fill="{palette.surface}" '
            f'stroke="{palette.ink}" stroke-width="1.8"/>'
            f'<text x="{x:.1f}" y="{y + _svg.NODE_R + font + 1:.1f}" text-anchor="middle" '
            f'font-size="{font}" font-weight="500" fill="{palette.ink}" stroke="{halo}" '
            f'stroke-width="4" stroke-linejoin="round" paint-order="stroke">'
            f"{escape(LABELS[k])}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def layout():
    """Node positions from the true DAG, shared by every panel."""
    _, dag = dataset()
    return andrey.viz.layout(dag)


def counts(statuses):
    """How many edges have each status."""
    tally = {status: 0 for status in STATUSES}
    for *_, status in statuses:
        tally[status] += 1
    return tally


def count_text(tally):
    """Name each status count in words."""
    return ", ".join(f"{tally[key]} {name.lower()}" for key, name in STATUSES.items())


def estimate_panel(package, method, estimate, truth, positions):
    """One estimate: its drawing, its three numbers, and its differences in words."""
    statuses = edge_statuses(estimate, truth)
    tally = count_text(counts(statuses))
    svg = panel_svg(
        estimate, positions, PAGE_PALETTE, label=f"{package} {method} graph: {tally}", truth=truth
    )
    values = metrics(estimate, truth)
    numbers = "".join(
        f'<div><dt>{name}</dt><dd data-metric="{key}">{number(key, values[key])}</dd></div>'
        for key, name in METRICS.items()
    )
    differences = "".join(
        f"<li>{escape(describe(status, estimate, truth, i, j))}</li>"
        for i, j, status in statuses
        if status != "correct"
    )
    listing = (
        f"<details><summary>Differences</summary><ul>{differences}</ul></details>"
        if differences
        else "<p>No differences.</p>"
    )
    return (
        f'<figure class="panel" data-package="{escape(package)}">'
        f"<figcaption><h2>{escape(package)}</h2><span>{method} estimate</span></figcaption>"
        f'{svg}<dl class="metrics">{numbers}</dl><p class="tally">{escape(tally)}.</p>'
        f"{listing}</figure>"
    )


def truth_panel(truth, positions):
    """The reference drawing, in the neutral edge color only."""
    marks = truth.to_numpy()
    edges = int(np.count_nonzero(np.triu(marks | marks.T)))
    undirected = int(np.count_nonzero(np.triu((marks == TAIL) & (marks.T == TAIL))))
    label = f"True graph: {edges} edges, {undirected} of them undirected"
    return (
        '<figure class="panel truth" data-package="truth">'
        "<figcaption><h2>True graph</h2><span>CPDAG</span></figcaption>"
        f"{panel_svg(truth, positions, PAGE_PALETTE, label=label)}"
        "<p>The truth is drawn as its CPDAG, the graph the data can identify. Edges without "
        "arrowheads are undirected.</p></figure>"
    )


def legend():
    """Line samples for the four statuses, with their names in text."""
    items = []
    for status, name in STATUSES.items():
        dash = ' stroke-dasharray="6 5"' if status == "missing" else ""
        items.append(
            '<li><svg viewBox="0 0 32 10" width="32" height="10" aria-hidden="true">'
            f'<line x1="2" y1="5" x2="30" y2="5" stroke="{status_color(PAGE_PALETTE, status)}" '
            f'stroke-width="2.5" stroke-linecap="round"{dash}/></svg>{name}</li>'
        )
    return f'<ul class="legend" aria-label="Edge colors">{"".join(items)}</ul>'


CSS = """
*{box-sizing:border-box}
body{margin:0;padding:20px;background:var(--andrey-paper);color:var(--andrey-ink);
font:16px/1.55 Inter,system-ui,sans-serif}
main{max-width:1320px;margin:0 auto}
h1{font-size:1.6rem;font-weight:600;margin:0 0 8px}
p{max-width:72ch;margin:0 0 10px}
.note{color:var(--andrey-muted);font-size:.9rem}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px 24px;margin:16px 0 12px}
fieldset{display:flex;gap:0;border:0;margin:0;padding:0}
legend{float:left;margin-right:10px;font-weight:600;line-height:36px}
.choice{position:relative}
.choice input{position:absolute;opacity:0;inset:0;margin:0;cursor:pointer}
.choice span{display:block;min-width:64px;padding:6px 16px;text-align:center;
border:1px solid var(--andrey-line);background:var(--andrey-surface);font-weight:600}
.choice:first-of-type span{border-radius:8px 0 0 8px}
.choice:last-of-type span{border-radius:0 8px 8px 0;border-left:0}
.choice input:checked+span{background:var(--andrey-accent);border-color:var(--andrey-accent);
color:var(--andrey-paper)}
.choice input:focus-visible+span{outline:2px solid var(--andrey-accent);outline-offset:3px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:0;padding:0;list-style:none;
font-size:.9rem}
.legend li{display:flex;align-items:center;gap:6px}
.panels{display:grid;gap:14px;margin-top:12px}
.panel{margin:0;padding:12px 14px;border:1px solid var(--andrey-line);border-radius:10px;
background:var(--andrey-surface);min-width:0}
figcaption{display:flex;align-items:baseline;gap:8px;margin-bottom:4px}
figcaption h2{font-size:1.05rem;font-weight:600;margin:0}
figcaption span{color:var(--andrey-muted);font-size:.85rem}
.panel svg{display:block;width:100%;height:auto;max-width:520px;margin:0 auto}
.panel p{font-size:.85rem;color:var(--andrey-muted);margin:6px 0 0}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0 0}
.metrics div{border-top:1px solid var(--andrey-line);padding-top:6px}
.metrics dt{font-size:.8rem;color:var(--andrey-muted)}
.metrics dd{margin:0;font-size:2rem;font-weight:600;line-height:1.15;
font-variant-numeric:tabular-nums}
details{font-size:.85rem;margin-top:6px}
summary{cursor:pointer;color:var(--andrey-muted)}
summary:focus-visible{outline:2px solid var(--andrey-accent);outline-offset:2px}
details ul{margin:4px 0 0;padding-left:18px}
[hidden]{display:none!important}
@media(max-width:520px){body{padding:12px}.panel{padding:10px}.metrics dd{font-size:1.7rem}}
@media(min-width:700px){.panels{grid-template-columns:1fr 1fr}.truth{grid-column:1/-1}
.truth svg{max-width:calc(50% - 7px)}}
@media(min-width:1080px){.panels{grid-template-columns:repeat(3,1fr)}.truth{grid-column:auto}
.truth svg{max-width:520px}}
"""

SCRIPT = """
const sections = [...document.querySelectorAll("section[data-method]")];
for (const input of document.querySelectorAll("input[name=method]")) {
  input.addEventListener("change", () => {
    for (const section of sections) section.hidden = section.dataset.method !== input.value;
  });
}
"""


def compare_page(heading=True):
    """Build the standalone comparison page: both methods, both packages, and the truth.

    ``heading=False`` leaves out the title and the introduction, for a page that gives its own.
    """
    data, _ = dataset()
    truth = truth_cpdag()
    positions = layout()
    ours, theirs = andrey_graphs(), causal_learn_graphs()
    version = json.loads(RECORDING.read_text())["causal-learn"]
    choices = "".join(
        f'<label class="choice"><input type="radio" name="method" value="{method}"'
        f"{' checked' if method == METHODS[0] else ''}><span>{method}</span></label>"
        for method in METHODS
    )
    sections = "".join(
        f'<section data-method="{method}" aria-label="{method}"'
        f'{"" if method == METHODS[0] else " hidden"}><div class="panels">'
        + truth_panel(truth, positions)
        + estimate_panel("Andrey", method, ours[method], truth, positions)
        + estimate_panel("causal-learn", method, theirs[method], truth, positions)
        + "</div></section>"
        for method in METHODS
    )
    n, d = data.shape
    intro = (
        "<h1>Compare on one dataset</h1>"
        "<p>Andrey and causal-learn each learn a graph from the same simulated crop data. Each "
        "estimate is drawn beside the true graph, with every edge colored by how it compares with "
        "the truth.</p>"
    )
    body = (
        f"<main>{intro if heading else ''}"
        f'<p class="note">{n:,} samples of {d} variables from a linear-Gaussian model, '
        f"standardized and rounded to six decimals, seed {SEED}. causal-learn's graphs were "
        "recorded with causal-learn "
        f"{escape(version)} on the same data.</p>"
        f'<div class="controls"><fieldset><legend>Method</legend>{choices}</fieldset>'
        f"{legend()}</div>"
        f'<p class="note">PC uses the Fisher-z test at alpha {ALPHA}. GES uses the BIC score with '
        "the same penalty and parent limit in both packages. A wrong direction includes an edge "
        "that is directed in one graph and undirected in the other.</p>"
        f"{sections}"
        '<p class="note">SHD counts missing, extra, and wrong-direction edges; lower SHD is '
        "better. Skeleton F1 scores which pairs are joined and arrow F1 scores arrowheads; higher "
        "F1 is better.</p></main>"
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Compare on one dataset</title><style>{palette_css()}{categorical_css()}{CSS}"
        f"</style><body>{body}<script>{FRAME_SCRIPT}{SCRIPT}</script></body></html>"
    )


def compare_thumbnail(method="PC"):
    """Andrey's colored estimate as a standalone light SVG for an ``<img>``."""
    truth = truth_cpdag()
    estimate = andrey_graphs()[method]
    tally = count_text(counts(edge_statuses(estimate, truth)))
    return panel_svg(
        estimate,
        layout(),
        LIGHT,
        label=f"Andrey {method} graph: {tally}",
        truth=truth,
        ground=LIGHT.paper,
        size=(WIDTH, HEIGHT),
        font=17,  # still clear of its neighbors; over 10 px on a 280 px thumbnail
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--record",
        action="store_true",
        help=f"record causal-learn's graphs (needs causal-learn {CAUSAL_LEARN})",
    )
    action.add_argument(
        "--out", type=Path, help="write compare.html and compare-thumbnail.svg into this directory"
    )
    args = parser.parse_args(argv)
    if args.record:
        record()
        return
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "compare.html").write_text(compare_page())
    (args.out / "compare-thumbnail.svg").write_text(compare_thumbnail())


if __name__ == "__main__":
    main()
