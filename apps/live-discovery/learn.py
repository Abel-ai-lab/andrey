"""Engine-backed PC and Meek replays, and d-separation explanations."""

from __future__ import annotations

import dataclasses
import itertools
import json
import xml.etree.ElementTree as ET
from collections import Counter
from html import escape

import networkx as nx
import numpy as np

import andrey.viz
from andrey import GraphStructure
from andrey.constraint.pc import pc as pc_engine
from andrey.core.orient import meek
from andrey.data import sample_scm
from andrey.data.latent import DSepOracle
from andrey.viz.theme import DARK, LIGHT

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

RULES = {
    "R1": "A directed edge enters a node with an undirected edge. Orient the undirected edge away "
    "from that node when the outer nodes are not adjacent, avoiding a new collider.",
    "R2": "A two-edge directed path connects the ends of an undirected edge. Orient that "
    "edge along the path, avoiding a directed cycle.",
    "R3": "Two nonadjacent parents point into a shared child. A node with undirected "
    "edges to both parents and the child must point into the child.",
}


# An embedded page takes its theme from the host and reports its height, so the host's iframe
# neither clips it nor follows the system theme when the host has chosen another.
FRAME_SCRIPT = """
addEventListener("message", (event) => {
  const theme = event.source === parent && event.data && event.data.andreyTheme;
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
});
new ResizeObserver(() => {
  parent.postMessage({ andreyHeight: document.body.scrollHeight }, "*");
}).observe(document.body);
"""
HOST_SCRIPT = """
(() => {
  const dark = matchMedia("(prefers-color-scheme: dark)");
  const theme = () => {
    const app = document.querySelector(".gradio-container");
    if (app) return app.closest(".dark") ? "dark" : "light";
    const chosen = document.documentElement.dataset.theme;
    return chosen === "light" || chosen === "dark" ? chosen : dark.matches ? "dark" : "light";
  };
  const send = (frame) => frame.contentWindow?.postMessage({ andreyTheme: theme() }, "*");
  const frames = () => [...document.querySelectorAll("iframe")];
  addEventListener("message", (event) => {
    const frame = frames().find((candidate) => candidate.contentWindow === event.source);
    const height = event.data && event.data.andreyHeight;
    if (!frame || !Number.isFinite(height)) return;
    frame.style.height = `${Math.min(Math.ceil(height), 20000)}px`;
    send(frame);
  });
  const broadcast = () => frames().forEach(send);
  dark.addEventListener("change", broadcast);
  addEventListener("DOMContentLoaded", () => {
    for (const node of [document.documentElement, document.body]) {
      new MutationObserver(broadcast).observe(node, { attributeFilter: ["class", "data-theme"] });
    }
  });
})();
"""


def palette_css():
    """Produce light/dark tokens that follow the host's theme, or the system's without a host."""

    def block(palette, scheme):
        return f"color-scheme:{scheme};" + ";".join(
            f"--andrey-{f.name.replace('_', '-')}:{getattr(palette, f.name)}"
            for f in dataclasses.fields(palette)
            if isinstance(getattr(palette, f.name), str)
        )

    light, dark = block(LIGHT, "light"), block(DARK, "dark")
    return (
        f":root{{{light}}}@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{{dark}}}}}"
        f":root[data-theme=dark]{{{dark}}}"
    )


TOKEN_PALETTE = dataclasses.replace(
    LIGHT,
    **{
        f.name: f"var(--andrey-{f.name.replace('_', '-')})"
        for f in dataclasses.fields(LIGHT)
        if isinstance(getattr(LIGHT, f.name), str)
    },
)


def drawing(graph, **options):
    """Render an SVG whose colors follow the active host palette."""
    return str(andrey.viz.draw(graph, palette=TOKEN_PALETTE, **options))


def estimate_and_truth(estimate, truth):
    """Draw a recovered graph beside its truth, with every node in the same place."""
    positions = andrey.viz.layout(truth)
    return (
        '<div class="graph-columns">'
        f"<div><h4>Andrey</h4>{drawing(estimate, positions=positions)}</div>"
        f"<div><h4>Known truth</h4>{drawing(truth, positions=positions)}</div></div>"
    )


def document(title, body, script="", style=""):
    """Build one offline, dependency-free teaching page; ``style`` extends the shared CSS."""
    css = (
        palette_css()
        + """
    *{box-sizing:border-box}body{margin:0;padding:20px;background:var(--andrey-paper);
    color:var(--andrey-ink);font:16px/1.6 Inter,system-ui,sans-serif}
    h1,h2{font-weight:600}h1{font-size:1.6rem;margin:0 0 12px}
    p{max-width:68ch}button,select{font:inherit;color:var(--andrey-ink);
    background:var(--andrey-surface);border:1px solid var(--andrey-line);border-radius:6px;
    padding:6px 14px}button{cursor:pointer}button:disabled{opacity:.5;cursor:default}
    button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid
    var(--andrey-accent);outline-offset:3px}nav{display:flex;gap:8px;margin:12px 0}
    .graph{max-width:620px}svg{max-height:330px}label{margin-right:12px}
    [hidden]{display:none!important}.step{border-top:1px solid var(--andrey-line);padding-top:12px}
    @media(prefers-reduced-motion:no-preference){.step{animation:reveal .15s ease-out}
    @keyframes reveal{from{opacity:.75}to{opacity:1}}}
    """
        + style
    )
    payload = f"<script>{FRAME_SCRIPT}{script}</script>"
    return (
        f'<!doctype html><html lang="en"><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title><style>{css}</style><body>{body}{payload}</body></html>"
    )


# Hand-placed so no edge passes a node it does not join; every frame of a replay reuses them.
MEEK_INPUTS = {
    "R1": (
        4,
        [(0, 1)],
        [(1, 2), (2, 3)],
        {0: (0, 0.5), 1: (1 / 3, 0.5), 2: (2 / 3, 0.5), 3: (1, 0.5)},
    ),
    "R2": (3, [(0, 1), (1, 2)], [(0, 2)], {0: (0, 1), 1: (0.5, 0), 2: (1, 1)}),
    "R3": (
        4,
        [(1, 3), (2, 3)],
        [(0, 1), (0, 2), (0, 3)],
        {0: (0.5, 0), 1: (0, 0.5), 2: (1, 0.5), 3: (0.5, 1)},
    ),
}


def meek_inputs():
    """Small input patterns that exercise the engine's three Meek rules."""
    patterns = {}
    for name, (n, directed, undirected, _) in MEEK_INPUTS.items():
        marks = np.zeros((n, n), dtype=np.int8)
        for i, j in directed:
            marks[i, j], marks[j, i] = 1, 2
        for i, j in undirected:
            marks[i, j] = marks[j, i] = 1
        patterns[name] = marks
    return patterns


def meek_frames():
    """Record each rule firing from the orientation engine as one drawn frame."""
    frames = []
    for name, initial in meek_inputs().items():
        positions = MEEK_INPUTS[name][3]
        trace = []
        meek(initial, _trace=trace)
        steps = [(f"{name}: input", "Start with the partially directed graph.", initial, None)]
        for step in trace:
            ((head, tail),) = np.argwhere(step["after"] != step["before"]).tolist()
            nodes = ", ".join(str(i) for i in step["triple"])
            title = f"{step['rule']}: nodes {nodes}; orient {tail} -> {head}"
            steps.append((title, RULES[step["rule"]], step["after"], (head, tail)))
        for title, note, marks, changed in steps:
            graph = GraphStructure.from_numpy(marks, kind="cpdag")
            svg = drawing(graph, positions=positions, highlight=[changed] if changed else [])
            frames.append(dict(replay=name, title=title, note=note, marks=marks, svg=svg))
    return frames


def meek_replay(heading=True):
    """Render frames solely from firings emitted by the real orientation engine.

    ``heading=False`` leaves out the title and the introduction, for a page that gives its own.
    """
    content = []
    for index, frame in enumerate(meek_frames()):
        content.append(
            f'<section class="step" {"" if index == 0 else "hidden"}>'
            f"<h2>{escape(frame['title'])}</h2><p>{escape(frame['note'])}</p>"
            f'<div class="graph">{frame["svg"]}</div></section>'
        )
    intro = (
        "<h1>Meek's rules, one step at a time</h1>"
        "<p>Orient edges without introducing a new collider or a directed cycle. A ring marks "
        "the arrowhead each step adds.</p>"
    )
    body = (
        (intro if heading else "")
        + '<nav aria-label="Replay controls"><button id="prev">Previous</button>'
        '<button id="next">Next</button><button id="reset">Reset</button></nav>'
        '<p id="position" aria-live="polite"></p>' + "".join(content)
    )
    script = """
    const frames=[...document.querySelectorAll('.step')];let index=0;
    function show(){frames.forEach((f,i)=>f.hidden=i!==index);
    document.getElementById('position').textContent='Step '+(index+1)+' of '+frames.length;
    document.getElementById('prev').disabled=index===0;
    document.getElementById('next').disabled=index===frames.length-1;}
    document.getElementById('prev').onclick=()=>{index=Math.max(0,index-1);show()};
    document.getElementById('next').onclick=()=>{index=Math.min(frames.length-1,index+1);show()};
    document.getElementById('reset').onclick=()=>{index=0;show()};show();
    """
    return document("Meek's rules", body, script)


# PC as andrey.constraint.pc runs it: PC-stable neighbourhoods, every subset tried, and the
# prioritize-existing collider rule. Each entry is (indent, text); frames name lines by number.
PC_LINES = (
    (0, "G <- the complete undirected graph"),
    (0, "for d = 0, 1, 2, … while some node has > d neighbours"),
    (1, "adj <- the neighbours in G, fixed for this depth"),
    (1, "for each adjacent pair (x, y)"),
    (2, "for each S of size d from adj(x) \\ {y} or adj(y) \\ {x}"),
    (3, "p <- test(x, y | S)"),
    (3, "if p > α: mark x – y, add S to sepset(x, y)"),
    (1, "remove the marked edges from G"),
    (0, "for each unshielded triple x – y – z"),
    (1, "if y ∉ sepset(x, z) and neither y -> x nor y -> z"),
    (2, "orient x -> y <- z"),
    (0, "apply Meek's rules R1–R3 until nothing changes"),
)
# The larger replay: a random linear-Gaussian DAG whose PC run reaches depth 4, orients colliders,
# and fires Meek's rules.
MATRIX_EXAMPLE = dict(d=40, n=2000, seed=1, density=2.0, scale="standardize")


class Replay:
    """Collect player frames and the distinct graph states they show."""

    def __init__(self, draw=None):
        self.draw, self.layers, self.frames, self.index = draw, [], [], {}

    def layer(self, marks, ring=()):
        key = (marks.tobytes(), tuple(ring))
        if key not in self.index:
            self.index[key] = len(self.layers)
            layer = dict(marks="".join(map(str, marks.ravel().tolist())))
            if self.draw:
                layer["svg"] = self.draw(marks, ring)
            self.layers.append(layer)
        return self.index[key]

    def add(self, lines, note, marks, *, ghost=None, ring=(), focus=(), context=(), variables=()):
        """``ghost`` is the start-of-depth graph; the player draws its marked edges faintly."""
        if ghost is not None and np.array_equal(ghost, marks):
            ghost = None
        self.frames.append(
            dict(
                lines=list(lines),
                note=note,
                now=self.layer(marks, ring),
                ghost=None if ghost is None else self.layer(ghost),
                focus=[int(k) for k in focus],
                context=[int(k) for k in context],
                vars=[[name, str(value)] for name, value in variables],
            )
        )

    def payload(self, **extra):
        return dict(layers=self.layers, frames=self.frames, **extra)


def trace_pc(X, alpha):
    """Run the PC engine once; return its CPDAG marks and every recorded step."""
    trace = []
    marks = pc_engine(np.asarray(X, dtype=np.float64), alpha=alpha, _trace=trace).to_numpy()
    return marks, trace


def complete_graph(n):
    return np.ones((n, n), dtype=np.int8) - np.eye(n, dtype=np.int8)


def counts(marks):
    """The directed and undirected edge counts of a CPDAG, as loop-variable rows."""
    directed = int(np.count_nonzero(marks == 2))
    return [
        ("directed", directed),
        ("undirected", int(np.count_nonzero(np.triu(marks, 1))) - directed),
    ]


def state_svg(marks, labels, positions, ring=()):
    """Draw one state with each node circle tagged so the player can ring the tested nodes."""
    graph = GraphStructure.from_numpy(marks, kind="cpdag", labels=labels)
    svg = ET.fromstring(drawing(graph, positions=positions, background=False, highlight=ring))
    nodes = [c for c in svg.iter(f"{{{SVG_NS}}}circle") if c.get("r") == "13.00"]
    for k, node in enumerate(nodes[-len(labels) :]):
        node.set("data-node", str(k))
    svg.set("role", "img")
    svg.set("aria-label", "The graph at this step")
    return ET.tostring(svg, "unicode")


def pc_walkthrough(X, labels, alpha=0.05):
    """One frame per independence test, collider decision, and Meek firing of a small PC run."""
    result, trace = trace_pc(X, alpha)
    n = len(labels)
    positions = andrey.viz.layout(GraphStructure.from_numpy(result, kind="cpdag"))
    replay = Replay(lambda marks, ring: state_svg(marks, labels, positions, ring))

    def names(nodes):
        return "{" + ", ".join(labels[k] for k in nodes) + "}" if nodes else "∅"

    adj = complete_graph(n)
    replay.add(
        [1],
        "PC starts from the complete graph: every pair of variables is adjacent.",
        adj,
        variables=[("edges", n * (n - 1) // 2)],
    )
    tests = [r for r in trace if r["step"] == "test"]
    depth = 0
    for depth in sorted({r["depth"] for r in tests}):
        snapshot, marked = adj.copy(), set()
        replay.add(
            [2, 3],
            f"Depth {depth}: test each adjacent pair given every set of {depth} of its "
            "neighbours, as they stand at the start of this depth."
            if depth
            else "Depth 0: test each adjacent pair with no conditioning set.",
            adj,
            variables=[("depth d", depth)],
        )
        for r in (r for r in tests if r["depth"] == depth):
            x, y, S = r["x"], r["y"], r["S"]
            given = f" given {names(S)}" if S else ""
            if r["removed"]:
                adj[x, y] = adj[y, x] = 0
                again = (x, y) in marked
                marked.add((x, y))
                note = f"{labels[x]} and {labels[y]} test independent{given}. " + (
                    f"{labels[x]} – {labels[y]} is already marked; S joins its separating set."
                    if again
                    else f"Mark {labels[x]} – {labels[y]}; it stays in adj until this depth ends."
                )
            else:
                note = f"{labels[x]} and {labels[y]} test dependent{given}, so the edge stays."
            relation = ">" if r["removed"] else "≤"
            replay.add(
                [7 if r["removed"] else 6],
                note,
                adj,
                ghost=snapshot,
                focus=(x, y),
                context=S,
                variables=[
                    ("depth d", depth),
                    ("pair x, y", f"{labels[x]}, {labels[y]}"),
                    ("set S", names(S)),
                    ("p vs α", f"{r['p']:.3f} {relation} {alpha:g}"),
                ],
            )
        replay.add(
            [8],
            f"Remove the {len(marked)} marked edge{'s' * (len(marked) != 1)}."
            if marked
            else "No edge was marked, so G is unchanged.",
            adj,
            variables=[("depth d", depth)],
        )
    replay.add(
        [2],
        f"No node has more than {depth + 1} neighbours, so the tests stop. What remains is the "
        "skeleton.",
        adj,
        variables=[("depth d", depth + 1)],
    )
    for r in trace:
        if r["step"] == "collider":
            x, y, z = r["triple"]
            sepset = names(r["sepset"])
            if r["oriented"]:
                line, ring = 11, [(y, x), (y, z)]
                note = f"{labels[y]} is not in sepset({labels[x]}, {labels[z]}), so orient "
                note += f"{labels[x]} -> {labels[y]} <- {labels[z]}."
            elif y in r["sepset"]:
                line, ring = 10, []
                note = f"{labels[y]} is in sepset({labels[x]}, {labels[z]}): not a collider."
            else:
                line, ring = 10, []
                note = f"An edge already points out of {labels[y]}, so this collider is skipped."
            replay.add(
                [line],
                note,
                r["after"],
                ring=ring,
                focus=(x, z),
                context=() if r["oriented"] else (y,),
                variables=[
                    ("triple", f"{labels[x]} – {labels[y]} – {labels[z]}"),
                    (f"sepset({labels[x]}, {labels[z]})", sepset),
                ],
            )
        elif r["step"] == "meek":
            ((head, tail),) = np.argwhere(r["after"] != r["before"]).tolist()
            replay.add(
                [12],
                f"{r['rule']}: orient {labels[tail]} -> {labels[head]}. {RULES[r['rule']]}",
                r["after"],
                ring=[(head, tail)],
                variables=[
                    ("rule", r["rule"]),
                    ("nodes", ", ".join(labels[k] for k in r["triple"])),
                ],
            )
    replay.add([], "Done: this is the CPDAG andrey.pc returns.", result, variables=counts(result))
    return replay.payload(n=n, labels=list(labels), interval=1100)


def matrix_example():
    """The larger replay's data, sampled at build time."""
    return sample_scm(**MATRIX_EXAMPLE).data


def pc_by_depth(X, alpha=0.05):
    """One matrix frame per depth's tests, per removal, and per orientation stage."""
    result, trace = trace_pc(X, alpha)
    n = result.shape[0]
    replay = Replay()

    def edges(marks):
        return int(np.count_nonzero(np.triu(marks, 1)))

    adj = complete_graph(n)
    replay.add(
        [1],
        f"PC starts with all {edges(adj):,} pairs adjacent.",
        adj,
        variables=[("edges", edges(adj))],
    )
    tests = [r for r in trace if r["step"] == "test"]
    depth = 0
    for depth in sorted({r["depth"] for r in tests}):
        at = [r for r in tests if r["depth"] == depth]
        snapshot = adj.copy()
        marked = {(r["x"], r["y"]) for r in at if r["removed"]}
        for x, y in marked:
            adj[x, y] = adj[y, x] = 0
        tally = [("depth d", depth), ("tests", f"{len(at):,}"), ("marked", len(marked))]
        replay.add(
            [4, 5, 6, 7],
            f"Depth {depth}: {len(at):,} tests mark {len(marked)} of {edges(snapshot):,} edges.",
            adj,
            ghost=snapshot,
            variables=tally,
        )
        replay.add(
            [8],
            f"Remove the marked edges; {edges(adj)} remain.",
            adj,
            variables=[*tally, ("edges left", edges(adj))],
        )
    replay.add(
        [2],
        f"No node has more than {depth + 1} neighbours, so the tests stop.",
        adj,
        variables=[("depth d", depth + 1), ("edges left", edges(adj))],
    )
    colliders = [r for r in trace if r["step"] == "collider"]
    if colliders:
        adj = colliders[-1]["after"]
    oriented = sum(r["oriented"] for r in colliders)
    replay.add(
        [9, 10, 11],
        f"Of {len(colliders)} unshielded triples, {oriented} orient as colliders.",
        adj,
        variables=[("triples", len(colliders)), ("colliders", oriented)],
    )
    firings = [r for r in trace if r["step"] == "meek"]
    if firings:
        adj = firings[-1]["after"]
    rules = Counter(r["rule"] for r in firings)
    replay.add(
        [12],
        f"Meek's rules orient {len(firings)} more edge{'s' * (len(firings) != 1)}.",
        adj,
        variables=[(rule, rules[rule]) for rule in RULES],
    )
    replay.add([], "Done: this is the CPDAG andrey.pc returns.", result, variables=counts(result))
    return replay.payload(n=n, labels=None, interval=1600)


PC_CSS = """
.scenes{display:inline-flex;gap:2px;padding:3px;border:1px solid var(--andrey-line);
border-radius:9px;background:var(--andrey-surface-2)}
.scenes button{border:0;background:none;padding:5px 14px}
.scenes button[aria-selected=true]{background:var(--andrey-surface);font-weight:600;
box-shadow:0 1px 2px color-mix(in srgb,var(--andrey-ink) 18%,transparent)}
.lede{color:var(--andrey-muted);margin:14px 0 0}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:14px 0 16px}
.controls .play{min-width:6.5em;background:var(--andrey-accent);border-color:var(--andrey-accent);
color:var(--andrey-paper);font-weight:600}.scrub{flex:1 1 140px;accent-color:var(--andrey-accent)}
.position{min-width:8.5em;color:var(--andrey-muted);font-size:.9rem;font-variant-numeric:tabular-nums}
.stage{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:20px;align-items:start}
@media(max-width:760px){.stage{grid-template-columns:minmax(0,1fr)}}
.panel{background:var(--andrey-surface);border:1px solid var(--andrey-line);border-radius:10px}
.code{margin:0;padding:10px 0;list-style:none;counter-reset:line;
font:13px/1.6 "DM Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.code li{display:flex;counter-increment:line;padding:1px 12px 1px 0}
.code li::before{content:counter(line);flex:none;width:2.6em;padding-right:.9em;text-align:right;
color:var(--andrey-muted)}.code li.current{background:color-mix(in srgb,var(--andrey-accent) 15%,
transparent);box-shadow:inset 3px 0 var(--andrey-accent)}
.code .i1{padding-left:1.5em}.code .i2{padding-left:3em}.code .i3{padding-left:4.5em}
.vars{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:3px 14px;margin:12px 0 0;
padding:10px 14px;min-height:7.6em;align-content:start;font-variant-numeric:tabular-nums}
.vars dt{color:var(--andrey-muted)}.vars dd{margin:0;overflow-wrap:anywhere;
font-family:"DM Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.note{margin:12px 0 0;min-height:4.8em}
.graph{display:grid;padding:6px}.graph>div{grid-area:1/1}.graph .ghost{opacity:.3}
.graph .leaving{opacity:0;pointer-events:none}.graph svg{display:block;max-height:none}
circle.focus{stroke:var(--andrey-accent);stroke-width:3.5}
circle.context{stroke:var(--andrey-accent);stroke-width:2.5;stroke-dasharray:4 3}
.matrix{display:grid;grid-template-columns:auto repeat(var(--n),minmax(0,1fr));gap:3px;
max-width:270px;margin:14px 0 0}
.matrix .c{position:relative;display:grid;place-items:center;aspect-ratio:1;border-radius:4px;
background:var(--andrey-surface-2)}.matrix .diag{background:none}
.matrix .u{color:var(--andrey-edge-undirected);
background:color-mix(in srgb,var(--andrey-edge-undirected) 18%,var(--andrey-surface))}
.matrix .d{color:var(--andrey-edge-directed);
background:color-mix(in srgb,var(--andrey-edge-directed) 16%,var(--andrey-surface))}
.matrix .m{opacity:.3}.matrix .f{outline:2px solid var(--andrey-accent);outline-offset:-2px}
.matrix .x{box-shadow:inset 0 0 0 2px var(--andrey-accent)}
.matrix .label{align-self:center;font-size:.72rem;color:var(--andrey-muted);overflow-wrap:anywhere}
.matrix .col{text-align:center;align-self:end}.matrix .row{text-align:right;padding-right:4px}
.matrix .t::after,.matrix .a::after{content:"";background:currentColor}
.matrix .t::after{width:46%;height:12%;border-radius:2px}
.matrix .a::after{width:44%;height:44%;clip-path:polygon(0 0,100% 50%,0 100%)}
.matrix.compact{grid-template-columns:repeat(var(--n),minmax(0,1fr));gap:1px;max-width:440px;
margin:0}.matrix.compact .c{border-radius:1px}.matrix.compact .u,.matrix.compact .d{background:none}
.matrix.compact .t::after{width:80%;height:34%}.matrix.compact .a::after{width:90%;height:90%}
.legend{margin:10px 0 0;font-size:.85rem;color:var(--andrey-muted)}
.legend b{font-weight:600}.legend .u{color:var(--andrey-edge-undirected)}
.legend .d{color:var(--andrey-edge-directed)}.legend .a{color:var(--andrey-accent)}
@media(prefers-reduced-motion:no-preference){.graph .leaving.fade{animation:leave .4s ease-out}
circle[data-node]{transition:stroke .2s,stroke-width .2s}.matrix .c{transition:opacity .3s}}
@keyframes leave{from{opacity:1}to{opacity:0}}
"""

PC_SCRIPT = """
const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
const MARK={1:'tail',2:'arrowhead'};
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;
  if(text!==undefined)e.textContent=text;return e}
function mount(root,data){
  const {n,labels,layers,frames}=data,q=s=>root.querySelector(s);
  const lines=[...root.querySelectorAll('.code li')],vars=q('.vars'),note=q('.note');
  const scrub=q('.scrub'),position=q('.position'),play=q('.play'),graph=q('.graph');
  const matrix=q('.matrix'),cells=[];
  matrix.style.setProperty('--n',n);
  if(labels){matrix.append(el('span'));labels.forEach(l=>matrix.append(el('span','label col',l)))}
  for(let i=0;i<n;i++){
    if(labels)matrix.append(el('span','label row',labels[i]));
    for(let j=0;j<n;j++){const c=el('span','c');matrix.append(c);cells.push(c)}
  }
  scrub.max=frames.length-1;
  let index=reduce?frames.length-1:0,timer=null,drawn=null,ghostDrawn=null;
  function paintGraph(frame){
    if(!graph)return;
    const [ghost,now,leaving]=graph.children;
    if(frame.ghost!==ghostDrawn){ghost.innerHTML=frame.ghost===null?'':layers[frame.ghost].svg;
      ghostDrawn=frame.ghost}
    if(frame.now!==drawn){
      if(!reduce&&drawn!==null){leaving.innerHTML=now.innerHTML;leaving.classList.remove('fade');
        void leaving.offsetWidth;leaving.classList.add('fade')}
      now.innerHTML=layers[frame.now].svg;drawn=frame.now;
    }
    now.querySelectorAll('[data-node]').forEach(node=>{const k=+node.dataset.node;
      node.classList.toggle('focus',frame.focus.includes(k));
      node.classList.toggle('context',frame.context.includes(k))});
  }
  function paintMatrix(frame,previous){
    const now=layers[frame.now].marks,ghost=frame.ghost===null?null:layers[frame.ghost].marks;
    const before=previous?layers[previous.now].marks:now;
    for(let i=0;i<n;i++)for(let j=0;j<n;j++){
      const k=i*n+j,c=cells[k],marked=ghost!==null&&now[k]==='0'&&ghost[k]!=='0';
      const marks=marked?ghost:now,mark=+marks[k],other=+marks[j*n+i];
      let cls='c';
      if(i===j)cls+=' diag';
      else if(mark){cls+=(mark===1&&other===1?' u':' d')+(mark===1?' t':' a');
        if(marked)cls+=' m';else if(before[k]!==now[k])cls+=' x'}
      if(i!==j&&frame.focus.includes(i)&&frame.focus.includes(j))cls+=' f';
      c.className=cls;
      if(labels){c.title=i===j||!mark?'':`${MARK[mark]} at ${labels[i]} on `+
          `${labels[i]} \\u2013 ${labels[j]}`+(marked?', marked for removal':'')}
    }
  }
  function show(target){
    index=Math.max(0,Math.min(frames.length-1,target));
    const frame=frames[index];
    lines.forEach((li,k)=>{const on=frame.lines.includes(k+1);li.classList.toggle('current',on);
      if(on)li.setAttribute('aria-current','step');else li.removeAttribute('aria-current')});
    vars.replaceChildren(...frame.vars.flatMap(([k,v])=>[el('dt','',k),el('dd','',v)]));
    note.textContent=frame.note;
    paintGraph(frame);paintMatrix(frame,frames[index-1]);
    const where=`Step ${index+1} of ${frames.length}`;
    scrub.value=index;scrub.setAttribute('aria-valuetext',where);position.textContent=where;
    q('.back').disabled=index===0;q('.forward').disabled=index===frames.length-1;
  }
  function pause(){clearInterval(timer);timer=null;play.textContent='Play';
    play.setAttribute('aria-pressed','false');note.setAttribute('aria-live','polite')}
  function start(){if(index===frames.length-1)show(0);note.setAttribute('aria-live','off');
    timer=setInterval(()=>index<frames.length-1?show(index+1):pause(),data.interval);
    play.textContent='Pause';play.setAttribute('aria-pressed','true')}
  play.onclick=()=>timer?pause():start();
  q('.back').onclick=()=>{pause();show(index-1)};
  q('.forward').onclick=()=>{pause();show(index+1)};
  scrub.oninput=()=>{pause();show(+scrub.value)};
  show(index);
  return pause;
}
const pauses=[...document.querySelectorAll('.player')].map(p=>mount(p,DATA[p.id]));
const tabs=[...document.querySelectorAll('[role=tab]')];
function select(tab){pauses.forEach(pause=>pause());tabs.forEach(t=>{const on=t===tab;
  t.setAttribute('aria-selected',String(on));t.tabIndex=on?0:-1;
  document.getElementById(t.getAttribute('aria-controls')).hidden=!on})}
tabs.forEach((tab,k)=>{tab.onclick=()=>select(tab);tab.onkeydown=event=>{
  const step={ArrowRight:1,ArrowLeft:-1}[event.key];if(!step)return;event.preventDefault();
  const next=tabs[(k+step+tabs.length)%tabs.length];next.focus();select(next)}});
"""


def player(key, lede, legends, *, graph):
    """One replay's markup: controls, pseudocode, loop variables, then the graph and matrix."""
    code = "".join(
        f'<li><span class="i{indent}">{escape(text)}</span></li>' for indent, text in PC_LINES
    )
    view = (
        '<div class="graph panel"><div class="ghost"></div><div class="now"></div>'
        '<div class="leaving" aria-hidden="true"></div></div>'
        if graph
        else ""
    )
    graph_legend, matrix_legend = (f'<p class="legend">{text}</p>' for text in legends)
    matrix_class = "matrix" if graph else "matrix compact"
    return (
        f'<section class="player" id="{key}" role="tabpanel" aria-labelledby="tab-{key}"'
        f'{" hidden" if key != "small" else ""}><p class="lede">{lede}</p>'
        '<div class="controls" role="group" aria-label="Replay controls">'
        '<button class="back">Back</button><button class="play" aria-pressed="false">Play'
        '</button><button class="forward">Step</button>'
        '<input class="scrub" type="range" min="0" value="0" aria-label="Step">'
        '<span class="position"></span></div>'
        '<div class="stage"><div><ol class="code panel" aria-label="PC pseudocode">'
        f'{code}</ol><dl class="vars panel" aria-label="Loop variables"></dl>'
        '<p class="note" aria-live="polite"></p></div>'
        f'<div>{view}{graph_legend if graph else ""}<div class="{matrix_class}" role="img" '
        'aria-label="Adjacency matrix of endpoint marks"></div>'
        f"{matrix_legend}</div></div></section>"
    )


def sprinkler_data(n=5000, seed=0):
    """Rain and a sprinkler wet the grass, and wet grass is slippery: the replay's four variables
    and their names."""
    rng = np.random.default_rng(seed)
    rain = rng.standard_normal(n)
    sprinkler = rng.standard_normal(n)
    wet = rain + sprinkler + rng.standard_normal(n)
    slippery = wet + rng.standard_normal(n)
    labels = ("rain", "sprinkler", "wet", "slippery")
    return np.column_stack([rain, sprinkler, wet, slippery]), labels


def pc_replay(example, labels, heading=True):
    """Replay PC on four variables, test by test, and on 40 variables by depth.

    ``heading=False`` leaves out the title and the introduction, for a page that gives its own.
    """
    data = {"small": pc_walkthrough(example, labels), "large": pc_by_depth(matrix_example())}
    names = ", ".join(labels)
    small = player(
        "small",
        "Rain and a sprinkler wet the grass, and wet grass is slippery: 5,000 samples of "
        f"{names}. One step per test.",
        (
            "Solid ring: the pair tested. Dashed ring: its conditioning set, or the middle of a "
            "triple. Faint edge: marked, removed when the depth ends.",
            "Row i, column j is the mark at i on the edge i – j: a bar for a tail, a triangle for "
            'an arrowhead; <b class="u">amber</b> undirected, <b class="d">graphite</b> directed.',
        ),
        graph=True,
    )
    settings = MATRIX_EXAMPLE
    large = player(
        "large",
        f"{settings['d']} variables, {settings['n']:,} samples from a random linear-Gaussian "
        f"graph (seed {settings['seed']}). One step per depth; the matrix alone shows the state.",
        (
            "",
            "Row i, column j is the mark at i on the edge i – j: a bar for a tail, a triangle for "
            'an arrowhead; <b class="u">amber</b> undirected, <b class="d">graphite</b> directed. '
            "Faint cells are marked at this depth.",
        ),
        graph=False,
    )
    intro = (
        "<h1>PC, one step at a time</h1>"
        "<p>PC removes edges by conditional-independence tests, orients colliders, then applies "
        "Meek's rules. Every step here was recorded by Andrey's PC as it ran; the page only "
        "replays them.</p>"
    )
    body = (
        (intro if heading else "") + '<div class="scenes" role="tablist" aria-label="Examples">'
        '<button role="tab" id="tab-small" aria-controls="small" aria-selected="true">'
        f"{len(labels)} variables</button>"
        '<button role="tab" id="tab-large" aria-controls="large" aria-selected="false" '
        f'tabindex="-1">{settings["d"]} variables</button></div>{small}{large}'
    )
    encoded = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    return document("PC step by step", body, f"const DATA={encoded};{PC_SCRIPT}", PC_CSS)


def preset_dag():
    """A fork, collider, and collider descendant in one small DAG."""
    dag = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)])
    dag.add_nodes_from(range(5))
    return dag


def classify_paths(dag, x, y, conditioned, limit=100):
    """Explain simple paths; cap enumeration independently of the oracle's verdict."""
    conditioned = set(conditioned)
    if x == y or x not in dag or y not in dag or {x, y} & conditioned:
        raise ValueError("Choose distinct endpoints outside the conditioning set.")
    if not conditioned <= set(dag) or not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Choose DAG nodes for the conditioning set.")
    paths = list(itertools.islice(nx.all_simple_paths(dag.to_undirected(), x, y), limit + 1))
    opened_colliders = conditioned | set().union(*(nx.ancestors(dag, z) for z in conditioned))
    details = []
    for path in paths[:limit]:
        blocker, reason = None, "No interior node blocks this path."
        for left, middle, right in zip(path, path[1:], path[2:]):
            collider = dag.has_edge(left, middle) and dag.has_edge(right, middle)
            if collider and middle not in opened_colliders:
                blocker = middle
                reason = f"Collider {middle} and its descendants are outside the conditioning set."
                break
            if not collider and middle in conditioned:
                blocker = middle
                reason = f"Non-collider {middle} is conditioned on."
                break
        details.append(dict(path=path, active=blocker is None, blocker=blocker, reason=reason))
    return {
        "separated": bool(DSepOracle(dag)(x, y, conditioned)),
        "paths": details,
        "truncated": len(paths) > limit,
    }


def dsep_html(x, y, conditioned):
    """Answer a Space query live through DSepOracle and explain its paths."""
    dag = preset_dag()
    x, y = int(x), int(y)
    result = classify_paths(dag, x, y, [int(z) for z in conditioned])
    verdict = "d-separated" if result["separated"] else "d-connected"
    lines = [f"<h3>{x} and {y} are {verdict}</h3>", "<ul>"]
    for path in result["paths"]:
        route = " - ".join(map(str, path["path"]))
        status = "Active" if path["active"] else "Blocked"
        lines.append(f"<li><strong>{route}: {status}.</strong> {escape(path['reason'])}</li>")
    lines.append("</ul>")
    if result["truncated"]:
        lines.append("<p>Only the first 100 paths are listed; the verdict uses the whole DAG.</p>")
    return "".join(lines)


def dsep_document(heading=True):
    """Export every small-preset query from the oracle for an offline explorer.

    ``heading=False`` leaves out the title and the introduction, for a page that gives its own.
    """
    dag = preset_dag()
    graph = GraphStructure.from_networkx(dag)
    answers = {}
    for x, y in itertools.permutations(sorted(dag), 2):
        remaining = sorted(set(dag) - {x, y})
        for count in range(len(remaining) + 1):
            for condition in itertools.combinations(remaining, count):
                key = f"{x},{y}:" + ",".join(map(str, condition))
                answers[key] = dsep_html(x, y, condition)
    options = "".join(f'<option value="{n}">{n}</option>' for n in dag)
    checks = "".join(f'<label><input type="checkbox" value="{n}"> {n}</label>' for n in dag)
    intro = (
        "<h1>Which paths are open?</h1><p>Choose two nodes and a conditioning set. "
        "Conditioning blocks a chain or fork, but opens a collider path when the collider "
        "or one of its descendants is conditioned on.</p>"
    )
    body = (
        (intro if heading else "") + f'<div class="graph">{drawing(graph)}</div>'
        f'<label>From <select id="x">{options}</select></label>'
        f'<label>To <select id="y">{options}</select></label>'
        f"<fieldset><legend>Condition on</legend>{checks}</fieldset>"
        '<div id="answer" aria-live="polite"></div>'
    )
    encoded = json.dumps(answers).replace("<", "\\u003c")
    script = (
        f"const answers={encoded};"
        + """
    const x=document.getElementById('x'),y=document.getElementById('y');
    const checks=[...document.querySelectorAll('input')];x.value='1';y.value='2';
    function show(){checks.forEach(c=>{c.disabled=c.value===x.value||c.value===y.value;
    if(c.disabled)c.checked=false});const z=checks.filter(c=>c.checked).map(c=>c.value).join(',');
    const answer=document.getElementById('answer');
    if(x.value===y.value){answer.textContent='Choose two different nodes.';return}
    answer.innerHTML=answers[x.value+','+y.value+':'+z]}
    document.querySelectorAll('input,select').forEach(c=>c.onchange=show);show();
    """
    )
    return document("d-separation explorer", body, script)


def iframe(page, title, height=760):
    """Isolate inline player scripts in a srcdoc iframe; ``height`` holds until the page reports."""
    return (
        f'<iframe title="{escape(title)}" srcdoc="{escape(page, quote=True)}" '
        'sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox" '
        f'style="width:100%;height:{height}px;border:0"></iframe>'
    )
