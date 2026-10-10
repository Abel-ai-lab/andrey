"""Unit tests for ``andrey.viz`` -- the edge-type classifier and the SVG emitter."""

from __future__ import annotations

import re
import sys
import xml.dom.minidom as minidom

import numpy as np
import pytest

import andrey.viz as viz
from andrey.core.structure import ARROW, CIRCLE, LATENT, TAIL, GraphStructure, TemporalStructure
from andrey.viz import classify_edge


def _from(pairs: dict[tuple[int, int], tuple[int, int]], n: int, kind: str, **kw) -> GraphStructure:
    """Build a structure from ``{(i, j): (mark_i, mark_j)}`` (writes both symmetric cells)."""
    m = np.zeros((n, n), np.int8)
    for (i, j), (mi, mj) in pairs.items():
        m[i, j] = mi
        m[j, i] = mj
    return GraphStructure.from_numpy(m, kind=kind, **kw)


def test_classify_covers_every_mark_pair() -> None:
    assert classify_edge(TAIL, ARROW, "cpdag") == "directed"
    assert classify_edge(ARROW, TAIL, "cpdag") == "directed"
    assert classify_edge(TAIL, TAIL, "cpdag") == "undirected"
    assert classify_edge(CIRCLE, ARROW, "pag") == "partial"
    assert classify_edge(CIRCLE, CIRCLE, "pag") == "partial"
    assert classify_edge(CIRCLE, TAIL, "pag") == "partial"


def test_double_arrow_is_kind_relative() -> None:
    # ARROW/ARROW is a latent confounder under a PAG but a 2-cycle under a directed graph.
    assert classify_edge(ARROW, ARROW, "pag") == "bidirected"
    assert classify_edge(ARROW, ARROW, "digraph") == "twocycle"


@pytest.mark.parametrize(
    "kind,pairs,n",
    [
        ("dag", {(0, 1): (TAIL, ARROW), (1, 2): (TAIL, ARROW), (0, 2): (TAIL, ARROW)}, 3),
        ("cpdag", {(0, 1): (TAIL, ARROW), (1, 2): (TAIL, TAIL)}, 3),
        ("pag", {(0, 1): (CIRCLE, ARROW), (1, 2): (ARROW, ARROW)}, 3),
        ("digraph", {(0, 1): (ARROW, ARROW), (2, 0): (TAIL, ARROW)}, 3),
    ],
)
def test_draw_emits_valid_svg(kind: str, pairs: dict, n: int) -> None:
    svg = viz.draw(_from(pairs, n, kind))
    minidom.parseString(svg)  # raises on malformed XML
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "polygon" in svg  # at least one arrowhead


def test_two_cycle_draws_two_curved_edges() -> None:
    # A digraph 2-cycle (ARROW/ARROW) renders as two <path> arcs, not one double-headed line.
    dig = _from({(0, 1): (ARROW, ARROW)}, 2, "digraph")
    assert viz.draw(dig).count("<path") == 2
    # The same pair under a PAG is one straight bidirected edge (a <line>, no <path>).
    pag = _from({(0, 1): (ARROW, ARROW)}, 2, "pag")
    assert "<path" not in viz.draw(pag)
    assert viz.draw(pag).count("<line") == 1


def test_latent_node_is_hollow_dashed() -> None:
    pag = _from(
        {(0, 1): (TAIL, ARROW)},
        2,
        "pag",
        node_types=np.array([LATENT, 0], np.int8),
    )
    assert "stroke-dasharray" in viz.draw(pag)


def test_mpl_palette_data_needs_no_matplotlib() -> None:
    from andrey.viz import mpl

    assert len(mpl.cycle()) == 8
    assert mpl.cycle()[0] == "#3b5bdb"
    assert mpl.cycle(dark=True)[0] == "#6e86f5"
    assert mpl.rc_params()["figure.facecolor"] == "#fafbfc"
    assert mpl.rc_params(dark=True)["figure.facecolor"] == "#0b1220"


def test_mpl_use_and_cmap_when_available() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    from andrey.viz import mpl

    mpl.use()
    assert matplotlib.rcParams["axes.prop_cycle"].by_key()["color"][0] == "#3b5bdb"
    assert mpl.cmap().name == "andrey"
    assert mpl.cmap(dark=True).name == "andrey-dark"


_APEX = re.compile(r'<polygon points="([\d.]+),')


def _arrow_apex_x(svg: str) -> float:
    """The x of the first arrowhead apex; its tip sits at the child node's boundary."""
    m = _APEX.search(svg)
    assert m, "no arrowhead polygon found"
    return float(m.group(1))


def test_arrowhead_sits_at_the_downstream_child() -> None:
    # Layering places the child downstream (right) and the arrowhead is drawn at the child. If the
    # orientation branch were inverted, the true child would land upstream and the apex on the LEFT.
    w = 400
    a = viz.draw(_from({(0, 1): (TAIL, ARROW)}, 2, "dag"), width=w)  # arrow at node 1 (child)
    b = viz.draw(_from({(0, 1): (ARROW, TAIL)}, 2, "dag"), width=w)  # arrow at node 0 (child)
    assert _arrow_apex_x(a) > w / 2
    assert _arrow_apex_x(b) > w / 2
    assert a != b  # orientation is reflected in the output


def test_self_loop_renders_a_path() -> None:
    m = np.zeros((2, 2), np.int8)
    m[0, 0] = ARROW  # a directed self-loop on node 0
    m[0, 1] = TAIL
    m[1, 0] = ARROW  # plus 0 -> 1 so the layout is not a lone node
    s = GraphStructure.from_numpy(m, kind="digraph", allow_self_loops=True)
    svg = viz.draw(s)
    minidom.parseString(svg)
    assert "<path" in svg  # the loop arc


def test_empty_and_single_node() -> None:
    empty = GraphStructure.from_numpy(np.zeros((0, 0), np.int8), kind="dag")
    assert viz.draw(empty).startswith("<svg")
    single = GraphStructure.from_numpy(np.zeros((1, 1), np.int8), kind="dag", labels=["A"])
    svg = viz.draw(single, width=200, height=200)
    minidom.parseString(svg)
    cx = float(re.search(r'<circle cx="([\d.]+)"', svg).group(1))
    assert abs(cx - 100) < 1  # a lone node is centered (width / 2), not at the top


def test_labels_are_escaped() -> None:
    s = _from({(0, 1): (TAIL, ARROW)}, 2, "dag", labels=["<x>", "A&B"])
    svg = viz.draw(s)
    assert "&lt;x&gt;" in svg and "A&amp;B" in svg


@pytest.mark.parametrize("labels", [("X", "Y"), ("x0", "x1")])
@pytest.mark.parametrize("marks", [(TAIL, TAIL), (CIRCLE, ARROW), (ARROW, ARROW)])
@pytest.mark.parametrize("dark", [False, True])
def test_short_labels_stay_inside_nodes_on_vertical_edges(labels, marks, dark) -> None:
    structure = _from({(0, 1): marks}, 2, "pag", labels=labels)
    vertical = {0: (0.5, 0.0), 1: (0.5, 1.0)}
    document = minidom.parseString(viz.draw(structure, dark=dark, positions=vertical))
    centers = {
        (float(node.getAttribute("cx")), float(node.getAttribute("cy")))
        for node in document.getElementsByTagName("circle")
        if float(node.getAttribute("r")) == 13
    }
    assert len(centers) == 2
    assert len({x for x, _ in centers}) == 1  # vertical edge, with marks below the upper node
    texts = document.getElementsByTagName("text")
    assert {text.firstChild.data for text in texts} == set(labels)
    for text in texts:
        assert (float(text.getAttribute("x")), float(text.getAttribute("y"))) in centers
        assert text.getAttribute("dominant-baseline") == "central"


@pytest.mark.parametrize("labels", [("Exposure", "Outcome"), ("X", "Y", "Age")])
@pytest.mark.parametrize("dark", [False, True])
def test_any_long_label_places_all_labels_below_nodes(labels, dark) -> None:
    pairs = {(i, i + 1): (TAIL, ARROW) for i in range(len(labels) - 1)}
    structure = _from(pairs, len(labels), "dag", labels=labels)
    document = minidom.parseString(viz.draw(structure, dark=dark))
    nodes = document.getElementsByTagName("circle")
    texts = document.getElementsByTagName("text")
    assert {text.firstChild.data for text in texts} == set(labels)
    for text in texts:
        node = next(n for n in nodes if n.getAttribute("cx") == text.getAttribute("x"))
        bottom = float(node.getAttribute("cy")) + float(node.getAttribute("r"))
        assert float(text.getAttribute("y")) > bottom
        assert text.getAttribute("fill") == (viz.DARK if dark else viz.LIGHT).muted


@pytest.mark.parametrize("n,baseline", [(100, "central"), (101, "")])
def test_numeric_labels_use_one_placement_for_the_figure(n, baseline) -> None:
    pairs = {(i, i + 1): (TAIL, ARROW) for i in range(n - 1)}
    document = minidom.parseString(viz.draw(_from(pairs, n, "dag")))
    texts = document.getElementsByTagName("text")
    assert {text.firstChild.data for text in texts} == {str(i) for i in range(n)}
    assert {text.getAttribute("dominant-baseline") for text in texts} == {baseline}


def test_transparent_background() -> None:
    s = _from({(0, 1): (TAIL, ARROW)}, 2, "dag")
    assert "<rect" in viz.draw(s)  # opaque paper by default
    assert "<rect" not in viz.draw(s, background=False)


def test_custom_dimensions() -> None:
    svg = viz.draw(_from({(0, 1): (TAIL, ARROW)}, 2, "dag"), width=300, height=200)
    assert 'viewBox="0 0 300 200"' in svg


def test_large_chain_keeps_nodes_clear() -> None:
    # A 30-node chain must not collapse: with a fixed canvas nodes would pack ~20px apart while
    # they are 26px wide. Auto-sizing keeps adjacent centers >= the node diameter apart.
    pairs = {(i, i + 1): (TAIL, ARROW) for i in range(29)}
    svg = viz.draw(_from(pairs, 30, "dag"))
    xs = sorted(float(x) for x in re.findall(r'<circle cx="([\d.]+)"', svg))
    gaps = [b - a for a, b in zip(xs, xs[1:])]
    assert gaps and min(gaps) >= 2 * 13  # NODE_R = 13 -> diameter 26


def _strokes(document) -> list[list[tuple[float, float]]]:
    """Sample every drawn edge stroke (lines and quadratic arcs) as a polyline."""
    strokes = []
    for line in document.getElementsByTagName("line"):
        x1, y1, x2, y2 = (float(line.getAttribute(k)) for k in ("x1", "y1", "x2", "y2"))
        strokes.append([(x1, y1), (x2, y2)])
    for path in document.getElementsByTagName("path"):
        d = path.getAttribute("d").split()
        if d[3] != "Q":
            continue  # a self-loop
        (ax, ay), (mx, my), (bx, by) = [(float(d[k]), float(d[k + 1])) for k in (1, 4, 6)]
        ts = [k / 64 for k in range(65)]
        strokes.append(
            [
                (
                    (1 - t) ** 2 * ax + 2 * t * (1 - t) * mx + t**2 * bx,
                    (1 - t) ** 2 * ay + 2 * t * (1 - t) * my + t**2 * by,
                )
                for t in ts
            ]
        )
    return strokes


def _centers(document) -> list[tuple[float, float]]:
    return [
        (float(c.getAttribute("cx")), float(c.getAttribute("cy")))
        for c in document.getElementsByTagName("circle")
        if float(c.getAttribute("r")) == 13
    ]


def _nearest_stroke_distance(stroke, point) -> float:
    px, py = point
    best = np.inf
    for (x1, y1), (x2, y2) in zip(stroke, stroke[1:]):
        sx, sy = x2 - x1, y2 - y1
        t = np.clip(((px - x1) * sx + (py - y1) * sy) / (sx * sx + sy * sy or 1.0), 0, 1)
        best = min(best, float(np.hypot(x1 + t * sx - px, y1 + t * sy - py)))
    return best


def _crossed_nodes(svg: str) -> int:
    """Count edge strokes that pass inside a node they do not join."""
    document = minidom.parseString(svg)
    centers = _centers(document)
    crossed = 0
    for stroke in _strokes(document):
        ends = {
            min(range(len(centers)), key=lambda k: np.hypot(*np.subtract(centers[k], stroke[e])))
            for e in (0, -1)
        }
        others = [c for k, c in enumerate(centers) if k not in ends]
        crossed += any(_nearest_stroke_distance(stroke, c) < 13 + 1 for c in others)
    return crossed


def test_edges_bend_around_nodes_they_would_cross() -> None:
    # 1 -> 3 <- 2 with 0 -- 1, 0 -- 2, 0 -- 3: the layering stacks 0, 1, 2 in one column, so a
    # straight 0 -- 2 edge would run through node 1 and draw a false 1 -- 2 adjacency.
    shield = _from(
        {
            (1, 3): (TAIL, ARROW),
            (2, 3): (TAIL, ARROW),
            (0, 1): (TAIL, TAIL),
            (0, 2): (TAIL, TAIL),
            (0, 3): (TAIL, TAIL),
        },
        4,
        "cpdag",
    )
    svg = viz.draw(shield)
    assert "<path" in svg
    assert _crossed_nodes(svg) == 0


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("d", [8, 20])
def test_random_dags_draw_no_edge_through_a_node(seed: int, d: int) -> None:
    rng = np.random.default_rng(seed)
    pairs = {
        (i, j): (TAIL, ARROW) for i in range(d) for j in range(i + 1, d) if rng.random() < 3 / d
    }
    assert _crossed_nodes(viz.draw(_from(pairs, d, "dag"))) == 0


def test_straight_edges_stay_straight() -> None:
    chain = _from({(0, 1): (TAIL, ARROW), (1, 2): (TAIL, ARROW)}, 3, "dag")
    assert "<path" not in viz.draw(chain)


@pytest.mark.parametrize(
    "structure",
    [
        _from({(0, 1): (TAIL, ARROW), (1, 2): (TAIL, TAIL)}, 3, "cpdag"),
        _from({(i, (i + 1) % 12): (CIRCLE, CIRCLE) for i in range(12)}, 12, "pag"),  # a ring
    ],
)
def test_layout_is_the_default_placement(structure) -> None:
    assert viz.draw(structure, positions=viz.layout(structure)) == viz.draw(structure)


def test_given_positions_never_overlap_nodes() -> None:
    crowded = {k: (0.5 + 0.02 * k, 0.5) for k in range(6)}  # 2% of the width apart
    centers = _centers(minidom.parseString(viz.draw(_from({}, 6, "dag"), positions=crowded)))
    assert min(abs(a[0] - b[0]) for a, b in zip(centers, centers[1:])) >= 52 - 0.1


def test_positions_keep_nodes_in_place_across_structures() -> None:
    before = _from({(0, 1): (TAIL, TAIL), (1, 2): (TAIL, TAIL)}, 3, "cpdag")
    after = _from({(0, 1): (TAIL, ARROW), (1, 2): (TAIL, ARROW)}, 3, "cpdag")
    fixed = {0: (0.0, 0.0), 1: (0.5, 1.0), 2: (1.0, 0.0)}
    placed = [_centers(minidom.parseString(viz.draw(s, positions=fixed))) for s in (before, after)]
    assert placed[0] == placed[1]
    assert placed[0] != _centers(minidom.parseString(viz.draw(after)))


@pytest.mark.parametrize("positions", [{0: (0, 0)}, {0: (0, 0), 1: (1.5, 0)}])
def test_positions_must_place_every_node_in_the_unit_square(positions) -> None:
    with pytest.raises(ValueError, match="unit square"):
        viz.draw(_from({(0, 1): (TAIL, ARROW)}, 2, "dag"), positions=positions)


@pytest.mark.parametrize("marks", [(TAIL, ARROW), (CIRCLE, CIRCLE), (TAIL, TAIL)])
def test_highlight_rings_the_named_endpoint(marks) -> None:
    s = _from({(0, 1): marks}, 2, "pag")
    document = minidom.parseString(viz.draw(s, highlight=[(1, 0)]))
    rings = [c for c in document.getElementsByTagName("circle") if c.getAttribute("class")]
    assert [r.getAttribute("class") for r in rings] == ["highlight"]
    ring = (float(rings[0].getAttribute("cx")), float(rings[0].getAttribute("cy")))
    near, far = _centers(document)[1], _centers(document)[0]
    assert np.hypot(*np.subtract(ring, near)) < np.hypot(*np.subtract(ring, far))
    assert rings[0].getAttribute("stroke") == viz.LIGHT.accent
    assert "highlight" not in viz.draw(s)


def test_highlight_needs_an_edge_between_two_nodes() -> None:
    with pytest.raises(ValueError, match="no endpoint"):
        viz.draw(_from({(0, 1): (TAIL, ARROW)}, 3, "dag"), highlight=[(0, 2)])
    m = np.zeros((2, 2), np.int8)
    m[0, 0], m[0, 1], m[1, 0] = ARROW, TAIL, ARROW
    looped = GraphStructure.from_numpy(m, kind="digraph", allow_self_loops=True)
    with pytest.raises(ValueError, match="no endpoint"):
        viz.draw(looped, highlight=[(0, 0)])


def test_two_cycle_arcs_bend_around_nodes_between_them() -> None:
    # 0 <-> 2 with node 1 just below their midpoint, where the default lower arc would pass.
    cycle = _from({(0, 2): (ARROW, ARROW), (0, 1): (TAIL, ARROW)}, 3, "digraph")
    line = {0: (0.0, 0.5), 1: (0.5, 0.62), 2: (1.0, 0.5)}
    svg = viz.draw(cycle, positions=line)
    assert svg.count("<path") == 2
    assert _crossed_nodes(svg) == 0


def test_mpl_context_scopes_the_style() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    from andrey.viz import mpl

    with mpl.context(dark=True):
        assert matplotlib.rcParams["axes.prop_cycle"].by_key()["color"][0] == "#6e86f5"


def _chain(n: int, **kw) -> GraphStructure:
    return _from({(i, i + 1): (TAIL, ARROW) for i in range(n - 1)}, n, "dag", **kw)


def _random_dag(seed: int, n: int = 12) -> GraphStructure:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    pairs = {}
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < 0.25:
                pairs[(int(order[a]), int(order[b]))] = (TAIL, ARROW)  # order[a] -> order[b]
    return _from(pairs, n, "dag")


def _parents_and_children(structure: GraphStructure) -> list[tuple[int, int]]:
    return [
        (int(r["i"]), int(r["j"])) if int(r["mark_j"]) == ARROW else (int(r["j"]), int(r["i"]))
        for r in structure.to_edges()
    ]


@pytest.mark.parametrize("seed", range(3))
def test_graphviz_runs_causes_left_to_right_or_as_asked(seed) -> None:
    pytest.importorskip("pygraphviz")
    dag = _random_dag(seed)
    across = viz.layout(dag)  # "auto" is dot when pygraphviz is installed
    assert across == viz.layout(dag, engine="dot") == viz.layout(dag)  # and it is deterministic
    down = viz.layout(dag, options={"rankdir": "TB"})
    for parent, child in _parents_and_children(dag):
        assert across[parent][0] < across[child][0]
        assert down[parent][1] < down[child][1]


@pytest.mark.parametrize("engine", ["neato", "fdp", "sfdp", "circo", "twopi"])
def test_every_graphviz_engine_places_every_node_in_the_unit_square(engine) -> None:
    pytest.importorskip("pygraphviz")
    positions = viz.layout(_random_dag(0), engine=engine)
    assert sorted(positions) == list(range(12))
    assert all(0 <= v <= 1 for point in positions.values() for v in point)


def test_to_graphviz_keeps_every_endpoint_mark() -> None:
    pytest.importorskip("pygraphviz")
    pag = _from({(0, 1): (TAIL, ARROW), (1, 2): (CIRCLE, ARROW), (2, 3): (TAIL, TAIL)}, 4, "pag")
    graph = viz.to_graphviz(pag, {"rankdir": "TB"})
    assert graph.graph_attr["rankdir"] == "TB"
    edges = {(a, b): dict(graph.get_edge(a, b).attr) for a, b in graph.edges()}
    assert edges[("0", "1")].get("dir") in (None, "")  # a plain arrow
    assert edges[("1", "2")]["dir"] == "both"
    assert (edges[("1", "2")]["arrowtail"], edges[("1", "2")]["arrowhead"]) == ("odot", "normal")
    assert edges[("2", "3")]["dir"] == "none"
    assert graph.get_node("0").attr["label"] == "0"


def test_to_graphviz_draws_a_self_loop_as_one_directed_edge() -> None:
    pytest.importorskip("pygraphviz")
    m = np.zeros((2, 2), np.int8)
    m[0, 0], m[0, 1], m[1, 0] = ARROW, TAIL, ARROW
    looped = GraphStructure.from_numpy(m, kind="digraph", allow_self_loops=True)
    loop = viz.to_graphviz(looped).get_edge("0", "0")
    assert loop.attr.get("dir") in (None, "")


def test_without_pygraphviz_the_default_is_the_builtin_layout(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pygraphviz", None)  # import now fails
    dag = _random_dag(0)
    assert viz.layout(dag) == viz.layout(dag, engine="builtin")
    with pytest.raises(ImportError, match=r"andrey-core\[viz\]"):
        viz.layout(dag, engine="dot")
    with pytest.raises(ImportError, match=r"andrey-core\[viz\]"):
        viz.to_graphviz(dag)


def test_an_unknown_engine_or_options_for_the_builtin_layout_are_refused() -> None:
    with pytest.raises(ValueError, match="engine must be one of"):
        viz.layout(_chain(3), engine="graphviz")
    with pytest.raises(ValueError, match="no options"):
        viz.layout(_chain(3), engine="builtin", options={"rankdir": "TB"})


@pytest.mark.parametrize("engine", ["builtin", "dot"])
def test_long_labels_stay_apart_and_inside_the_canvas(engine) -> None:
    if engine == "dot":
        pytest.importorskip("pygraphviz")
    labels = ["HYPOVOLEMIA", "LVEDVOLUME", "STROKEVOLUME", "ERRLOWOUTPUT"]
    svg = viz.draw(_chain(4, labels=labels), engine=engine)
    width = float(re.search(r'viewBox="0 0 ([\d.]+) ', svg).group(1))
    xs = sorted(float(x) for x in re.findall(r'<circle cx="([\d.]+)"', svg))
    half = [len(label) * 7.0 / 2 for label in labels]  # the drawing's widest label character
    assert xs[0] - max(half) >= 0 and xs[-1] + max(half) <= width
    assert all(b - a >= 2 * max(half) for a, b in zip(xs, xs[1:]))


def test_a_canvas_narrower_than_its_labels_is_widened_not_mirrored() -> None:
    # A long label needs more side margin than a 100-pixel canvas has.
    row = {0: (0.0, 0.5), 1: (1.0, 0.5)}
    graph = _chain(2, labels=["a_very_long_variable_name", "b"])
    document = minidom.parseString(viz.draw(graph, positions=row, width=100))
    width = float(document.documentElement.getAttribute("viewBox").split()[2])
    (x0, _), (x1, _) = _centers(document)
    assert width > 100
    assert 0 < x0 < x1 < width  # node 0 stays on the left, both inside the canvas


def test_a_fixed_canvas_keeps_the_layout_proportions() -> None:
    row = {0: (0.0, 0.5), 1: (1.0, 0.5)}
    centers = _centers(
        minidom.parseString(viz.draw(_chain(2), positions=row, width=300, height=300))
    )
    assert {y for _, y in centers} == {150.0}  # centered, not stretched


@pytest.mark.parametrize(
    ("marks", "kind", "message"),
    [((TAIL, ARROW), "foo", "kind"), ((9, 9), "dag", "marks"), ((0, ARROW), "pag", "marks")],
)
def test_classify_edge_rejects_unknown_input(marks, kind, message):
    with pytest.raises(ValueError, match=message):
        classify_edge(*marks, kind)


@pytest.mark.parametrize("render", [viz.draw, viz.layout])
def test_a_temporal_structure_is_drawn_one_graph_at_a_time(render):
    empty = GraphStructure.from_numpy(np.zeros((2, 2), int))
    temporal = TemporalStructure.from_lag_graphs([empty, empty])
    with pytest.raises(TypeError, match=r"lag\(k\).*summary_graph\(\)"):
        render(temporal)
