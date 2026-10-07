"""Graphviz placement for ``andrey.viz``, through pygraphviz (the ``viz`` extra).

Graphviz only places the nodes; ``andrey.viz`` still draws them. Each node is laid out as a box the
size of what the drawing puts there, its circle and the label under it, so Graphviz leaves room
for the labels.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from andrey.core.structure import ARROW, CIRCLE, TAIL, GraphStructure

from . import _svg

if TYPE_CHECKING:
    import pygraphviz

ENGINES = ("dot", "neato", "fdp", "sfdp", "circo", "twopi", "osage", "patchwork")
INSTALL = 'Graphviz layouts need pygraphviz: pip install "andrey-core[viz]".'
_ARROWS = {ARROW: "normal", CIRCLE: "odot"}  # a tail draws no mark


def available() -> bool:
    """Whether pygraphviz imports."""
    try:
        import pygraphviz  # noqa: F401
    except ImportError:
        return False
    return True


def to_graphviz(
    structure: GraphStructure, options: Mapping[str, object] | None = None
) -> pygraphviz.AGraph:
    """Return ``structure`` as a Graphviz graph.

    Parameters
    ----------
    structure : GraphStructure
        Any kind. Node ``k`` is named ``"k"`` and carries its label as ``label``; each edge keeps
        its endpoint marks as Graphviz arrow shapes: ``normal`` for an arrowhead, ``odot`` for a
        circle, none for a tail.
    options : mapping of str to object or None, default=None
        Graphviz graph attributes, such as ``{"rankdir": "TB", "nodesep": 0.5}``. Causes run left
        to right (``rankdir="LR"``) unless ``options`` says otherwise.

    Returns
    -------
    pygraphviz.AGraph
        A directed graph. A directed edge runs from tail to arrowhead; an undirected edge has
        ``dir="none"``; any other edge has ``dir="both"``, with ``arrowhead`` and ``arrowtail``
        set from its marks; a ``digraph`` 2-cycle becomes two directed edges. Render or lay it out
        with pygraphviz, or pass it to any Graphviz tool through ``to_string()``.

    Raises
    ------
    ImportError
        If pygraphviz is not installed.
    """
    try:
        import pygraphviz
    except ImportError as err:
        raise ImportError(INSTALL) from err
    n = int(structure.n_nodes)
    labels = list(structure.labels) if structure.labels is not None else [str(k) for k in range(n)]
    graph = pygraphviz.AGraph(directed=True, strict=False)
    graph.graph_attr["rankdir"] = "LR"
    graph.graph_attr.update({key: str(value) for key, value in (options or {}).items()})
    graph.node_attr["shape"] = "circle"
    for k in range(n):
        graph.add_node(str(k), label=labels[k])
    for row in structure.to_edges():
        i, j = int(row["i"]), int(row["j"])
        mark_i, mark_j = int(row["mark_i"]), int(row["mark_j"])
        if i == j:  # a self-loop is one directed edge
            graph.add_edge(str(i), str(i))
        elif (mark_i, mark_j) == (TAIL, ARROW):
            graph.add_edge(str(i), str(j))
        elif (mark_i, mark_j) == (ARROW, TAIL):
            graph.add_edge(str(j), str(i))
        elif (mark_i, mark_j) == (TAIL, TAIL):
            graph.add_edge(str(i), str(j), dir="none")
        elif structure.kind == "digraph":  # arrowheads at both ends: a 2-cycle
            graph.add_edge(str(i), str(j))
            graph.add_edge(str(j), str(i))
        else:
            head, tail = _ARROWS.get(mark_j, "none"), _ARROWS.get(mark_i, "none")
            graph.add_edge(str(i), str(j), dir="both", arrowhead=head, arrowtail=tail)
    return graph


def positions(
    structure: GraphStructure, engine: str, options: Mapping[str, object] | None = None
) -> dict[int, tuple[float, float]]:
    """Each node's center as Graphviz ``engine`` places it, in points, ``y`` downward."""
    graph = to_graphviz(structure, options)
    n = int(structure.n_nodes)
    labels_inside = all(len(graph.get_node(str(k)).attr["label"]) <= 2 for k in range(n))
    graph.node_attr.update(shape="box", fixedsize="true")
    for k in range(n):
        node = graph.get_node(str(k))
        width, height = _svg.footprint(node.attr["label"], labels_inside)
        node.attr.update(label="", width=width / 72, height=height / 72)
    graph.layout(prog=engine)
    placed = {}
    for k in range(n):
        x, y = (float(v) for v in graph.get_node(str(k)).attr["pos"].split(",")[:2])
        placed[k] = (x, -y)  # Graphviz's y grows upward
    return placed
