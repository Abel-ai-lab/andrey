"""Graphviz placement for ``andrey.viz``, through pygraphviz (the ``viz`` extra).

Graphviz only places the nodes; ``andrey.viz`` still draws them. Each node is laid out as a box the
size of what the drawing puts there, its circle and the label under it, so Graphviz leaves room
for the labels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from andrey.core.structure import GraphStructure

from . import _svg

if TYPE_CHECKING:
    import pygraphviz

ENGINES = ("dot", "neato", "fdp", "sfdp", "circo", "twopi", "osage", "patchwork")
INSTALL = 'Graphviz layouts need pygraphviz: pip install "andrey-core[viz]".'


def available() -> bool:
    """Whether pygraphviz imports."""
    try:
        import pygraphviz  # noqa: F401
    except ImportError:
        return False
    return True


def intervened(targets: Iterable[int | str], labels: list[str]) -> frozenset[int]:
    """Return the positions of ``targets``, each named by label or by 0-based position."""
    marked = set()
    for name in targets:
        if isinstance(name, str) and name in labels:
            marked.add(labels.index(name))
        elif (
            isinstance(name, (int, np.integer))
            and not isinstance(name, bool)
            and 0 <= int(name) < len(labels)
        ):
            marked.add(int(name))
        else:
            raise ValueError(
                f"targets names {name!r}, which is not a variable's label or position."
            )
    return frozenset(marked)


def to_graphviz(
    structure: GraphStructure,
    options: Mapping[str, object] | None = None,
    *,
    targets: Iterable[int | str] = (),
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
    targets : iterable of int or str, default=()
        The intervened variables, named by label or by 0-based position, marked as
        :func:`~andrey.viz.draw` marks them. Each is a ``square`` node, and each arrowhead into it
        is ``normaltee``, an arrowhead with a bar behind it. A target no arrowhead points into is
        marked by its square alone. The default marks none.

    Returns
    -------
    pygraphviz.AGraph
        A directed graph. A directed edge runs from tail to arrowhead; an undirected edge has
        ``dir="none"``; any other edge has ``dir="both"``, with ``arrowhead`` and ``arrowtail``
        set from its marks; a ``digraph`` 2-cycle becomes two directed edges. Render or lay it out
        with pygraphviz, or pass it to any Graphviz tool through ``to_string()``.

    Raises
    ------
    ValueError
        If ``targets`` names a variable the structure does not have.
    ImportError
        If pygraphviz is not installed.
    """
    try:
        import pygraphviz
    except ImportError as err:
        raise ImportError(INSTALL) from err
    n = int(structure.n_nodes)
    labels = list(structure.labels) if structure.labels is not None else [str(k) for k in range(n)]
    marked = intervened(targets, labels)
    graph = pygraphviz.AGraph(directed=True, strict=False)
    graph.graph_attr["rankdir"] = "LR"
    graph.graph_attr.update({key: str(value) for key, value in (options or {}).items()})
    graph.node_attr["shape"] = "circle"
    for k in range(n):
        graph.add_node(str(k), label=labels[k], **({"shape": "square"} if k in marked else {}))
    for i, j, attrs in structure._dot_edges():
        # j holds an arrowhead unless the edge is undirected; i holds one only with dir="both".
        head = attrs.get("arrowhead", "none" if attrs.get("dir") == "none" else "normal")
        if i != j and j in marked and head == "normal":
            attrs["arrowhead"] = "normaltee"
        if i != j and i in marked and attrs.get("arrowtail") == "normal":
            attrs["arrowtail"] = "normaltee"
        graph.add_edge(str(i), str(j), **attrs)
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
