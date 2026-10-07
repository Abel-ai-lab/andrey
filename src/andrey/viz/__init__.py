"""One visual grammar for causal graphs.

:func:`draw` renders a :class:`~andrey.core.structure.GraphStructure` as a self-contained SVG.
Marks are drawn by shape (arrowhead, hollow circle, tail), and color is kept for uncertain edges
and latent nodes, so a drawing reads without color. Graphviz places the nodes when pygraphviz (the
``viz`` extra) is installed, and a built-in layering otherwise; edges bend around nodes.

Examples
--------
>>> import numpy as np
>>> from andrey.core.structure import GraphStructure, TAIL, ARROW
>>> import andrey.viz
>>> M = np.zeros((2, 2), np.int8); M[0, 1] = TAIL; M[1, 0] = ARROW  # X -> Y
>>> svg = andrey.viz.draw(GraphStructure.from_numpy(M, kind="dag", labels=["X", "Y"]))
>>> svg.startswith("<svg") and "polygon" in svg
True
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np

from andrey.core.structure import ARROW, CIRCLE, LATENT, TAIL, GraphStructure

from . import _graphviz, _layout, _svg
from ._graphviz import to_graphviz
from .theme import DARK, LIGHT, Palette

__all__ = ["draw", "layout", "to_graphviz", "classify_edge", "SVG", "Palette", "LIGHT", "DARK"]
ENGINES = ("auto", "builtin", *_graphviz.ENGINES)


class SVG(str):
    """An SVG document that renders itself in notebooks and doc pages.

    A plain :class:`str` subclass, so it slices, concatenates, and writes to a file exactly like the
    markup it holds. The only addition is ``_repr_html_``, which Jupyter calls to display the
    graph inline instead of printing several kilobytes of markup.

    Examples
    --------
    >>> from andrey.viz import SVG
    >>> svg = SVG("<svg></svg>")
    >>> isinstance(svg, str) and svg._repr_html_() == svg
    True
    """

    __slots__ = ()

    def _repr_html_(self) -> str:
        """Return the markup itself: an SVG document is already valid inline HTML."""
        return str(self)


# The smallest canvas a default drawing gets, in pixels.
_MIN_WIDTH, _MIN_HEIGHT = 360, 240


def _scale(unit: dict[int, tuple[float, float]], widths: list[float]) -> float:
    """Pixels per unit of ``unit`` that keep every pair of nodes clear of each other.

    Two nodes are clear when they are a column apart across, wide enough for both labels, or a row
    apart down.
    """
    if len(unit) < 2:
        return 1.0
    points = np.array([unit[k] for k in range(len(unit))])
    dx = np.abs(points[:, None, 0] - points[None, :, 0])
    dy = np.abs(points[:, None, 1] - points[None, :, 1])
    w = np.asarray(widths)
    across = np.maximum(_layout.COLUMN, (w[:, None] + w[None, :]) / 2 + 12)
    with np.errstate(divide="ignore"):
        need = np.minimum(
            np.where(dx > 0, across / dx, np.inf), np.where(dy > 0, _layout.ROW / dy, np.inf)
        )
    need = need[np.triu_indices(len(unit), 1)]
    need = need[np.isfinite(need)]  # nodes on the same point cannot be told apart by scaling
    return float(need.max()) if need.size else 1.0


def classify_edge(mark_i: int, mark_j: int, kind: str) -> str:
    """Name the edge type from its endpoint-mark pair and the graph kind.

    The name selects the edge's color and mark shapes: ``directed`` is the graphite backbone, and
    the rest carry the palette's uncertainty colors.

    Parameters
    ----------
    mark_i, mark_j : int
        The endpoint marks at nodes ``i`` and ``j``: ``TAIL``, ``ARROW``, or ``CIRCLE`` from
        :mod:`andrey.core.structure`.
    kind : str
        The graph kind (``dag`` / ``cpdag`` / ``pag`` / ``digraph``). Only ``ARROW``/``ARROW``
        reads differently across kinds.

    Returns
    -------
    str
        ``"directed"`` (``->``), ``"undirected"`` (``--``), ``"bidirected"`` (``<->`` outside a
        ``digraph``), ``"twocycle"`` (``<->`` in a ``digraph``: two directed edges), or
        ``"partial"`` (an edge with a circle end: ``o->``, ``-o``, or ``o-o``).

    Notes
    -----
    ``ARROW``/``ARROW`` is kind-relative: a ``bidirected`` edge (latent confounder) under a PAG, a
    ``twocycle`` (feedback, two directed edges) under a directed graph. The two are drawn
    differently on purpose, so a reader never mistakes feedback for a hidden common cause.

    Examples
    --------
    >>> from andrey.core.structure import ARROW, TAIL
    >>> from andrey.viz import classify_edge
    >>> classify_edge(TAIL, ARROW, "dag")
    'directed'
    >>> classify_edge(ARROW, ARROW, "pag"), classify_edge(ARROW, ARROW, "digraph")
    ('bidirected', 'twocycle')
    """
    marks = {int(mark_i), int(mark_j)}
    if marks == {TAIL, ARROW}:
        return "directed"
    if int(mark_i) == TAIL and int(mark_j) == TAIL:
        return "undirected"
    if int(mark_i) == ARROW and int(mark_j) == ARROW:
        return "twocycle" if kind == "digraph" else "bidirected"
    if CIRCLE in marks:
        return "partial"
    return "directed"


def _endpoints(structure: GraphStructure) -> list[tuple[int, int, int, int, str]]:
    """List each edge as ``(i, j, mark_i, mark_j, edge_type)``."""
    edges = []
    for row in structure.to_edges():
        i, j = int(row["i"]), int(row["j"])
        mi, mj = int(row["mark_i"]), int(row["mark_j"])
        edges.append((i, j, mi, mj, classify_edge(mi, mj, structure.kind)))
    return edges


def _builtin(n: int, edges: list[tuple[int, int, int, int, str]]) -> dict[int, tuple[float, float]]:
    """Layered by the directed edges, or on a circle."""
    directed: list[tuple[int, int]] = []
    for i, j, _, mj, et in edges:
        if i == j:
            continue
        if et == "directed":
            directed.append((i, j) if mj == ARROW else (j, i))  # arrow end is the child
        elif et == "twocycle":
            directed.append((i, j))
            directed.append((j, i))
    placed = _layout.layered(n, directed) if directed else None
    return _layout.circular(n) if placed is None else placed


def layout(
    structure: GraphStructure, engine: str = "auto", options: Mapping[str, object] | None = None
) -> dict[int, tuple[float, float]]:
    """Return the node positions :func:`draw` uses when none are given.

    Pass the result as ``positions=`` to draw several structures over the same nodes, such as an
    estimate and its truth, with every node in the same place.

    Parameters
    ----------
    structure : GraphStructure
        Any kind.
    engine : str, default="auto"
        ``"dot"`` lays the nodes out in ranks along the directed edges, causes on the left;
        ``"neato"``, ``"fdp"``, ``"sfdp"``, ``"circo"``, ``"twopi"``, ``"osage"``, and
        ``"patchwork"`` are Graphviz's other layouts. These need pygraphviz, the ``viz`` extra.
        ``"builtin"`` needs nothing: nodes in layers along the directed edges, or on a circle when
        there are none or they form a cycle. ``"auto"`` is ``"dot"`` when pygraphviz is installed,
        and ``"builtin"`` otherwise.
    options : mapping of str to object or None, default=None
        Graphviz graph attributes, such as ``{"rankdir": "TB"}`` to run causes top to bottom, or
        ``{"nodesep": 1}`` to spread nodes within a rank. The built-in layout takes none.

    Returns
    -------
    dict of int to (float, float)
        Each node's ``(x, y)`` in the unit square, ``y`` downward, in the layout's proportions:
        the longer side spans the square, and the shorter one is centered.

    Raises
    ------
    ValueError
        If ``engine`` is not one of the above, or the built-in layout is given ``options``.
    ImportError
        If a Graphviz engine is named and pygraphviz is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey.core.structure import GraphStructure, TAIL, ARROW
    >>> import andrey.viz
    >>> M = np.zeros((2, 2), np.int8); M[0, 1] = TAIL; M[1, 0] = ARROW  # X -> Y
    >>> andrey.viz.layout(GraphStructure.from_numpy(M, kind="dag"), engine="builtin")
    {0: (0.0, 0.5), 1: (1.0, 0.5)}
    """
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {', '.join(ENGINES)}, not {engine!r}.")
    if engine == "auto":
        engine = "dot" if _graphviz.available() else "builtin"
    if engine == "builtin":
        if options:
            raise ValueError("the built-in layout takes no options; they are Graphviz's.")
        placed = _builtin(int(structure.n_nodes), _endpoints(structure))
    else:
        placed = _graphviz.positions(structure, engine, options)
    return _layout.unit(placed)


def draw(
    structure: GraphStructure,
    *,
    dark: bool = False,
    palette: Palette | None = None,
    width: int | None = None,
    height: int | None = None,
    background: bool = True,
    positions: Mapping[int, tuple[float, float]] | None = None,
    engine: str = "auto",
    engine_options: Mapping[str, object] | None = None,
    highlight: Iterable[tuple[int, int]] = (),
) -> SVG:
    """Render a causal structure to an SVG string.

    Parameters
    ----------
    structure : GraphStructure
        Any kind (``dag`` / ``cpdag`` / ``pag`` / ``digraph``). Its ``.node_types`` flag latent
        variables, which are drawn hollow and dashed. If every label has at most two characters,
        all labels sit inside nodes; otherwise, all labels sit below them.
    dark : bool, default=False
        Select the built-in dark palette instead of the light one. Ignored when ``palette`` is set.
    palette : Palette or None, default=None
        A :class:`Palette` (``andrey.viz.LIGHT`` / ``DARK`` or a custom one); overrides ``dark``.
    width : int or None, default=None
        Canvas width in pixels. ``None`` sizes the canvas to the layout, with every node and label
        clear of the others. Either way the layout keeps its proportions and is centered. The SVG
        is fluid (``width="100%"``); this sets the coordinate system. A width too narrow for the
        widest label and the margins is widened to the narrowest that holds them.
    height : int or None, default=None
        Canvas height; ``None`` sizes it to the layout. A height too short for the margins is
        raised the same way.
    background : bool, default=True
        Paint the canvas with the palette's paper color. Set ``False`` for a transparent SVG to
        composite onto an already-themed page.
    positions : mapping of int to (float, float) or None, default=None
        Each node's ``(x, y)`` in the unit square, ``y`` downward, as :func:`layout` returns.
        ``None`` uses :func:`layout` with ``engine`` and ``engine_options``.
    engine : str, default="auto"
        The layout :func:`layout` runs when ``positions`` is ``None``: ``"dot"`` and Graphviz's
        other layouts, or ``"builtin"``. ``"auto"`` is ``"dot"`` when pygraphviz is installed.
    engine_options : mapping of str to object or None, default=None
        Graphviz graph attributes for the layout, such as ``{"rankdir": "TB"}``.
    highlight : iterable of (int, int), default=()
        Endpoints to ring in the palette's ``accent`` color. ``(i, j)`` rings the mark at node
        ``i`` on the edge between nodes ``i`` and ``j``; add ``(j, i)`` to ring both ends. The
        default rings nothing.

    Returns
    -------
    SVG
        A self-contained SVG document (no external dependencies or references). :class:`SVG` is a
        :class:`str` subclass, so it behaves as the markup everywhere a string is expected, and
        additionally renders inline in notebooks and doc pages.

    Raises
    ------
    ValueError
        If ``positions`` misses a node or places one outside the unit square, ``engine`` is
        unknown, or ``highlight`` names a pair ``(i, j)`` with ``i == j`` or no edge between
        ``i`` and ``j``.
    ImportError
        If a Graphviz engine is named and pygraphviz is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> import andrey
    >>> import andrey.viz
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.5 * rng.standard_normal(500)
    >>> out = andrey.pc(np.column_stack([x, y]))
    >>> svg = andrey.viz.draw(out.structure, dark=True)
    >>> svg.startswith("<svg")
    True
    """
    n = int(structure.n_nodes)
    labels = list(structure.labels) if structure.labels is not None else [str(i) for i in range(n)]
    node_types = structure.node_types
    latent = {k for k in range(n) if node_types is not None and int(node_types[k]) == LATENT}
    edges = _endpoints(structure)

    if positions is None:
        unit = layout(structure, engine, engine_options)
    else:
        unit = {}
        for k in range(n):
            point = positions.get(k)
            if (
                point is None
                or len(point) != 2
                or not all(math.isfinite(v) and 0 <= v <= 1 for v in point)
            ):
                raise ValueError("positions must give every node an (x, y) inside the unit square.")
            unit[k] = (float(point[0]), float(point[1]))
    ends = {(i, j) for i, j, *_ in edges if i != j} | {(j, i) for i, j, *_ in edges if i != j}
    rings = frozenset((int(i), int(j)) for i, j in highlight)
    if not rings <= ends:
        raise ValueError(
            f"highlight names no endpoint of an edge between two nodes: {sorted(rings - ends)}."
        )

    inside = all(len(label) <= 2 for label in labels)
    widths = [_svg.footprint(label, inside)[0] for label in labels]
    pad = _svg.PAD
    # Room for the widest label at either side, so none is cut off at the canvas edge.
    pad_x = max(pad, max(widths, default=0.0) / 2 + 4)
    xs = [unit[k][0] for k in range(n)]
    ys = [unit[k][1] for k in range(n)]
    span_x = max(xs, default=0.0) - min(xs, default=0.0)
    span_y = max(ys, default=0.0) - min(ys, default=0.0)
    fits = []  # the scale each given side allows
    # A side too small for its padding and one padding of drawing is enlarged, so the scale stays
    # positive: a long label widens the side padding past the fixed floor.
    if width is not None:
        width = max(int(width), math.ceil(2 * pad_x + pad))
        fits.append((width - 2 * pad_x) / span_x if span_x else math.inf)
    if height is not None:
        height = max(int(height), 3 * int(pad))
        fits.append((height - 2 * pad) / span_y if span_y else math.inf)
    scale = min(fits, default=math.inf)
    if not math.isfinite(scale):
        scale = _scale(unit, widths)
    if width is None:
        width = max(round(span_x * scale + 2 * pad_x), _MIN_WIDTH)
    if height is None:
        height = max(round(span_y * scale + 2 * pad), _MIN_HEIGHT)
    left = (width - span_x * scale) / 2 - min(xs, default=0.0) * scale
    top = (height - span_y * scale) / 2 - min(ys, default=0.0) * scale
    pos = {k: (left + xs[k] * scale, top + ys[k] * scale) for k in range(n)}
    resolved = palette if palette is not None else (DARK if dark else LIGHT)
    return SVG(
        _svg.render(
            n=n,
            pos=pos,
            edges=edges,
            labels=labels,
            latent=latent,
            width=width,
            height=height,
            palette=resolved,
            background=background,
            highlight=rings,
        )
    )
