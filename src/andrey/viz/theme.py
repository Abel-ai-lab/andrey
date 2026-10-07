"""Design tokens for ``andrey.viz`` -- the package-native cobalt palette.

Concrete colors per theme; a :class:`Palette` is selected at draw time. The rationale is in
``DESIGN.md``. The governing idea: a graphite backbone for resolved
(directed) edges, with color reserved for uncertainty; marks are drawn by shape, so color is a
redundant, colorblind-safe channel.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    """The colors ``andrey.viz`` draws with, for one theme.

    ``andrey.viz.LIGHT`` and ``andrey.viz.DARK`` are the built-in palettes. To match a page with
    its own colors, build a ``Palette`` and pass it to :func:`~andrey.viz.draw` as ``palette=``.
    Every color is a CSS color string, such as ``"#3b5bdb"``. Each parameter is also a read-only
    attribute of the same name.

    Parameters
    ----------
    paper : str
        The background: the canvas of :func:`~andrey.viz.draw` (unless ``background=False``) and
        of a styled chart. Latent nodes are filled with it, so they read as hollow.
    surface : str
        The fill of an observed node and of a circle endpoint mark.
    surface_2 : str
        A second panel color for a page around the figures. :func:`~andrey.viz.draw` and
        :mod:`andrey.viz.mpl` do not use it.
    ink : str
        The ring and inside label of an observed node, and a chart's text, titles, and axis labels.
    muted : str
        The dashed ring and inside label of a latent node, every label drawn below its node, and a
        chart's tick labels.
    line : str
        A chart's axis lines and grid lines. :func:`~andrey.viz.draw` does not use it.
    accent : str
        The ring and mark of each endpoint named in the ``highlight`` of
        :func:`~andrey.viz.draw`.
    edge_directed : str
        Directed edges (``->``), including both edges of a 2-cycle and self-loops. A neutral gray,
        so edges with an uncertain orientation stand out.
    edge_undirected : str
        Undirected edges (``--``): in a CPDAG, an orientation the data leave open.
    edge_bidirected : str
        Bidirected edges (``<->``): in a PAG, a probable latent common cause.
    edge_partial : str
        Edges with a circle end (``o->``, ``-o``, ``o-o``) in a PAG.
    categorical : tuple of str
        Eight colorblind-safe colors for chart series, in the order to use them;
        :func:`andrey.viz.mpl.cycle` returns them.
    ramp : tuple of str
        The colors of :func:`andrey.viz.mpl.cmap`, from the lowest value to the highest. The lowest
        is nearest the background, so low values fade into the page in either theme.
    """

    paper: str
    surface: str
    surface_2: str
    ink: str
    muted: str
    line: str
    accent: str
    edge_directed: str
    edge_undirected: str
    edge_bidirected: str
    edge_partial: str
    categorical: tuple[str, ...]
    ramp: tuple[str, ...]  # sequential cobalt ramp (confidence / weight), low value -> high


LIGHT = Palette(
    paper="#fafbfc",
    surface="#ffffff",
    surface_2="#eef1f6",
    ink="#182233",
    muted="#5c6675",
    line="#d9dde6",
    accent="#3b5bdb",
    edge_directed="#3a4658",
    edge_undirected="#e0982b",
    edge_bidirected="#d6495b",
    edge_partial="#7c5cd8",
    categorical=(
        "#3b5bdb",
        "#e0982b",
        "#2e9e5b",
        "#e5573f",
        "#7c5cd8",
        "#2f9bd0",
        "#c74e9b",
        "#c9a227",
    ),
    ramp=("#e7ebfb", "#bcc8f4", "#7c93ee", "#3b5bdb", "#1f338f"),
)

DARK = Palette(
    paper="#0b1220",
    surface="#121a2b",
    surface_2="#182236",
    ink="#e8ecf5",
    muted="#8d97ab",
    line="#25314c",
    accent="#6e86f5",
    edge_directed="#98a4bd",
    edge_undirected="#f0b357",
    edge_bidirected="#f0687a",
    edge_partial="#a08cf0",
    categorical=(
        "#6e86f5",
        "#f0b357",
        "#47b673",
        "#f06e58",
        "#a08cf0",
        "#4fb6e6",
        "#df79bf",
        "#d9b94a",
    ),
    ramp=("#141d33", "#293a6b", "#4d63c8", "#6e86f5", "#aebdf7"),
)

# Edge type -> the Palette attribute that colors it. A 2-cycle is directed feedback, so graphite.
EDGE_COLOR = {
    "directed": "edge_directed",
    "twocycle": "edge_directed",
    "undirected": "edge_undirected",
    "bidirected": "edge_bidirected",
    "partial": "edge_partial",
}
