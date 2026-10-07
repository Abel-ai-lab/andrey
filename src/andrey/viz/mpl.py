"""Matplotlib styling for ``andrey.viz`` -- the package palette applied to charts.

matplotlib is an optional dependency (the ``viz`` extra: ``pip install 'andrey-core[viz]'``). The
color data (:func:`cycle`) and the :func:`rc_params` dict need no matplotlib import; :func:`use`,
:func:`context`, and :func:`cmap` do, and raise a clear error if it is missing. Every color comes
from the same :mod:`andrey.viz.theme` tokens as the graph renderer, so charts and structure diagrams
speak one visual language.

Examples
--------
>>> from andrey.viz import mpl
>>> len(mpl.cycle()) == 8 and mpl.cycle()[0] == "#3b5bdb"
True
>>> mpl.rc_params(dark=True)["figure.facecolor"]
'#0b1220'
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .theme import DARK, LIGHT, Palette

if TYPE_CHECKING:
    from matplotlib.colors import LinearSegmentedColormap

__all__ = ["cycle", "rc_params", "use", "context", "cmap"]

_FONT = ["Inter", "DejaVu Sans", "system-ui", "sans-serif"]


def _palette(dark: bool) -> Palette:
    return DARK if dark else LIGHT


def _require_matplotlib() -> Any:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:  # pragma: no cover - trivial import guard
        raise ModuleNotFoundError(
            "andrey.viz.mpl needs matplotlib; install the viz extra: pip install 'andrey-core[viz]'"
        ) from exc
    return matplotlib


def cycle(dark: bool = False) -> list[str]:
    """Return the categorical color cycle for line / bar / scatter series.

    Parameters
    ----------
    dark : bool, default=False
        Select the dark palette instead of the light one.

    Returns
    -------
    list of str
        The colorblind-safe categorical eight, as hex strings, in the order series should consume
        them. Needs no matplotlib import, so it also serves manual coloring and non-matplotlib
        plotters.
    """
    return list(_palette(dark).categorical)


def rc_params(dark: bool = False) -> dict[str, Any]:
    """Return the ``andrey.viz`` chart style as a dict of matplotlib rcParams.

    Building the dict needs no matplotlib. Apply it with ``matplotlib.rcParams.update`` or
    ``matplotlib.rc_context``; :func:`use` and :func:`context` do this and also set the color cycle.

    Parameters
    ----------
    dark : bool, default=False
        Select the dark palette instead of the light one.

    Returns
    -------
    dict of str to object
        Colors from the palette for the figure, axes, grid, text, and ticks, plus font, line, and
        legend settings. It holds no ``axes.prop_cycle``; :func:`cycle` gives the series colors.
    """
    p = _palette(dark)
    return {
        "figure.facecolor": p.paper,
        "figure.edgecolor": p.paper,
        "savefig.facecolor": p.paper,
        "savefig.edgecolor": p.paper,
        "axes.facecolor": p.paper,
        "axes.edgecolor": p.line,
        "axes.linewidth": 1.0,
        "axes.labelcolor": p.ink,
        "axes.titlecolor": p.ink,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12.5,
        "axes.labelsize": 10.5,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": p.line,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "text.color": p.ink,
        "xtick.color": p.muted,
        "ytick.color": p.muted,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "font.family": "sans-serif",
        "font.sans-serif": list(_FONT),  # copy: rcParams may mutate the value in place
        "font.size": 10.5,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.markersize": 5.5,
        "patch.edgecolor": p.paper,
        "patch.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "figure.dpi": 120,
    }


def use(dark: bool = False) -> None:
    """Apply the style to the global matplotlib rcParams, for the rest of the process.

    Parameters
    ----------
    dark : bool, default=False
        Select the dark palette instead of the light one.

    Raises
    ------
    ModuleNotFoundError
        If matplotlib is not installed. Install the extra: ``pip install 'andrey-core[viz]'``.

    See Also
    --------
    context : Scope the same style to a ``with`` block instead of the whole process.
    """
    mpl = _require_matplotlib()
    from cycler import cycler

    mpl.rcParams.update(rc_params(dark))
    mpl.rcParams["axes.prop_cycle"] = cycler(color=cycle(dark))


def context(dark: bool = False) -> Any:
    """Return a context manager that applies the style within a ``with`` block.

    Parameters
    ----------
    dark : bool, default=False
        Select the dark palette instead of the light one.

    Returns
    -------
    contextlib.AbstractContextManager
        The result of ``matplotlib.rc_context``: on exit it restores the previous rcParams, so
        styling one figure leaves the rest of the process untouched.

    Raises
    ------
    ModuleNotFoundError
        If matplotlib is not installed. Install the extra: ``pip install 'andrey-core[viz]'``.
    """
    matplotlib = _require_matplotlib()
    from cycler import cycler

    rc = dict(rc_params(dark))
    rc["axes.prop_cycle"] = cycler(color=cycle(dark))
    return matplotlib.rc_context(rc)  # top-level: avoids importing pyplot / selecting a backend


def cmap(dark: bool = False, name: str | None = None) -> LinearSegmentedColormap:
    """Return the sequential cobalt colormap for heatmaps, weights, and densities.

    This is the confidence ramp: reach for it whenever a value is ordered (edge stability, an
    adjacency weight, a density), and for :func:`cycle` when it is categorical.

    Parameters
    ----------
    dark : bool, default=False
        Select the dark palette instead of the light one.
    name : str or None, default=None
        The name carried on the returned colormap object. ``None`` uses ``"andrey"``
        (``"andrey-dark"`` when ``dark``). Nothing is registered globally, so pass the object
        itself as ``cmap=``, not the name.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The ramp, interpolated from the palette's five cobalt stops.

    Raises
    ------
    ModuleNotFoundError
        If matplotlib is not installed. Install the extra: ``pip install 'andrey-core[viz]'``.
    """
    _require_matplotlib()
    from matplotlib.colors import LinearSegmentedColormap

    resolved = name if name is not None else ("andrey-dark" if dark else "andrey")
    return LinearSegmentedColormap.from_list(resolved, list(_palette(dark).ramp))
