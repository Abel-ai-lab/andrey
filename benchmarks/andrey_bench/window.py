"""Window marks: every link of a time-series graph over lags ``0..L``, with the marks at its ends.

PCMCI+ and LPCMCI return links at lag 0 and at lags ``1..L``, and a link carries a mark at each end:
a tail, an arrowhead, a circle, or a conflict. A :class:`~andrey.core.TemporalStructure` holds
lagged links as a ``digraph`` per lag, which keeps only a link's direction, so these solutions
return a window-mark stack instead, an int8 array of shape ``(L + 1, n, n)``:

- ``marks[0]`` holds the same-step links as endpoint marks: ``marks[0, i, j]`` is the mark at
  ``X_i(t)`` on its link with ``X_j(t)``, so ``i -> j`` is ``marks[0, i, j] = TAIL`` and
  ``marks[0, j, i] = ARROW``.
- ``marks[k, i, j]``, ``k >= 1``, is the mark at the earlier end ``X_i(t-k)`` of the link
  ``X_i(t-k) *-> X_j(t)``, and ``0`` when there is no link. The later end of a lagged link is always
  an arrowhead, because a later variable cannot cause an earlier one, so one mark describes the link.

The codes are Andrey's endpoint marks plus one: ``NULL`` 0, ``TAIL`` 1, ``ARROW`` 2, ``CIRCLE`` 3,
and ``CONFLICT`` 4, a mark that orientation rules set both ways.

Both packages' outputs reach this layout through :func:`marks_from_link_strings`, which reads the
three-character link strings that tigramite returns in ``results["graph"]`` and that Andrey's
LPCMCI returns in ``metadata["links"]``. This module imports nothing but numpy, because the worker
in the tigramite environment imports it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NULL, TAIL, ARROW, CIRCLE, CONFLICT = 0, 1, 2, 3, 4

#: The mark at the first end of a link string (``"<->"[0]``) and at its last end (``"<->"[2]``).
_FIRST = {"-": TAIL, "<": ARROW, "o": CIRCLE, "x": CONFLICT}
_LAST = {"-": TAIL, ">": ARROW, "o": CIRCLE, "x": CONFLICT}

#: The kinds a window's lag 0 can have: an estimate's ``cpdag`` or ``pag``, or the true ``dag``.
LAG0_KINDS = ("cpdag", "pag", "dag")


def _mirror(link: str) -> str:
    """The same link read from its other end: ``"o->"`` becomes ``"<-o"``."""
    return link[::-1].translate(str.maketrans("<>", "><"))


def marks_from_link_strings(graph: np.ndarray) -> np.ndarray:
    """Convert an ``(n, n, L + 1)`` array of link strings to the ``(L + 1, n, n)`` window marks.

    ``graph[i, j, k]`` is the link between ``X_i(t-k)`` and ``X_j(t)``: ``""`` for none, else three
    characters, the mark at ``X_i(t-k)`` (``-``, ``<``, ``o``, ``x``), ``-``, and the mark at
    ``X_j(t)`` (``-``, ``>``, ``o``, ``x``). Lag 0 lists every link from both ends, so
    ``graph[j, i, 0]`` must be ``graph[i, j, 0]`` read backwards. Raises ``RuntimeError`` on any
    other string, on a lag-0 link without its mirror, and on a lagged link without an arrowhead at
    its later end, so an unexpected encoding fails the fit instead of being read as no link.
    """
    g = np.asarray(graph).astype(str)
    if g.ndim != 3 or g.shape[0] != g.shape[1] or g.shape[2] < 1:
        raise RuntimeError(f"a link-string graph has shape (n, n, L + 1), got {g.shape}")
    marks = np.zeros((g.shape[2], g.shape[0], g.shape[1]), dtype=np.int8)
    for i, j, k in zip(*np.nonzero(g != ""), strict=True):
        link = str(g[i, j, k])
        if len(link) != 3 or link[1] != "-" or link[0] not in _FIRST or link[2] not in _LAST:
            raise RuntimeError(f"unexpected link {link!r} at ({i}, {j}, lag {k})")
        if k == 0:
            if i == j or str(g[j, i, 0]) != _mirror(link):
                raise RuntimeError(
                    f"lag-0 link {link!r} at ({i}, {j}) has {str(g[j, i, 0])!r} as its mirror"
                )
            marks[0, i, j] = _FIRST[link[0]]
        elif link[2] != ">":
            raise RuntimeError(f"lagged link {link!r} at ({i}, {j}, lag {k}) points back in time")
        else:
            marks[k, i, j] = _FIRST[link[0]]
    return marks


@dataclass(frozen=True, eq=False)
class WindowGraph:
    """A window-mark stack and the kind of its lag 0.

    The estimate of a PCMCI+ (``lag0="cpdag"``) or LPCMCI (``lag0="pag"``) solution, and the truth
    of a regime with same-step links (``lag0="dag"``). ``marks`` keeps every code the solution
    returned, conflicts included, so two estimates are the same graph exactly when their marks are
    equal; scoring reads conflicts as its kind does (:func:`andrey_bench.scoring.window_scores`).
    """

    marks: np.ndarray
    lag0: str

    def __post_init__(self) -> None:
        marks = np.asarray(self.marks)
        if marks.ndim != 3 or marks.shape[1] != marks.shape[2]:
            raise ValueError(f"window marks have shape (L + 1, n, n), got {marks.shape}")
        if self.lag0 not in LAG0_KINDS:
            raise ValueError(f"lag0 must be one of {LAG0_KINDS}, got {self.lag0!r}")
        object.__setattr__(self, "marks", marks.astype(np.int8, copy=False))

    @property
    def n_nodes(self) -> int:
        """The number of variables."""
        return int(self.marks.shape[1])

    @property
    def tau_max(self) -> int:
        """The largest lag the window covers."""
        return int(self.marks.shape[0]) - 1


def window_marks_of_dag_and_links(lag0: np.ndarray, links: np.ndarray) -> np.ndarray:
    """Stack a lag-0 endpoint-mark matrix and an ``(L, n, n)`` 0/1 link stack as window marks.

    ``links[k - 1, i, j]`` is the link ``X_i(t-k) -> X_j(t)``; each becomes a tail at its earlier
    end.
    """
    lagged = np.where(np.asarray(links) != 0, TAIL, NULL)
    return np.concatenate([np.asarray(lag0)[None], lagged]).astype(np.int8)
