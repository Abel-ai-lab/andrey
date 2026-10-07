"""Endpoint-mark builders shared by the private facade adapters.

Every adapter translates its algorithm's coefficient / support matrix into the one clean, unsigned
endpoint-mark matrix ``GraphStructure.from_numpy`` accepts: a square ``int8`` array with a zero
diagonal whose cell ``M[a][b]`` is the mark at node ``a`` on edge ``a-b`` (``NULL`` 0 / ``TAIL`` 1 /
``ARROW`` 2 / ``CIRCLE`` 3). The directed convention is LiNGAM's: ``B[i, j] != 0`` means an edge
``j -> i`` (row = effect, column = cause), so an arrowhead lands on ``i`` and a tail on ``j``.

These helpers own that translation once so every adapter reuses it instead of
re-deriving the cell convention. ``from_numpy`` is neither NaN-safe nor self-loop-tolerant, so the
builders here always return a finite, diagonal-zeroed integer matrix.
"""

from __future__ import annotations

import numpy as np

from andrey.core import ARROW, CIRCLE, TAIL


def directed_marks(B: np.ndarray, *, allow_self_loops: bool = False) -> np.ndarray:
    """Endpoint-mark matrix for the directed support of ``B`` (``B[i, j] != 0`` is edge ``j -> i``).

    Accepts a boolean or float support/coefficient matrix. Non-finite entries (``NaN`` / ``inf``)
    are treated as *no* directed edge -- confounded / order-unknown pairs are added afterwards with
    :func:`set_bidirected` / :func:`set_circle` (or in one step via :func:`marks_from_signed`). A
    single direction becomes ``TAIL``/``ARROW``; when both ``B[i, j]`` and ``B[j, i]`` are nonzero
    the pair is a double arrowhead ``ARROW``/``ARROW`` (a bidirected edge under ``kind="pag"``, a
    2-cycle under ``kind="digraph"`` -- the caller's kind decides which).

    ``allow_self_loops`` keeps the diagonal: off by default (static graphs are self-loop free); a
    lag>=1 graph sets it so an autoregressive edge ``X(t-k) -> X(t)`` survives as ``M[i][i]=ARROW``.
    ``GraphStructure.from_numpy`` must then also get ``allow_self_loops=True``.

    Returns an ``(n, n)`` ``int8`` matrix ready for ``GraphStructure.from_numpy``.
    """
    arr = np.asarray(B, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"expected a square (n, n) matrix, got shape {arr.shape}")
    sup = np.isfinite(arr) & (arr != 0.0)
    if not allow_self_loops:
        np.fill_diagonal(sup, False)
    M = np.zeros(arr.shape, dtype=np.int8)
    # Tail at ``a`` for every edge ``a -> b`` (that is, ``B[b, a] != 0`` == ``sup.T[a, b]``); then
    # an arrowhead at ``a`` for every edge ``b -> a`` (``sup[a, b]``) overwrites it, so a pair
    # carrying both directions collapses to a double arrowhead ``ARROW``/``ARROW`` (a 2-cycle or
    # bidirected).
    M[sup.T] = TAIL
    M[sup] = ARROW
    return M


def set_bidirected(M: np.ndarray, i: int, j: int) -> np.ndarray:
    """Mark the pair ``i <-> j`` bidirected (``ARROW``/``ARROW``); mutates and returns ``M``.

    For latent-confounder edges (for example, RCD's ``NaN`` sentinel). Requires ``kind="pag"``
    downstream.
    """
    if i == j:
        raise ValueError("cannot set a self-loop mark")
    M[i, j] = ARROW
    M[j, i] = ARROW
    return M


def set_circle(M: np.ndarray, i: int, j: int) -> np.ndarray:
    """Mark the pair ``i o-o j`` with circles (``CIRCLE``/``CIRCLE``); mutates and returns ``M``.

    For order-unknown edges (for example, BottomUpParceLiNGAM's ``NaN`` sentinel). Requires
    ``kind="pag"``.
    """
    if i == j:
        raise ValueError("cannot set a self-loop mark")
    M[i, j] = CIRCLE
    M[j, i] = CIRCLE
    return M


def marks_from_signed(B: np.ndarray, *, nan_mark: str) -> np.ndarray:
    """Endpoint marks for a signed matrix whose ``NaN`` entries carry a per-algorithm meaning.

    Builds the directed support with :func:`directed_marks`, then marks every unordered ``NaN``
    pair once with ``nan_mark`` -- ``"bidirected"`` (RCD: latent confounder) or ``"circle"``
    (BottomUpParceLiNGAM: order-unknown). Both require ``kind="pag"`` downstream. Centralizes the
    NaN sweep both latent-confounder adapters share, so neither hand-rolls the sentinel translation;
    the two marks stay distinct because they mean different things (a world-claim vs. epistemic).

    Returns an ``(n, n)`` ``int8`` matrix ready for ``GraphStructure.from_numpy``.
    """
    mark = {"bidirected": ARROW, "circle": CIRCLE}.get(nan_mark)
    if mark is None:
        raise ValueError(f"nan_mark must be 'bidirected' or 'circle', got {nan_mark!r}")
    arr = np.asarray(B, dtype=np.float64)
    marks = directed_marks(arr)
    pair = np.isnan(arr)
    pair |= pair.T  # symmetric: mark both endpoints of every NaN pair
    np.fill_diagonal(pair, False)  # a diagonal NaN is not a pairwise relation
    marks[pair] = mark
    return marks
