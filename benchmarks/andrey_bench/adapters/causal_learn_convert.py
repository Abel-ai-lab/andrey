"""Convert causal-learn endpoints to CPDAG adjacency or Andrey marks.

GES, PC, BOSS, and GRaSP return the same ``GeneralGraph`` CPDAG encoding. FCI uses
:func:`remap_to_unsigned` to preserve its PAG circle endpoints. DirectLiNGAM and ICA-LiNGAM return a
weight matrix read by :func:`lingam_to_adjacency`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

# causal-learn endpoint codes (``Endpoint.TAIL`` / ``Endpoint.ARROW``), verified empirically.
_CL_TAIL, _CL_ARROW = -1, 1


def generalgraph_to_adjacency(graph_matrix: np.ndarray) -> np.ndarray:
    """``GeneralGraph.graph`` (signed endpoints) -> ``0/1`` adjacency, zero on the diagonal.

    causal-learn writes the mark at node ``i`` on edge ``i-j`` into ``graph[i, j]``, so a directed
    edge ``i -> j`` is ``graph[i, j] == TAIL`` with ``graph[j, i] == ARROW``, and an undirected edge
    is the symmetric ``TAIL-TAIL``. Both set ``adj[i, j] = 1`` wherever node ``i`` carries a tail
    toward ``j`` — directed sets one direction, undirected sets both, which is exactly what
    :func:`andrey_bench.contracts.structure_from_adjacency` reconstructs.

    Parameters
    ----------
    graph_matrix : np.ndarray
        A square ``GeneralGraph.graph`` matrix from a CPDAG-producing method (GES, PC).

    Returns
    -------
    np.ndarray
        A ``(d, d)`` ``int8`` 0/1 adjacency.
    """
    g = np.asarray(graph_matrix)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError(f"graph_matrix must be square 2-D, got {g.shape}")
    adj = np.zeros(g.shape, dtype=np.int8)
    directed = (g == _CL_TAIL) & (g.T == _CL_ARROW)
    undirected = (g == _CL_TAIL) & (g.T == _CL_TAIL)
    adj[directed] = 1
    adj[undirected] = 1
    np.fill_diagonal(adj, 0)
    return adj


def lingam_to_adjacency(b_matrix: npt.ArrayLike) -> np.ndarray:
    """Convert LiNGAM ``adjacency_matrix_`` to 0/1 adjacency with ``adj[i, j] = 1`` for ``i -> j``.

    causal-learn's LiNGAM estimators fit ``x = B x + e``. ``B[i, j] != 0`` represents ``j -> i``, so
    the adjacency is the transposed nonzero pattern.
    """
    b = np.asarray(b_matrix)
    if b.ndim != 2 or b.shape[0] != b.shape[1]:
        raise ValueError(f"expected a square (n, n) weight matrix, got shape {b.shape}")
    return np.ascontiguousarray((b != 0).T.astype(np.int8))


# causal-learn signed Endpoint codes (bundle .../causallearn/graph/Endpoint.py) -> Andrey marks.
_SIGNED_TO_UNSIGNED: dict[int, int] = {-1: 1, 0: 0, 1: 2, 2: 3}


def _remap(M: npt.ArrayLike, table: dict[int, int], domain: str) -> np.ndarray:
    arr = np.asarray(M)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"expected a square (n, n) endpoint matrix, got shape {arr.shape}")
    unknown = sorted({int(c) for c in np.unique(arr).tolist()} - table.keys())
    if unknown:
        raise ValueError(
            f"unmappable {domain} endpoint code(s) {unknown}; andrey represents only "
            f"NULL/TAIL/ARROW/CIRCLE (codes {sorted(table)})"
        )
    out = np.zeros(arr.shape, dtype=np.int8)
    for src, dst in table.items():
        out[arr == src] = dst
    return out


def remap_to_unsigned(graph: npt.ArrayLike) -> np.ndarray:
    """Remap a causal-learn signed endpoint matrix to Andrey's unsigned marks (``int8``).

    Fails fast on any code outside ``{-1, 0, 1, 2}`` (for example, a compound ``Endpoint``), so an
    unrepresentable mark surfaces instead of silently mis-mapping.
    """
    return _remap(graph, _SIGNED_TO_UNSIGNED, "signed causal-learn")
