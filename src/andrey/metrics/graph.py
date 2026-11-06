"""Graph-structure metrics: structural Hamming distance, skeleton / orientation / confounder scores.

All operate on the endpoint-mark matrix, so a metric never asks which algorithm produced a graph --
only what the graph carries. Skeleton counts any adjacency; orientation counts arrowhead (``ARROW``)
marks; confounder pairs are bidirected edges. ``shd`` is per-pair-structural (a reversal costs 1);
``shd(..., endpoint_aware=True)`` sums per-endpoint mark mismatches for PAG comparison (a reversal
costs 2, a deleted edge 2).
"""

from __future__ import annotations

import numpy as np

from andrey.core.orient import dag2cpdag
from andrey.core.structure import ARROW, GraphStructure, Kind, is_ancestral_kind

from ._common import _offdiag, as_marks, prf_masks, two_marks

# ---- marks-level core (extract once, then stay in dense int8) ------------------------------------


def _shd_structural(
    m_est: np.ndarray, m_true: np.ndarray, *, include_diagonal: bool = False
) -> int:
    """Per-pair structural SHD: one edit per unordered pair whose ``(mark_i, mark_j)`` differs.

    ``include_diagonal`` also counts self-loop (diagonal) mismatches -- used for the digraph family,
    whose autoregressive self-loops are real structure the off-diagonal count would miss.
    """
    iu = np.triu_indices(m_est.shape[0], 0 if include_diagonal else 1)
    differ = (m_est[iu] != m_true[iu]) | (m_est.T[iu] != m_true.T[iu])
    return int(differ.sum())


def _shd_skeleton(m_est: np.ndarray, m_true: np.ndarray) -> int:
    """Skeleton SHD: one edit per unordered pair adjacent in one graph and not in the other."""
    iu = np.triu_indices(m_est.shape[0], 1)
    est = ((m_est != 0) | (m_est.T != 0))[iu]
    true_ = ((m_true != 0) | (m_true.T != 0))[iu]
    return int((est != true_).sum())


def _shd_endpoint(m_est: np.ndarray, m_true: np.ndarray) -> int:
    """Endpoint-aware SHD: count off-diagonal positions whose mark differs (a reversal costs 2)."""
    return int(((m_est != m_true) & _offdiag(m_est.shape[0])).sum())


def _skeleton(
    m_est: np.ndarray, m_true: np.ndarray, *, include_diagonal: bool = False
) -> dict[str, float]:
    iu = np.triu_indices(m_est.shape[0], 0 if include_diagonal else 1)
    est = ((m_est != 0) | (m_est.T != 0))[iu]
    true_ = ((m_true != 0) | (m_true.T != 0))[iu]
    return prf_masks(est, true_)


def _orientation(
    m_est: np.ndarray, m_true: np.ndarray, *, include_diagonal: bool = False
) -> dict[str, float]:
    n = m_est.shape[0]
    mask = np.ones((n, n), dtype=bool) if include_diagonal else _offdiag(n)
    return prf_masks((m_est == ARROW) & mask, (m_true == ARROW) & mask)


def _confounder(m_est: np.ndarray, m_true: np.ndarray) -> dict[str, float]:
    iu = np.triu_indices(m_est.shape[0], 1)
    est = ((m_est == ARROW) & (m_est.T == ARROW))[iu]
    true_ = ((m_true == ARROW) & (m_true.T == ARROW))[iu]
    return prf_masks(est, true_)


def canonicalize_marks(marks: np.ndarray, kind: Kind) -> np.ndarray:
    """CPDAG marks of a DAG (its essential graph); CPDAG, PAG, and MAG marks pass through."""
    return dag2cpdag(marks) if kind == "dag" else marks


def _digraph_pair(k_est: Kind, k_true: Kind) -> bool:
    """Whether either kind is ``digraph`` and neither ``undirected``; then the diagonal
    (self-loops) is scored, not masked."""
    return "digraph" in (k_est, k_true) and not undirected_pair(k_est, k_true)


def undirected_pair(k_est: Kind, k_true: Kind) -> bool:
    """Whether either kind is ``undirected``; then only the skeletons compare."""
    return "undirected" in (k_est, k_true)


def reject_undirected(k_est: Kind, k_true: Kind, claim: str) -> None:
    """Raise when either kind is ``undirected``: such a graph makes no ``claim`` to compare."""
    if undirected_pair(k_est, k_true):
        raise ValueError(f"an undirected graph claims no {claim}; compare skeletons instead")


def reject_digraph_pag(k_est: Kind, k_true: Kind) -> None:
    """Raise when one kind is a ``digraph`` and the other a ``pag`` or ``mag`` -- incompatible.

    A digraph and a PAG share the ``ARROW``/``ARROW`` mark pattern but mean opposite things (a
    directed 2-cycle versus a latent-confounded bidirected edge), so identical marks would score a
    perfect match between contradictory claims. No comparison family covers the pair; fail fast.
    """
    if "digraph" in (k_est, k_true) and (is_ancestral_kind(k_est) or is_ancestral_kind(k_true)):
        raise ValueError(
            "cannot compare a digraph to a PAG or MAG: ARROW/ARROW marks mean a 2-cycle in a "
            "digraph but a latent confounder in a PAG or MAG"
        )


# ---- public object-level surface ----------------------------------------------------------------


def shd(estimated: object, true: object, *, endpoint_aware: bool = False) -> int:
    """Return the structural Hamming distance between two graphs.

    The default structural count charges one edit per unordered pair whose ``(mark_i, mark_j)``
    differs, so an inserted, deleted, or reversed edge each costs 1. ``endpoint_aware=True`` sums
    the per-endpoint mark mismatches instead, the standard PAG distance, where a reversal or a
    deleted directed edge costs 2 and tail, arrowhead, and circle are distinguished. When either
    graph is ``undirected``, whose edges claim no direction, the count is the skeleton SHD: one
    edit per pair adjacent in one graph and not in the other, whatever the marks.

    Parameters
    ----------
    estimated : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The estimated graph. A raw array is a directed adjacency: a nonzero ``A[i, j]`` is the
        edge ``i -> j``.
    true : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The true graph, over the same variables in the same order.
    endpoint_aware : bool, default=False
        Count the mismatched endpoint marks instead of the pairs that differ.

    Returns
    -------
    int
        The number of edits, lower being better; ``shd(g, g) == 0``.

    Raises
    ------
    ValueError
        If the graphs differ in size or labels, one is a ``digraph`` and the other a ``pag`` or
        ``mag``, or ``endpoint_aware=True`` with an ``undirected`` graph.

    Examples
    --------
    >>> import numpy as np
    >>> shd(np.array([[0, 1], [0, 0]]), np.array([[0, 0], [1, 0]]))  # 0->1 vs 1->0
    1
    """
    m_est, k_est, m_true, k_true = two_marks(estimated, true)
    reject_digraph_pag(k_est, k_true)  # identical ARROW/ARROW marks would read as SHD 0
    if undirected_pair(k_est, k_true):
        if endpoint_aware:
            reject_undirected(k_est, k_true, "endpoint marks")
        return _shd_skeleton(m_est, m_true)
    if endpoint_aware:
        return _shd_endpoint(m_est, m_true)
    return _shd_structural(m_est, m_true, include_diagonal=_digraph_pair(k_est, k_true))


def skeleton_scores(estimated: object, true: object) -> dict[str, float]:
    """Return the precision, recall, and F1 of the estimated skeleton against the true one.

    The skeleton is the set of adjacent pairs, whatever their marks. When either graph is a
    ``digraph``, its self-loops count as well.

    Parameters
    ----------
    estimated : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The estimated graph. A raw array is a directed adjacency: a nonzero ``A[i, j]`` is the
        edge ``i -> j``.
    true : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The true graph, over the same variables in the same order.

    Returns
    -------
    dict
        ``"precision"``, ``"recall"``, and ``"f1"``, each a float in ``[0, 1]``. A ratio with no
        denominator scores 1.0, so two graphs with nothing to score agree perfectly.

    Examples
    --------
    >>> skeleton_scores([[0, 1], [0, 0]], [[0, 1], [0, 0]])["f1"]
    1.0
    """
    m_est, k_est, m_true, k_true = two_marks(estimated, true)
    return _skeleton(m_est, m_true, include_diagonal=_digraph_pair(k_est, k_true))


def orientation_scores(estimated: object, true: object) -> dict[str, float]:
    """Return the precision, recall, and F1 of the estimated arrowheads against the true ones.

    For DAGs this is the directed-edge F1: one arrowhead per directed edge, so a reversal is one
    false positive and one false negative. For PAGs it scores the arrowhead marks.

    Parameters
    ----------
    estimated : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The estimated graph. A raw array is a directed adjacency: a nonzero ``A[i, j]`` is the
        edge ``i -> j``.
    true : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The true graph, over the same variables in the same order.

    Returns
    -------
    dict
        ``"precision"``, ``"recall"``, and ``"f1"``, each a float in ``[0, 1]``. A ratio with no
        denominator scores 1.0, so two graphs with nothing to score agree perfectly.

    Examples
    --------
    >>> orientation_scores([[0, 1], [0, 0]], [[0, 1], [0, 0]])["f1"]
    1.0
    """
    m_est, k_est, m_true, k_true = two_marks(estimated, true)
    reject_digraph_pag(k_est, k_true)  # identical ARROW/ARROW marks would read as perfect agreement
    reject_undirected(k_est, k_true, "orientation")
    return _orientation(m_est, m_true, include_diagonal=_digraph_pair(k_est, k_true))


def confounder_pair_scores(estimated: object, true: object) -> dict[str, float]:
    """Return the precision, recall, and F1 of the estimated bidirected (``<->``) pairs.

    A bidirected edge is a PAG's sign of a latent confounder, so both graphs should be PAGs or
    MAGs. Scoring a PAG estimate against a true DAG, which has no bidirected edge, reports the class
    mismatch as low recall, not a method error.

    Parameters
    ----------
    estimated : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The estimated graph. A raw array is a directed adjacency: a nonzero ``A[i, j]`` is the
        edge ``i -> j``.
    true : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The true graph, over the same variables in the same order.

    Returns
    -------
    dict
        ``"precision"``, ``"recall"``, and ``"f1"``, each a float in ``[0, 1]``. A ratio with no
        denominator scores 1.0, so two graphs with nothing to score agree perfectly.

    Examples
    --------
    >>> from andrey.core import GraphStructure
    >>> import numpy as np
    >>> g = GraphStructure.from_numpy(np.array([[0, 2], [2, 0]], dtype=np.int8), kind="pag")
    >>> confounder_pair_scores(g, g)["precision"]
    1.0
    """
    m_est, k_est, m_true, k_true = two_marks(estimated, true)
    reject_digraph_pag(k_est, k_true)  # a digraph 2-cycle is not a latent confounder
    reject_undirected(k_est, k_true, "latent confounder")
    return _confounder(m_est, m_true)


def _vstructures(m: np.ndarray) -> set[tuple[int, frozenset[int]]]:
    """Unshielded colliders of an endpoint-mark matrix: ``(z, {x, y})`` with ``x -> z <- y``.

    A collider at ``z`` needs an arrowhead at ``z`` from each of two non-adjacent sources ``x``,
    ``y`` (``m[z, x] == m[z, y] == ARROW``); the pair being non-adjacent (``m[x, y] == m[y, x] ==
    0``) is what makes it a *v-structure* rather than a shielded collider. The unordered ``{x, y}``
    canonicalizes the pair so orientation-order does not matter.
    """
    n = m.shape[0]
    out: set[tuple[int, frozenset[int]]] = set()
    for z in range(n):
        heads = [k for k in range(n) if m[z, k] == ARROW]  # k -> z (arrowhead at z)
        for a in range(len(heads)):
            for b in range(a + 1, len(heads)):
                x, y = heads[a], heads[b]
                if m[x, y] == 0 and m[y, x] == 0:  # x, y non-adjacent -> unshielded collider
                    out.add((z, frozenset((x, y))))
    return out


def markov_equivalent(estimated: object, true: object) -> bool:
    """Return whether two graphs lie in the same Markov equivalence class (Verma and Pearl 1990).

    Two DAGs, or their CPDAGs, are Markov equivalent when they share the same skeleton and the same
    set of v-structures (unshielded colliders). Comparing a PC or GES result this way lets a
    discovered CPDAG match any member of its equivalence class, which is looser than comparing the
    adjacencies and is exactly the invariance a score search is entitled to. A raw DAG and its
    CPDAG compare equal.

    Parameters
    ----------
    estimated : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The estimated graph. A raw array is a directed adjacency: a nonzero ``A[i, j]`` is the
        edge ``i -> j``.
    true : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The true graph, over the same variables in the same order.

    Returns
    -------
    bool
        ``True`` when the graphs share their skeleton and their v-structures.

    Raises
    ------
    ValueError
        If the graphs differ in size or labels, or either is a ``digraph`` (directed, cyclic) or
        ``undirected``, which has no equivalence class.

    Examples
    --------
    >>> markov_equivalent([[0, 1, 0], [0, 0, 1], [0, 0, 0]], [[0, 0, 0], [1, 0, 1], [0, 0, 0]])
    True
    """
    m_est, k_est, m_true, k_true = two_marks(estimated, true)  # fails loud on node-count mismatch
    if "digraph" in (k_est, k_true):
        raise ValueError("a digraph (directed, cyclic) has no Markov equivalence class")
    reject_undirected(k_est, k_true, "equivalence class")
    same_skeleton = np.array_equal(m_est != 0, m_true != 0)
    return bool(same_skeleton) and _vstructures(m_est) == _vstructures(m_true)


def to_cpdag(graph: object) -> GraphStructure:
    """Return the CPDAG (essential graph) of a DAG; a CPDAG, PAG, or MAG passes through.

    Comparing graphs that a method identifies up to their Markov equivalence class, such as PC's or
    GES's, through their CPDAGs lets a DAG and any member of its class score identically. A
    ``digraph`` (directed, cyclic) or an ``undirected`` graph has no essential graph and raises.

    Parameters
    ----------
    graph : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The graph. A raw array is a directed adjacency: a nonzero ``A[i, j]`` is the edge
        ``i -> j``.

    Returns
    -------
    GraphStructure
        The CPDAG of a DAG, of kind ``"cpdag"``; a CPDAG, PAG, or MAG with its marks and kind.

    Examples
    --------
    >>> import numpy as np
    >>> to_cpdag(np.array([[0, 1], [0, 0]])).kind  # a lone edge is unorientable
    'cpdag'
    """
    marks, kind = as_marks(graph)
    if kind == "digraph":
        raise ValueError("a digraph (directed, cyclic) has no CPDAG")
    if kind == "undirected":
        raise ValueError("an undirected graph has no CPDAG; compare skeletons instead")
    canonical = canonicalize_marks(marks, kind)
    return GraphStructure.from_numpy(canonical, kind="cpdag" if kind == "dag" else kind)
