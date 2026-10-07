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
from andrey.core.structure import ARROW, GraphStructure, Kind

from ._common import _offdiag, as_marks, prf, two_marks

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


def _shd_endpoint(m_est: np.ndarray, m_true: np.ndarray) -> int:
    """Endpoint-aware SHD: count off-diagonal positions whose mark differs (a reversal costs 2)."""
    return int(((m_est != m_true) & _offdiag(m_est.shape[0])).sum())


def _prf_masks(est: np.ndarray, true_: np.ndarray) -> dict[str, float]:
    """Precision/recall/f1 between two boolean masks over the same index set."""
    tp = int((est & true_).sum())
    return prf(tp, int((est & ~true_).sum()), int((~est & true_).sum()))


def _skeleton(
    m_est: np.ndarray, m_true: np.ndarray, *, include_diagonal: bool = False
) -> dict[str, float]:
    iu = np.triu_indices(m_est.shape[0], 0 if include_diagonal else 1)
    est = ((m_est != 0) | (m_est.T != 0))[iu]
    true_ = ((m_true != 0) | (m_true.T != 0))[iu]
    return _prf_masks(est, true_)


def _orientation(
    m_est: np.ndarray, m_true: np.ndarray, *, include_diagonal: bool = False
) -> dict[str, float]:
    n = m_est.shape[0]
    mask = np.ones((n, n), dtype=bool) if include_diagonal else _offdiag(n)
    return _prf_masks((m_est == ARROW) & mask, (m_true == ARROW) & mask)


def _confounder(m_est: np.ndarray, m_true: np.ndarray) -> dict[str, float]:
    iu = np.triu_indices(m_est.shape[0], 1)
    est = ((m_est == ARROW) & (m_est.T == ARROW))[iu]
    true_ = ((m_true == ARROW) & (m_true.T == ARROW))[iu]
    return _prf_masks(est, true_)


def canonicalize_marks(marks: np.ndarray, kind: Kind) -> np.ndarray:
    """CPDAG marks of a DAG (its essential graph); CPDAG / PAG marks pass through unchanged."""
    return dag2cpdag(marks) if kind == "dag" else marks


def _digraph_pair(k_est: Kind, k_true: Kind) -> bool:
    """Whether either kind is ``digraph``; then the diagonal (self-loops) is scored, not masked."""
    return k_est == "digraph" or k_true == "digraph"


def reject_digraph_pag(k_est: Kind, k_true: Kind) -> None:
    """Raise when one kind is a ``digraph`` and the other a ``pag`` -- semantically incompatible.

    A digraph and a PAG share the ``ARROW``/``ARROW`` mark pattern but mean opposite things (a
    directed 2-cycle versus a latent-confounded bidirected edge), so identical marks would score a
    perfect match between contradictory claims. No comparison family covers the pair; fail fast.
    """
    if {k_est, k_true} == {"digraph", "pag"}:
        raise ValueError(
            "cannot compare a digraph to a PAG: ARROW/ARROW marks mean a 2-cycle in a digraph "
            "but a latent confounder in a PAG"
        )


# ---- public object-level surface ----------------------------------------------------------------


def shd(estimated: object, true: object, *, endpoint_aware: bool = False) -> int:
    """Structural Hamming distance between two graphs (lower is better; ``shd(g, g) == 0``).

    The default structural count charges one edit per unordered pair whose ``(mark_i, mark_j)``
    differs, so an inserted, deleted, or reversed edge each costs 1 (the pcalg / bnlearn
    convention; note cdt doubles reversals by default). ``endpoint_aware=True`` sums the
    per-endpoint mark mismatches instead -- the standard PAG distance, where a reversal or a deleted
    directed edge costs 2 and tail/arrow/circle are distinguished.

    Examples
    --------
    >>> import numpy as np
    >>> shd(np.array([[0, 1], [0, 0]]), np.array([[0, 0], [1, 0]]))  # 0->1 vs 1->0
    1
    """
    m_est, k_est, m_true, k_true = two_marks(estimated, true)
    reject_digraph_pag(k_est, k_true)  # identical ARROW/ARROW marks would read as SHD 0
    if endpoint_aware:
        return _shd_endpoint(m_est, m_true)
    return _shd_structural(m_est, m_true, include_diagonal=_digraph_pair(k_est, k_true))


def skeleton_scores(estimated: object, true: object) -> dict[str, float]:
    """Precision/recall/f1 of the undirected skeleton (edge presence, orientation ignored).

    Examples
    --------
    >>> skeleton_scores([[0, 1], [0, 0]], [[0, 1], [0, 0]])["f1"]
    1.0
    """
    m_est, k_est, m_true, k_true = two_marks(estimated, true)
    return _skeleton(m_est, m_true, include_diagonal=_digraph_pair(k_est, k_true))


def orientation_scores(estimated: object, true: object) -> dict[str, float]:
    """Precision/recall/f1 of arrowhead (``ARROW``) marks -- the orientation agreement.

    For DAGs this is directed-edge F1 (one arrowhead per directed edge, a reversal is one false
    positive and one false negative); for PAGs it is arrowhead precision/recall.

    Examples
    --------
    >>> orientation_scores([[0, 1], [0, 0]], [[0, 1], [0, 0]])["f1"]
    1.0
    """
    m_est, k_est, m_true, k_true = two_marks(estimated, true)
    reject_digraph_pag(k_est, k_true)  # identical ARROW/ARROW marks would read as perfect agreement
    return _orientation(m_est, m_true, include_diagonal=_digraph_pair(k_est, k_true))


def confounder_pair_scores(estimated: object, true: object) -> dict[str, float]:
    """Precision/recall/f1 of bidirected (``<->``) pairs -- the latent-confounder signals of a PAG.

    Assumes both graphs speak the same PAG vocabulary (a latent confounder shows as a bidirected
    edge). Scoring a PAG estimate against a plain true DAG that has no bidirected edges reports the
    class mismatch as low recall, not a method error.

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
    """Whether two graphs lie in the same Markov equivalence class (Verma-Pearl 1990).

    Two DAGs (or their CPDAGs) are Markov equivalent iff they share the same skeleton and the same
    set of v-structures (unshielded colliders). Comparing GES / PC output this way lets a discovered
    CPDAG match any score-equivalent member of its equivalence class -- looser than a byte-for-byte
    adjacency comparison, exactly the invariance a score search is entitled to. Both inputs are read
    as endpoint marks (a :class:`~andrey.core.GraphStructure`, a :class:`StructureOutput`, or a raw
    directed adjacency), so a raw DAG and its essential-graph CPDAG compare equal.

    Examples
    --------
    >>> markov_equivalent([[0, 1, 0], [0, 0, 1], [0, 0, 0]], [[0, 0, 0], [1, 0, 1], [0, 0, 0]])
    True
    """
    m_est, _k_est, m_true, _k_true = two_marks(estimated, true)  # fails loud on node-count mismatch
    same_skeleton = np.array_equal(m_est != 0, m_true != 0)
    return bool(same_skeleton) and _vstructures(m_est) == _vstructures(m_true)


def to_cpdag(graph: object) -> GraphStructure:
    """Canonicalize a DAG to its CPDAG (essential graph); CPDAG/PAG inputs pass through unchanged.

    Comparing MEC-identified graphs (PC / GES / ...) through their CPDAG lets a DAG and
    any score-equivalent member of its Markov equivalence class score identically. A ``digraph``
    (directed, cyclic) has no essential graph and raises.

    Examples
    --------
    >>> import numpy as np
    >>> to_cpdag(np.array([[0, 1], [0, 0]])).kind  # a lone edge is unorientable
    'cpdag'
    """
    marks, kind = as_marks(graph)
    if kind == "digraph":
        raise ValueError("a digraph (directed, cyclic) has no CPDAG")
    canonical = canonicalize_marks(marks, kind)
    return GraphStructure.from_numpy(canonical, kind="cpdag" if kind == "dag" else kind)
