"""The answers of local methods, and the scorers of the target and latent lanes.

A local method answers for one target, under hidden variables or selection bias, or claims pairs
instead of drawing a graph. This module holds what every such solution and every such lane share:
how an adapter hands its answer to the driver (:func:`pack_answer`), what the answer is scored
against (:func:`target_edge_scores`, :func:`claim_scores`), and the contract an adapter of a
target-centered method fills (:class:`LocalAdapter`).

An answer at a target is a ``(d, d)`` array of endpoint marks in Andrey's convention
(:mod:`andrey_bench.contracts`): ``marks[i, j]`` is the mark at node ``i`` of the edge between ``i``
and ``j``, and ``0`` no edge, ``1`` tail, ``2`` arrowhead, ``3`` circle. Only the target's row and
column are read, so a global method's whole graph is an answer at every target. A pair claim is a
``(d, d)`` array of scores, ``scores[i, j] > 0`` for the claim that ``i`` is an ancestor of ``j``.

The module imports NumPy only, so an adapter in a reference's own environment can use it.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

#: The key of ``params`` that holds the target's column. The driver adds it to the parameters of
#: every solution with ``local = True`` and no other.
TARGET_PARAM = "target"

#: ``n_tests`` of an answer whose solution reports no count.
NO_COUNT = -1

# Andrey endpoint marks (mirrors contracts.py).
_NULL, _TAIL, _ARROW, _CIRCLE = 0, 1, 2, 3


@runtime_checkable
class LocalAdapter(Protocol):
    """What a solution that answers at a target provides, beside a ``SolutionAdapter``'s members.

    ``fit(data, params)`` receives the dataset's columns and ``params`` with the target's column
    under :data:`TARGET_PARAM`, and returns :func:`pack_answer` of the marks at the target and the
    number of conditional-independence tests it evaluated, or ``None`` when it reports none. The
    driver fits a solution once per target. A solution that learns a whole graph, such as FCI, sets
    ``local = False`` (or omits it): the driver fits it once per dataset and reads every target's
    row and column from that graph.

    ``output_type`` is ``"cpdag"`` for a method that answers in the CPDAG's marks (tails and
    arrowheads) and ``"pag"`` for one that answers in the PAG's (circles too); the driver scores
    each against the truth of its kind. ``to_structure(native)`` returns a ``GraphStructure`` of
    that kind, from :func:`unpack_answer`'s marks.
    """

    local: bool
    output_type: str

    def params(self) -> Mapping[str, Any]:
        """The solution's settings, without the target."""
        ...

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Answer at ``params[TARGET_PARAM]``; the timed region."""
        ...


@runtime_checkable
class ClaimAdapter(Protocol):
    """What a solution that claims ancestral pairs provides, beside a ``SolutionAdapter``'s members.

    ``fit(data, params)`` receives the columns of a dataset with the context variable in the last
    one, and returns :func:`pack_answer` of the score matrix: ``scores[i, j] > 0`` claims that
    ``i`` is an ancestor of ``j``, and the larger the score the more confident the claim. A solution
    that reports no tests packs ``None``. The claims are among the system columns, every column but
    the last; what the solution says about the context is not scored. ``output_type`` is
    ``"claims"``, and ``to_structure(native)`` returns the graph without edges over the columns,
    as a claim of ancestry is not an edge.
    """

    output_type: str

    def params(self) -> Mapping[str, Any]:
        """The solution's settings."""
        ...

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Claim ancestral pairs among the columns; the timed region."""
        ...


def pack_answer(matrix: np.ndarray, n_tests: int | None = None) -> np.ndarray:
    """Pack a ``(d, d)`` answer and its test count into the ``(d + 1, d)`` array ``fit`` returns.

    The last row holds the count in its first entry, :data:`NO_COUNT` when the solution reports
    none. A mark matrix is widened to ``int64`` and a score matrix stays ``float64``, so a count of
    millions of tests survives the packing.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"an answer is a square matrix, got shape {matrix.shape}")
    dtype = np.float64 if matrix.dtype.kind == "f" else np.int64
    packed = np.zeros((matrix.shape[0] + 1, matrix.shape[1]), dtype=dtype)
    packed[:-1] = matrix
    packed[-1, 0] = NO_COUNT if n_tests is None else int(n_tests)
    return packed


def unpack_answer(native: np.ndarray) -> tuple[np.ndarray, int | None]:
    """Return ``(matrix, n_tests)`` of what ``fit`` returned.

    A ``(d + 1, d)`` array is :func:`pack_answer`'s. A square array is a bare answer, such as a
    solution's whole graph or the empty graph, and has no count.
    """
    native = np.asarray(native)
    rows, cols = native.shape
    if rows == cols:
        return native, None
    if rows != cols + 1:
        raise ValueError(f"an answer is a (d, d) or a (d + 1, d) array, got shape {native.shape}")
    count = int(native[-1, 0])
    return native[:-1], None if count == NO_COUNT else count


def target_edges(marks: np.ndarray, target: int) -> dict[int, tuple[int, int]]:
    """The edges at ``target``: ``{other: (mark at target, mark at other)}``, adjacent nodes only."""
    marks = np.asarray(marks)
    row, col = marks[target], marks[:, target]
    return {
        int(j): (int(row[j]), int(col[j]))
        for j in np.flatnonzero((row != _NULL) | (col != _NULL))
        if j != target
    }


def _ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


def target_edge_scores(marks: np.ndarray, truth: np.ndarray, target: int) -> dict[str, float]:
    """Score the edges an answer draws at ``target`` against the truth's.

    ``marks`` and ``truth`` are endpoint-mark arrays over the same columns; ``truth`` is the
    observed PAG when nodes are hidden and the CPDAG when none are. An edge of the answer is a
    *true edge* when the truth has it too with both marks the same (a circle is a mark: ``o->``
    is not ``->``), the paper's rule for a target's edges.

    Counts, all over the pairs ``(target, j)``:

    - ``est_edges``, ``true_edges``: the edges of the answer and of the truth;
    - ``correct_edges``: the true edges the answer has with both marks right;
    - ``adj_correct``: the edges of the answer the truth has, whatever the marks;
    - ``extra``, ``missing``, ``mismarked``: an edge only the answer has, one only the truth has,
      and one both have with a mark that differs; ``shd`` is their sum, the local endpoint SHD,
      and ``empty_shd`` the empty answer's SHD, the truth's degree at the target.

    Ratios, ``nan`` when the denominator is 0 (the empty answer has no precision; pool the counts
    over datasets): ``precision`` and ``recall`` over true edges, ``f1``, ``distance`` the
    Euclidean distance of ``(recall, precision)`` to ``(1, 1)``, and ``adj_precision`` and
    ``adj_recall`` over adjacencies.
    """
    est, true = target_edges(marks, target), target_edges(truth, target)
    correct = sum(1 for j, m in est.items() if true.get(j) == m)
    both = est.keys() & true.keys()
    extra, missing = len(est.keys() - true.keys()), len(true.keys() - est.keys())
    mismarked = len(both) - correct
    precision, recall = _ratio(correct, len(est)), _ratio(correct, len(true))
    f1 = _ratio(2 * correct, len(est) + len(true))
    distance = (
        float(np.hypot(1 - recall, 1 - precision)) if len(est) and len(true) else float("nan")
    )
    return {
        "est_edges": len(est),
        "true_edges": len(true),
        "correct_edges": correct,
        "adj_correct": len(both),
        "extra": extra,
        "missing": missing,
        "mismarked": mismarked,
        "shd": extra + missing + mismarked,
        "empty_shd": len(true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "distance": distance,
        "adj_precision": _ratio(len(both), len(est)),
        "adj_recall": _ratio(len(both), len(true)),
    }


def claim_scores(
    scores: np.ndarray, ancestral: np.ndarray, system: np.ndarray | list[int]
) -> dict[str, float]:
    """Score ancestral claims among the ``system`` columns.

    ``scores[i, j] > 0`` is the claim that ``i`` is an ancestor of ``j``, with the score as its
    confidence; ``ancestral[i, j]`` says it is true. Pairs with a column outside ``system``, such
    as a context node, and the diagonal are left out.

    ``claims``, ``tp``, and ``fp`` count the claims and how they fare; ``positives`` counts the
    true ancestral pairs, ``pairs`` all ordered pairs, and ``base_rate`` is their ratio: the
    precision of claiming every pair, the floor a claim-only method has to beat. ``precision`` and
    ``recall`` are ``nan`` where undefined. ``average_precision`` is the area under the
    precision-recall curve of the ranking by score, ties as one threshold, the pairs that are not
    claimed last, as the standard average-precision estimate takes it.
    """
    scores = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(ancestral, dtype=bool)
    system = np.asarray(system, dtype=int)
    block = np.ix_(system, system)
    s, t = scores[block], truth[block]
    off = ~np.eye(len(system), dtype=bool)
    s, t = s[off], t[off]
    claimed = s > 0
    tp, fp = int((claimed & t).sum()), int((claimed & ~t).sum())
    positives = int(t.sum())
    return {
        "claims": tp + fp,
        "tp": tp,
        "fp": fp,
        "positives": positives,
        "pairs": int(t.size),
        "base_rate": _ratio(positives, t.size),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, positives),
        "average_precision": _average_precision(s, t),
    }


def _average_precision(score: np.ndarray, truth: np.ndarray) -> float:
    """Average precision of ``score`` for ``truth``: ``sum_k (R_k - R_{k-1}) P_k`` over the
    distinct scores, descending."""
    positives = int(truth.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    score, truth = score[order], truth[order]
    # The last index of each run of equal scores: the cut after it keeps the whole run.
    cuts = np.flatnonzero(np.r_[score[1:] != score[:-1], True])
    tp = np.cumsum(truth)[cuts]
    kept = cuts + 1
    recall = tp / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * (tp / kept)))


def marks_to_entries(marks: np.ndarray) -> list[list[int]]:
    """The nonzero marks as ``[i, j, mark]`` triples, for a record that keeps the graph small."""
    marks = np.asarray(marks)
    return [[int(i), int(j), int(marks[i, j])] for i, j in zip(*np.nonzero(marks), strict=True)]


def entries_to_marks(entries: list[list[int]], d: int) -> np.ndarray:
    """The ``(d, d)`` marks of :func:`marks_to_entries`'s triples."""
    marks = np.zeros((d, d), dtype=np.int8)
    for i, j, mark in entries:
        marks[i, j] = mark
    return marks
