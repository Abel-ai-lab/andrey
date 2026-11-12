"""The answers of Markov blanket and parents-and-children learners, and the scorers of their lane.

A set learner answers for one target: its Markov blanket, or its parents and children. Asked of
every variable it answers for the whole graph, in the graph the sets give: the moral graph, the
skeleton, or the CPDAG. This module holds what every solution and the lane share: how an adapter
hands its answer to the driver (:func:`pack_answer`), the true sets (:func:`true_set`,
:func:`true_graph`), and the scorers (:func:`set_scores`, :func:`pooled_set_scores`,
:func:`pair_scores`).

An answer is a ``(d, d)`` array. For sets, ``answer[t, j] != 0`` puts ``j`` in the set of target
``t``, and the rows of the targets that were not asked for are zero. For the moral graph or the
skeleton it is the graph's symmetric 0/1 adjacency, and for the CPDAG a 0/1 adjacency or endpoint
marks, as :func:`andrey_bench.contracts.structure_from_adjacency` reads them.

The module imports NumPy only, except :func:`answer_structure` and :func:`true_set`, which import
Andrey when called.
"""

from __future__ import annotations

import numpy as np

#: The ``params`` key that lists the targets, as column positions. Without it a solution answers
#: for the whole graph, in the graph :data:`STRUCTURE_PARAM` names.
TARGETS_PARAM = "targets"

#: The ``params`` key of the graph a whole-graph answer takes.
STRUCTURE_PARAM = "structure"

#: The sets: a Markov blanket (``"mb"``), and parents and children (``"pc"``).
KINDS = ("mb", "pc")

#: The graphs the sets give: every variable joined to its Markov blanket (``"moral"``) or to its
#: parents and children (``"skeleton"``), and the CPDAG.
STRUCTURES = ("moral", "skeleton", "cpdag")

#: The test count of a target, or of a fit, that its solution does not report.
NO_COUNT = -1

# The arrowhead of Andrey's endpoint marks (mirrors contracts.py).
_ARROW = 2


def pack_answer(
    answer: np.ndarray, counts: np.ndarray | None = None, total: int | None = None
) -> np.ndarray:
    """Pack a ``(d, d)`` answer and its test counts into the ``(d + 2, d)`` array ``fit`` returns.

    Row ``d`` holds the tests of each target's own search, :data:`NO_COUNT` where the solution
    reports none; entry 0 of row ``d + 1`` holds the tests of the whole fit, :data:`NO_COUNT` when
    none. The array is ``int64``, so a count of millions of tests survives the packing.
    """
    answer = np.asarray(answer)
    d = answer.shape[0]
    if answer.shape != (d, d):
        raise ValueError(f"an answer is a square matrix, got shape {answer.shape}")
    packed = np.full((d + 2, d), NO_COUNT, dtype=np.int64)
    packed[:d] = answer
    if counts is not None:
        packed[d] = np.asarray(counts, dtype=np.int64)
    packed[d + 1, 0] = NO_COUNT if total is None else int(total)
    return packed


def unpack_answer(native: np.ndarray) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Return ``(answer, counts, total)`` of what ``fit`` returned.

    ``counts`` has a :data:`NO_COUNT` entry per target without a count, and ``total`` is ``None``
    when the fit reports none. A bare square array, such as the empty answer, has no counts.
    """
    native = np.asarray(native)
    rows, d = native.shape
    if rows == d:
        return native, np.full(d, NO_COUNT, dtype=np.int64), None
    if rows != d + 2:
        raise ValueError(f"an answer is a (d, d) or a (d + 2, d) array, got shape {native.shape}")
    total = int(native[d + 1, 0])
    return native[:d], native[d].copy(), None if total == NO_COUNT else total


def answer_structure(native: np.ndarray, structure: str | None):
    """The answer `native`, as ``fit`` returned it, as a ``GraphStructure``.

    A ``"cpdag"`` answer is read as endpoint marks when it holds an arrowhead and as a 0/1
    adjacency, a directed edge ``i -> j`` at ``[i, j]``, otherwise. Any other answer, the moral
    graph, the skeleton, or the sets of several targets (``structure`` ``None``), is the
    undirected graph that joins a pair when either entry marks it.
    """
    from andrey import GraphStructure
    from andrey_bench.contracts import structure_from_adjacency

    answer, _, _ = unpack_answer(native)
    answer = np.asarray(answer)
    if structure == "cpdag":
        if (answer == _ARROW).any():
            return GraphStructure.from_numpy(answer.astype(np.int8), kind="cpdag")
        return structure_from_adjacency(answer, kind="cpdag")
    adjacent = (answer != 0) | (answer != 0).T
    np.fill_diagonal(adjacent, False)
    return structure_from_adjacency(adjacent.astype(np.int8), kind="undirected")


def sets_of(answer: np.ndarray, targets: list[int]) -> dict[int, list[int]]:
    """The set of each target in `targets`: the columns its row marks, ascending."""
    answer = np.asarray(answer)
    return {int(t): [int(j) for j in np.flatnonzero(answer[t]) if j != t] for t in targets}


def true_set(marks: np.ndarray, target: int, kind: str) -> list[int]:
    """The true set of `target` in the DAG `marks` (endpoint marks): ``kind`` ``"pc"`` is its
    parents and children, ``"mb"`` those and its children's other parents, the oracle answers of
    :func:`andrey.metrics.target_oracle`."""
    from andrey import GraphStructure
    from andrey.metrics import target_oracle

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    dag = GraphStructure.from_numpy(np.asarray(marks), kind="dag")
    answer = "markov_blanket" if kind == "mb" else "parents_and_children"
    return sorted(target_oracle(dag, target, answer).neighbors(target).tolist())


def true_graph(marks: np.ndarray, structure: str) -> np.ndarray:
    """The 0/1 adjacency of the true ``"moral"`` graph or ``"skeleton"`` of the DAG `marks`."""
    if structure not in ("moral", "skeleton"):
        raise ValueError(f"structure must be 'moral' or 'skeleton', got {structure!r}")
    kind = "mb" if structure == "moral" else "pc"
    d = np.asarray(marks).shape[0]
    adjacency = np.zeros((d, d), dtype=np.int8)
    for t in range(d):
        adjacency[t, true_set(marks, t, kind)] = 1
    return adjacency


def _ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


def set_scores(found: list[int], truth: list[int]) -> dict[str, float]:
    """Score one target's set against the true set.

    Counts, as ``est`` (the set found), ``true``, and ``tp`` (in both); ``fp`` and ``fn`` are the
    members only the answer or only the truth has, and ``exact`` is 1 when the two sets are equal.
    Ratios, ``nan`` where the denominator is 0, so an empty answer has no precision of its own and
    an isolated target no recall: ``precision``, ``recall``, ``f1``, and ``distance``, the
    Euclidean distance of ``(recall, precision)`` to ``(1, 1)``. Pool the counts over targets with
    :func:`pooled_set_scores`.
    """
    found, truth = set(found), set(truth)
    tp = len(found & truth)
    precision, recall = _ratio(tp, len(found)), _ratio(tp, len(truth))
    distance = float(np.hypot(1 - recall, 1 - precision)) if found and truth else float("nan")
    return {
        "est": len(found),
        "true": len(truth),
        "tp": tp,
        "fp": len(found - truth),
        "fn": len(truth - found),
        "exact": int(found == truth),
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * tp, len(found) + len(truth)),
        "distance": distance,
    }


def pooled_set_scores(scores: list[dict[str, float]]) -> dict[str, float]:
    """Pool the counts of several :func:`set_scores` into one precision, recall, F1, and distance.

    Also the ``mean_`` of each ratio over the sets where it is defined, and ``exact``, the share of
    the sets found exactly. The pooled ratios weigh a target by its set's size, the means weigh
    every target alike.
    """
    tp, est, true = (sum(s[k] for s in scores) for k in ("tp", "est", "true"))
    precision, recall = _ratio(tp, est), _ratio(tp, true)
    pooled = {
        "targets": len(scores),
        "exact": _ratio(sum(s["exact"] for s in scores), len(scores)),
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * tp, est + true),
        "distance": float(np.hypot(1 - recall, 1 - precision)) if est and true else float("nan"),
    }
    for name in ("precision", "recall", "f1", "distance"):
        defined = [s[name] for s in scores if s[name] == s[name]]
        pooled[f"mean_{name}"] = float(np.mean(defined)) if defined else float("nan")
    return pooled


def pair_scores(found: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Score a whole graph's adjacencies against the true ones, over the unordered pairs.

    `found` and `truth` are 0/1 adjacencies, any nonzero entry an edge; a pair is an edge when
    either of its two entries is. Counts: ``tp``, ``fp``, ``fn``, and
    ``shd``, the pairs the two disagree on; ``empty_shd`` is the empty graph's, the true edges.
    Ratios: ``precision``, ``recall``, ``f1``, ``nan`` where the denominator is 0.
    """
    found, truth = np.asarray(found) != 0, np.asarray(truth) != 0
    upper = np.triu(np.ones(found.shape, dtype=bool), k=1)
    est = (found | found.T) & upper
    true = (truth | truth.T) & upper
    tp, fp, fn = int((est & true).sum()), int((est & ~true).sum()), int((~est & true).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "shd": fp + fn,
        "empty_shd": tp + fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }
