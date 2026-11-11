"""Scores of a local answer: the edges at one target, and the claims about pairs of variables.

:func:`target_edge_scores` scores a target-centered answer, a graph that holds only the edges at
its target, against the true graph's edges there. :func:`claim_scores` scores the ``effects``
records of :mod:`andrey.core.effects` that decide whether an effect is nonzero, against the true
DAG's ancestry.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from andrey.core.effects import check_effects
from andrey.core.output import StructureOutput
from andrey.core.structure import ARROW, TAIL

from ._common import _labels, as_marks, prf, two_marks


def _position(node: object, labels: Sequence[str] | None, n: int, what: str) -> int:
    """The 0-based position ``node`` names among ``n`` nodes: a label, or an int position."""
    if isinstance(node, str):
        if labels is None or node not in labels:
            raise ValueError(f"{what} names {node!r}, which the true graph does not label")
        return list(labels).index(node)
    if isinstance(node, bool) or not isinstance(node, int | np.integer) or not 0 <= node < n:
        raise ValueError(f"{what} must be a label or a position below {n}, got {node!r}")
    return int(node)


def target_edge_scores(estimated: object, true: object, target: str | int) -> dict[str, float]:
    """Score the edges at ``target`` of a target-centered answer against the true graph's.

    Only the edges with ``target`` at one end count; an edge between two other variables is
    ignored in both graphs, as a target-centered answer says nothing about it. An edge is
    adjacent when both graphs join the pair, and right when its marks at both ends also agree.
    Marks are compared as they are, so give the true graph in the kind the answer identifies: the
    CPDAG of the true DAG (:func:`~andrey.metrics.to_cpdag`) for a CPDAG answer, the true PAG for
    a PAG answer, and for a set the undirected graph that joins the target to its true set, such
    as the moral graph for a Markov blanket.

    Parameters
    ----------
    estimated : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The answer. A raw array is a directed adjacency: a nonzero ``A[i, j]`` is the edge
        ``i -> j``.
    true : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The true graph, over the same variables in the same order, of the same kind.
    target : str or int
        The target, by label or by 0-based position.

    Returns
    -------
    dict
        ``"adjacency_precision"``, ``"adjacency_recall"``, ``"adjacency_f1"``: the target's
        adjacent variables; ``"edge_precision"``, ``"edge_recall"``, ``"edge_f1"``: the edges with
        both marks right, each a float in ``[0, 1]``, a ratio with no denominator scoring 1.0.
        ``"missing"``, ``"extra"``, ``"wrong_marks"``: the true edges the answer lacks, the
        answer's edges the truth lacks, and the edges in both with a mark that differs, of which
        ``"reversed"`` are directed both ways but opposite; ``"shd"``, their sum, the structural
        Hamming distance at the target.

    Raises
    ------
    ValueError
        If the graphs differ in size, labels, or kind, or ``target`` names no node.

    Examples
    --------
    >>> answer = [[0, 1, 0], [0, 0, 0], [0, 1, 0]]  # 0 -> 1 <- 2
    >>> truth = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]  # 0 -> 1 -> 2
    >>> s = target_edge_scores(answer, truth, 1)
    >>> s["adjacency_f1"], s["reversed"], s["shd"]
    (1.0, 1, 1)
    """
    m_est, k_est, m_true, k_true = two_marks(estimated, true)
    if k_est != k_true:
        raise ValueError(
            f"the answer is a {k_est} and the true graph a {k_true}; give the true graph in the "
            "answer's kind, such as to_cpdag(true) for a CPDAG"
        )
    labels = _labels(estimated) or _labels(true)
    t = _position(target, labels, len(m_est), "target")
    others = np.arange(len(m_est)) != t
    est = (m_est[t] != 0) & others
    tru = (m_true[t] != 0) & others
    both = est & tru
    right = both & (m_est[t] == m_true[t]) & (m_est[:, t] == m_true[:, t])
    reversed_ = both & _directed(m_est, t) & _directed(m_true, t) & (m_est[t] != m_true[t])
    missing, extra = int((tru & ~est).sum()), int((est & ~tru).sum())
    wrong = int((both & ~right).sum())
    adjacency = prf(int(both.sum()), extra, missing)
    edges = prf(int(right.sum()), int(est.sum() - right.sum()), int(tru.sum() - right.sum()))
    return {
        **{f"adjacency_{k}": v for k, v in adjacency.items()},
        **{f"edge_{k}": v for k, v in edges.items()},
        "missing": missing,
        "extra": extra,
        "wrong_marks": wrong,
        "reversed": int(reversed_.sum()),
        "shd": missing + extra + wrong,
    }


def _directed(m: np.ndarray, t: int) -> np.ndarray:
    """Whether the edge of ``t`` with each node is directed: a tail at one end, an arrow at the
    other."""
    at_t, at_x = m[t], m[:, t]
    return ((at_t == TAIL) & (at_x == ARROW)) | ((at_t == ARROW) & (at_x == TAIL))


def _reach(direct: np.ndarray) -> np.ndarray:
    """``R[i, j]`` when a directed path leads from ``i`` to ``j`` in the support ``direct``."""
    reach = direct.copy()
    while True:
        grown = reach | ((reach.astype(np.int64) @ direct.astype(np.int64)) > 0)
        if (grown == reach).all():
            return reach
        reach = grown


def claim_scores(estimated: object, true: object) -> dict[str, float]:
    """Score the decided ``effects`` records of an answer against the true DAG.

    A record whose ``nonzero`` is ``True`` or ``False`` is a claim; one whose ``nonzero`` is
    ``None`` is undecided. A claim is true when it matches the truth: for a ``"total"`` record,
    whether the treatment is an ancestor of the outcome in the true graph; for a ``"direct"``
    record, whether the treatment is a parent of the outcome.

    Parameters
    ----------
    estimated : StructureOutput or list of dict
        A fit whose ``metadata["effects"]`` holds the records, or the records.
    true : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The true DAG, over every variable the records name and possibly more, such as hidden
        ones. A record names a variable by label, read from the true graph's labels, or by
        0-based position. A raw array is a directed adjacency.

    Returns
    -------
    dict
        ``"claims"``, ``"true_claims"``, ``"false_claims"``, and ``"undecided"`` counts, and
        ``"precision"``, the share of claims that are true, 1.0 when there are none.

    Raises
    ------
    ValueError
        If the records break the record's rules (:func:`~andrey.core.effects.check_effects`),
        name a variable the true graph lacks, or the true graph has an edge without a direction.

    Examples
    --------
    >>> from andrey.core.effects import effect_record
    >>> claims = [effect_record(0, 2, estimand="total", nonzero=True)]
    >>> claim_scores(claims, [[0, 1, 0], [0, 0, 1], [0, 0, 0]])["precision"]
    1.0
    """
    records = (
        estimated.metadata.get("effects", [])
        if isinstance(estimated, StructureOutput)
        else estimated
    )
    check_effects(records)
    assert isinstance(records, list)  # check_effects refuses anything else
    m, kind = as_marks(true)
    if kind not in ("dag", "digraph"):
        raise ValueError(f"claims are scored against a DAG or a digraph, got a {kind}")
    direct = (m == TAIL) & (m.T == ARROW)
    reach = _reach(direct)
    labels = _labels(true)
    found = {"claims": 0, "true_claims": 0, "false_claims": 0, "undecided": 0}
    for i, record in enumerate(records):
        if record["nonzero"] is None:
            found["undecided"] += 1
            continue
        a = _position(record["treatment"], labels, len(m), f"effects[{i}]: treatment")
        b = _position(record["outcome"], labels, len(m), f"effects[{i}]: outcome")
        truth = bool((direct if record["estimand"] == "direct" else reach)[a, b])
        found["claims"] += 1
        found["true_claims" if record["nonzero"] == truth else "false_claims"] += 1
    claims = found["claims"]
    return {**found, "precision": found["true_claims"] / claims if claims else 1.0}
