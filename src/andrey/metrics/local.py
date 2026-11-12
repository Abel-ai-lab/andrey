"""Scores of a local answer: the edges at one target, and the claims about pairs of variables.

:func:`target_oracle` gives the answer a target-centered method should return, its definition
applied to the true graph. :func:`target_answer_scores` scores an answer against it query by query
(:func:`~andrey.metrics.answer_scores`), and :func:`target_edge_scores` counts the edges at the
target as the papers do. :func:`claim_scores` scores the ``effects`` records of
:mod:`andrey.core.effects` that decide whether an effect is nonzero, against the true DAG's
ancestry.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from typing import Any

import numpy as np

from andrey.core import _queries
from andrey.core.effects import check_effects
from andrey.core.output import StructureOutput
from andrey.core.structure import ARROW, CIRCLE, LATENT, TAIL, GraphStructure

from ._common import _labels, as_marks, prf, two_marks, unwrap
from .answers import answer_scores
from .graph import to_cpdag

#: The answers :func:`target_oracle` gives: a set (``"markov_blanket"``,
#: ``"parents_and_children"``) or the edges at the target in a graph of that kind.
TARGET_ANSWERS: tuple[str, ...] = ("markov_blanket", "parents_and_children", "dag", "cpdag", "pag")

#: The constant answers of the base-rate policy for each kind of query at a target.
TARGET_CHOICES: dict[str, tuple[Any, ...]] = {
    "adjacency": (True, False),
    "marks": ("tail", "arrowhead", None),
}


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


def claim_scores(
    estimated: object, true: object, *, pairs: Iterable[tuple[str | int, str | int]] | None = None
) -> dict[str, Any]:
    """Score the decided ``effects`` records of an answer against the true DAG.

    A record whose ``nonzero`` is ``True`` or ``False`` is a claim; one whose ``nonzero`` is
    ``None`` is undecided. A claim is true when it matches the truth: for a ``"total"`` record,
    whether the treatment is an ancestor of the outcome in the true graph; for a ``"direct"``
    record, whether the treatment is a parent of the outcome.

    With ``pairs``, the candidate pairs the claims are drawn from, each pair is also a query of
    :func:`~andrey.metrics.answer_scores`: a claim is its answer, an undecided record answers
    "cannot tell" (incomplete), and a pair with no record is abstained. The base-rate policy
    makes the answer's own kind of claim about every pair: for ancestor claims its net rate is
    ``2b - 1`` with ``b`` the share of pairs whose treatment is an ancestor of the outcome, at
    most 1/2 in any DAG, so the floor is abstaining unless the answer also claims zero effects.

    Parameters
    ----------
    estimated : StructureOutput or list of dict
        A fit whose ``metadata["effects"]`` holds the records, or the records.
    true : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The true DAG, over every variable the records name and possibly more, such as hidden
        ones. A record names a variable by label, read from the true graph's labels, or by
        0-based position. A raw array is a directed adjacency.
    pairs : iterable of (str or int, str or int), optional
        The ordered (treatment, outcome) pairs the claims are drawn from, named as the records
        name them. Every record's pair must be among them, and the records must share one
        ``estimand``.

    Returns
    -------
    dict
        ``"claims"``, ``"true_claims"``, ``"false_claims"``, and ``"undecided"`` counts, and
        ``"precision"``, the share of claims that are true, 1.0 when there are none. With
        ``pairs``, also the keys of :func:`~andrey.metrics.answer_scores` over those pairs.

    Raises
    ------
    ValueError
        If the records break the record's rules (:func:`~andrey.core.effects.check_effects`),
        name a variable the true graph lacks, or the true graph has an edge without a direction;
        with ``pairs``, if a record's pair is not among them or the records mix estimands.

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
    answers: dict[tuple[int, int], bool | None] = {}
    for i, record in enumerate(records):
        a = _position(record["treatment"], labels, len(m), f"effects[{i}]: treatment")
        b = _position(record["outcome"], labels, len(m), f"effects[{i}]: outcome")
        answers[a, b] = record["nonzero"]
        if record["nonzero"] is None:
            found["undecided"] += 1
            continue
        truth = bool((direct if record["estimand"] == "direct" else reach)[a, b])
        found["claims"] += 1
        found["true_claims" if record["nonzero"] == truth else "false_claims"] += 1
    claims = found["claims"]
    scores: dict[str, Any] = {
        **found,
        "precision": found["true_claims"] / claims if claims else 1.0,
    }
    if pairs is None:
        return scores
    estimands = {record["estimand"] for record in records}
    if len(estimands) > 1:
        raise ValueError(f"the records mix estimands {sorted(estimands)}; score each apart")
    truth = direct if estimands == {"direct"} else reach
    oracle = {}
    for a, b in pairs:
        key = (_position(a, labels, len(m), "pairs"), _position(b, labels, len(m), "pairs"))
        oracle[key] = bool(truth[key])
    kinds = sorted({value for value in answers.values() if value is not None})
    return {**scores, **answer_scores(answers, oracle, choices=kinds)}


def true_dag(true: object) -> GraphStructure:
    """The true graph as a :class:`GraphStructure` of kind ``dag``; raise for any other kind."""
    graph = unwrap(true)
    if not isinstance(graph, GraphStructure):
        marks, kind = as_marks(graph)
        graph = GraphStructure.from_numpy(marks, kind=kind)
    if graph.kind != "dag":
        raise ValueError(f"the true graph must be a dag, got a {graph.kind}")
    return graph


def hidden_nodes(
    dag: GraphStructure, latent: Collection[str | int] | None, selection: Collection[str | int]
) -> tuple[set[int], set[int]]:
    """The positions of the hidden nodes (``latent``, or those ``node_types`` flags) and of the
    selection nodes."""
    if latent is not None:
        hidden = set(_queries.nodes(dag, latent, "latent"))
    elif dag.node_types is not None:
        hidden = set(np.flatnonzero(dag.node_types == LATENT).tolist())
    else:
        hidden = set()
    return hidden, set(_queries.nodes(dag, selection, "selection"))


def target_oracle(
    true: object,
    target: str | int,
    answer: str,
    *,
    latent: Collection[str | int] | None = None,
    selection: Collection[str | int] = (),
) -> GraphStructure:
    """Return the answer at ``target`` that a method defined by ``answer`` gives on the true graph.

    The answer has the shape a target-centered method returns: a graph over the observed
    variables that holds only the edges at the target. For a set it is the undirected star that
    joins the target to each member; otherwise it is the target's edges in the graph of that kind.

    - ``"markov_blanket"``: the target's Markov blanket, the fewest variables that separate it
      from the rest: in a DAG its parents, children, and children's other parents; with hidden or
      selection variables, the blanket in the maximal ancestral graph over the observed variables
      (:meth:`~andrey.GraphStructure.markov_blanket`).
    - ``"parents_and_children"``: the variables adjacent to the target, those that no set of
      observed variables separates from it: in a DAG its parents and children; with hidden or
      selection variables, its neighbors in the maximal ancestral graph.
    - ``"dag"``: the target's edges in the true DAG (no hidden or selection variables).
    - ``"cpdag"``: the target's edges in the CPDAG of the true DAG, the edges every DAG of its
      Markov equivalence class shares directed and the rest undirected (no hidden or selection
      variables).
    - ``"pag"``: the target's edges in the PAG over the observed variables, after the hidden
      variables are marginalized and the selection variables conditioned on: a tail or arrowhead
      every maximal ancestral graph of the class has, a circle where they differ.

    Parameters
    ----------
    true : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The true DAG over every variable, hidden and selection ones included. A raw array is a
        directed adjacency: a nonzero ``A[i, j]`` is the edge ``i -> j``.
    target : str or int
        The target, by label or by 0-based position in ``true``; an observed variable.
    answer : {"markov_blanket", "parents_and_children", "dag", "cpdag", "pag"}
        What the method answers.
    latent : collection of str or int, optional
        The hidden variables, each a label or a 0-based position. Default: those the true graph's
        ``node_types`` flags as latent, none when it flags none.
    selection : collection of str or int, default=()
        The selection variables: variables the data are conditioned on.

    Returns
    -------
    GraphStructure
        Over the observed variables, in their order and with their labels; of kind
        ``"undirected"`` for a set, else of the kind ``answer`` names.

    Raises
    ------
    ValueError
        If ``answer`` is unknown, the true graph is not a DAG, the target is hidden or a
        selection variable, or ``"dag"`` or ``"cpdag"`` is asked with hidden or selection
        variables.

    Examples
    --------
    >>> import numpy as np
    >>> dag = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])  # 0 -> 2 <- 1
    >>> target_oracle(dag, 0, "markov_blanket").neighbors(0).tolist()
    [1, 2]
    >>> target_oracle(dag, 0, "cpdag").parents(2)
    [0]
    """
    if answer not in TARGET_ANSWERS:
        raise ValueError(f"answer must be one of {TARGET_ANSWERS}, got {answer!r}")
    dag = true_dag(true)
    t = _queries.node(dag, target, "target")
    hidden, chosen = hidden_nodes(dag, latent, selection)
    if t in hidden | chosen:
        raise ValueError(f"the target {target!r} is a hidden or selection variable")
    projected = bool(hidden or chosen)
    if answer in ("dag", "cpdag") and projected:
        raise ValueError(f"a {answer} answer assumes no hidden or selection variables")
    kept = [i for i in range(dag.n_nodes) if i not in hidden | chosen]
    project = {"latent": sorted(hidden), "selection": sorted(chosen)}
    if answer == "pag":
        graph = dag.to_pag(**project)
    elif projected:
        graph = dag.to_mag(**project)
    elif answer == "cpdag":
        graph = GraphStructure.from_numpy(to_cpdag(dag).to_numpy(), kind="cpdag", labels=dag.labels)
    else:
        graph = dag
    k = kept.index(t)
    marks = graph.to_numpy()
    star = np.zeros_like(marks)
    if answer in ("markov_blanket", "parents_and_children"):
        found = graph.markov_blanket(k, index=True) if answer == "markov_blanket" else None
        members = [int(j) for j in (found if found is not None else graph.neighbors(k))]
        star[k, members] = TAIL
        star[members, k] = TAIL
        return GraphStructure.from_numpy(star, kind="undirected", labels=graph.labels)
    star[k] = marks[k]
    star[:, k] = marks[:, k]
    return GraphStructure.from_numpy(star, kind=graph.kind, labels=graph.labels)


def target_answer_scores(
    estimated: object, oracle: object, target: str | int
) -> dict[str, dict[str, Any]]:
    """Score a target-centered answer query by query against the oracle answer at the target.

    Two kinds of query, each scored by :func:`~andrey.metrics.answer_scores`:

    - **adjacency**, one per other variable: whether it is joined to the target. A set answer
      answers every one (a variable left out is answered "not a member"), so its wrong answers
      are its missing and extra members. The base-rate policy here is mostly the empty set.
    - **marks**, two per edge of the oracle at the target, the mark at each end: ``"tail"``,
      ``"arrowhead"``, or "cannot tell": a circle, or either end of an undirected edge in a
      CPDAG (an undirected edge in a PAG has two tails, which are decided). An end of an edge
      the answer lacks is abstained; an undirected edge or a circle where the oracle decides the
      mark is incomplete, never wrong; a decided mark where the oracle cannot tell is wrong. An
      undirected answer has no marks.

    Parameters
    ----------
    estimated : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The answer, only its edges at the target read. A raw array is a directed adjacency.
    oracle : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The oracle answer of the same kind over the same variables, such as
        :func:`target_oracle`'s.
    target : str or int
        The target, by label or by 0-based position.

    Returns
    -------
    dict
        ``"adjacency"`` and ``"marks"``, each the dict :func:`~andrey.metrics.answer_scores`
        returns.

    Raises
    ------
    ValueError
        If the two differ in size, labels, or kind, or ``target`` names no node.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import GraphStructure
    >>> dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # 0 -> 1 -> 2
    >>> oracle = target_oracle(dag, 1, "cpdag")  # 0 - 1 - 2: no mark decided
    >>> answer = np.array([[0, 1, 0], [2, 0, 1], [0, 1, 0]])  # 0 -> 1 - 2
    >>> s = target_answer_scores(GraphStructure.from_numpy(answer, kind="cpdag"), oracle, 1)
    >>> s["adjacency"]["correct"], s["marks"]["correct"], s["marks"]["wrong"]
    (2, 2, 2)
    """
    queries = target_queries(estimated, oracle, target)
    return {
        part: answer_scores(*queries[part], choices=TARGET_CHOICES[part])
        for part in ("adjacency", "marks")
    }


def target_queries(
    estimated: object, oracle: object, target: str | int
) -> dict[str, tuple[dict[Any, Any], dict[Any, Any]]]:
    """The queries :func:`target_answer_scores` scores, as ``{part: (answers, oracle)}``.

    ``"adjacency"`` is keyed by the other variable's position, its answers ``True`` or ``False``;
    ``"marks"`` by ``(position, end)``, ``end`` 0 for the mark at the target and 1 for the mark at
    the other variable, its answers ``"tail"``, ``"arrowhead"``, or ``None``. A campaign pools
    queries across targets and datasets from these.
    """
    m_est, k_est, m_or, k_or = two_marks(estimated, oracle)
    if k_est != k_or:
        raise ValueError(f"the answer is a {k_est} and the oracle a {k_or}; give them one kind")
    labels = _labels(estimated) or _labels(oracle)
    t = _position(target, labels, len(m_est), "target")
    others = [j for j in range(len(m_est)) if j != t]
    joined_est = {j: bool(m_est[t, j] or m_est[j, t]) for j in others}
    joined_or = {j: bool(m_or[t, j] or m_or[j, t]) for j in others}
    mark_answers: dict[tuple[int, int], str | None] = {}
    mark_oracle: dict[tuple[int, int], str | None] = {}
    if k_or != "undirected":
        for j in others:
            if not joined_or[j]:
                continue
            for end, (a, b) in enumerate(((t, j), (j, t))):
                mark_oracle[j, end] = _mark(m_or, a, b, k_or)
                if joined_est[j]:
                    mark_answers[j, end] = _mark(m_est, a, b, k_est)
    return {"adjacency": (joined_est, joined_or), "marks": (mark_answers, mark_oracle)}


def _mark(m: np.ndarray, a: int, b: int, kind: str) -> str | None:
    """The mark at ``a`` on the edge ``a``-``b``: ``"tail"``, ``"arrowhead"``, or ``None`` for a
    circle and for an undirected edge of a CPDAG."""
    if m[a, b] == CIRCLE or (kind == "cpdag" and m[a, b] == TAIL and m[b, a] == TAIL):
        return None
    return "tail" if m[a, b] == TAIL else "arrowhead"
