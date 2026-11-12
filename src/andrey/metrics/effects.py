"""Oracle answers and scores for the pair answers a method lists in ``metadata["effects"]``.

:func:`effect_oracle` gives the record (:mod:`andrey.core.effects`) a method should return for a
treatment-outcome pair, computed on the true graph; :func:`effect_scores` scores a method's
records against the oracle records field by field.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Literal

import numpy as np

from andrey.core import _queries
from andrey.core.effects import check_effects, effect_record
from andrey.core.output import StructureOutput

from . import _adjustment as adj
from .answers import answer_scores, tally
from .graph import to_cpdag
from .local import _position, hidden_nodes, true_dag

#: What an oracle record is identified from: the true DAG's class, or the true DAG itself.
IDENTIFIED_FROM: tuple[str, ...] = ("cpdag", "dag")

#: The constant answer of the base-rate policy for adjustment sets: adjust for nothing.
EMPTY_SET = "empty set"


def effect_oracle(
    true: object,
    treatment: str | int,
    outcome: str | int,
    *,
    estimand: Literal["total", "direct"] = "total",
    set_kind: str | None = None,
    identified_from: str = "cpdag",
    latent: Collection[str | int] | None = None,
) -> dict[str, Any]:
    """Return the ``effects`` record a method should give for one pair, computed on the true DAG.

    ``nonzero`` is the truth: for the ``"total"`` effect, whether the treatment is an ancestor of
    the outcome in the true DAG; for the ``"direct"`` effect, whether it is a parent.
    ``identifiable`` and the sets follow what the method is taken to know:

    - ``identified_from="cpdag"``, the Markov equivalence class of the true DAG (no hidden
      variables). The total effect is identifiable when one adjustment set is valid in every DAG
      of the class: the CPDAG is amenable for the pair, no possibly directed path from the
      treatment to the outcome starting with an undirected edge (Perkovic et al. 2018), and the
      outcome is not a parent of the treatment, an edge no set can block. The direct effect is
      identifiable when the outcome has no undirected edge, so that its parents are the same in
      every DAG of the class.
    - ``identified_from="dag"``, the true DAG, with the hidden variables kept out of every set.
      The total effect is identifiable when some set of observed variables is valid: then the
      observed ancestors of the pair that are not forbidden are one (van der Zander et al. 2014).
      The direct effect is identifiable when every parent of the outcome is observed.

    The sets, by ``set_kind``: ``"optimal"``, the optimal adjustment set (Henckel et al. 2022),
    the parents of the nodes on causal paths from the treatment to the outcome that are neither
    those nodes' descendants nor the treatment, valid in every DAG of an amenable class;
    ``"valid"``, one valid set (the optimal set from the class, the observed ancestors that are
    not forbidden from the DAG), since an answer of this kind is judged by its validity;
    ``"local_parents"``, the parent sets of the treatment the class allows (Maathuis et al.
    2009), its parents with each clique of its undirected neighbors, the outcome left out of a
    set that holds it; ``"parents_of_outcome"``, the outcome's parents but the treatment, for
    the direct effect. A total effect that is zero or not identifiable has no optimal or valid
    set.

    Parameters
    ----------
    true : StructureOutput, GraphStructure, or array-like of shape (n, n)
        The true DAG, hidden variables included. A raw array is a directed adjacency.
    treatment : str or int
        The treatment, by label or by 0-based position in ``true``; an observed variable.
    outcome : str or int
        The outcome, by label or by 0-based position in ``true``; an observed variable.
    estimand : {"total", "direct"}, default="total"
        The effect.
    set_kind : {"optimal", "valid", "local_parents", "parents_of_outcome"}, optional
        The sets the method answers with. Default: ``"optimal"`` for the total effect,
        ``"parents_of_outcome"`` for the direct effect, the only kind it takes.
        ``"local_parents"`` needs ``identified_from="cpdag"``, and ``"optimal"`` no hidden
        variables.
    identified_from : {"cpdag", "dag"}, default="cpdag"
        What the method is taken to identify the effect from.
    latent : collection of str or int, optional
        The hidden variables. Default: those the true graph's ``node_types`` flags as latent.

    Returns
    -------
    dict
        An ``effects`` record (:func:`~andrey.core.effects.effect_record`) naming variables by
        the true graph's labels, or by position in it when it has none.

    Raises
    ------
    ValueError
        If the true graph is not a DAG, the pair is not two observed variables, or the options
        do not fit together (see ``set_kind``; ``"cpdag"`` with hidden variables).

    References
    ----------
    Henckel, L., Perkovic, E., and Maathuis, M. H. (2022). Graphical criteria for efficient total
    effect estimation via adjustment in causal linear models. Journal of the Royal Statistical
    Society Series B 84(2), 579-599.

    Maathuis, M. H., Kalisch, M., and Buhlmann, P. (2009). Estimating high-dimensional
    intervention effects from observational data. The Annals of Statistics 37(6A), 3133-3164.

    Perkovic, E., Textor, J., Kalisch, M., and Maathuis, M. H. (2018). Complete graphical
    characterization and construction of adjustment sets in Markov equivalence classes of
    ancestral graphs. Journal of Machine Learning Research 18(220), 1-62.

    van der Zander, B., Liskiewicz, M., and Textor, J. (2014). Constructing separators and
    adjustment sets in ancestral graphs. Proceedings of the Thirtieth Conference on Uncertainty
    in Artificial Intelligence, 907-916.

    Examples
    --------
    >>> import numpy as np
    >>> dag = np.zeros((4, 4), dtype=int)
    >>> dag[0, 1] = dag[0, 2] = dag[1, 3] = dag[2, 3] = 1  # 1 <- 0 -> 2, 1 -> 3 <- 2
    >>> effect_oracle(dag, 1, 3)["identifiable"]  # in another DAG of the class, 1 -> 0 -> 2 -> 3
    False
    >>> effect_oracle(dag, 1, 3, identified_from="dag")["adjustment_sets"]
    [[2]]
    """
    if identified_from not in IDENTIFIED_FROM:
        raise ValueError(
            f"identified_from must be one of {IDENTIFIED_FROM}, got {identified_from!r}"
        )
    dag = true_dag(true)
    x = _queries.node(dag, treatment, "treatment")
    y = _queries.node(dag, outcome, "outcome")
    hidden, _ = hidden_nodes(dag, latent, ())
    if x == y or x in hidden or y in hidden:
        raise ValueError("the treatment and the outcome must be two observed variables")
    if identified_from == "cpdag" and hidden:
        raise ValueError("identified_from='cpdag' assumes no hidden variables")
    kind = set_kind or ("optimal" if estimand == "total" else "parents_of_outcome")
    allowed = (
        ("optimal", "valid", "local_parents") if estimand == "total" else ("parents_of_outcome",)
    )
    if kind not in allowed:
        raise ValueError(f"a {estimand} effect takes set_kind in {allowed}, got {kind!r}")
    if (kind == "local_parents" and identified_from != "cpdag") or (kind == "optimal" and hidden):
        raise ValueError(f"set_kind={kind!r} does not fit identified_from={identified_from!r}")
    marks = dag.to_numpy()
    D = adj.directed(marks)
    cpdag = to_cpdag(dag).to_numpy()
    directed_c, undirected_c = adj.cpdag_parts(cpdag)
    observed = [i for i in range(dag.n_nodes) if i not in hidden]
    if estimand == "total":
        nonzero = x in adj.reach(D.T, [y])
        if identified_from == "cpdag":
            identifiable = adj.amenable(cpdag, x, y) and not directed_c[y, x]
        else:
            identifiable = adj.is_valid(D, x, y, adj.canonical_set(D, x, y, observed))
        if kind == "local_parents":
            sets = [[v for v in s if v != y] for s in adj.local_parent_sets(cpdag, x)]
        elif not (identifiable and nonzero):
            sets = []
        elif kind == "optimal" or identified_from == "cpdag":
            sets = [adj.optimal_set(D, x, y)]
        else:
            sets = [adj.canonical_set(D, x, y, observed)]
    else:
        nonzero = bool(D[x, y])
        parents = set(np.flatnonzero(D[:, y]).tolist()) - {x}
        if identified_from == "cpdag":
            identifiable = not undirected_c[y].any()
        else:
            identifiable = not parents & hidden
        sets = [sorted(parents)] if identifiable else []
    labels = dag.labels
    return effect_record(
        x,
        y,
        estimand=estimand,
        nonzero=nonzero,
        identifiable=bool(identifiable),
        adjustment_sets=sets,
        set_kind=kind if sets else None,
        name=int if labels is None else labels.__getitem__,
    )


def effect_scores(
    estimated: object, oracle: object, *, true: object | None = None
) -> dict[str, dict[str, Any]]:
    """Score a method's ``effects`` records against the oracle records, field by field.

    Each pair of the oracle records is a query of :func:`~andrey.metrics.answer_scores` in three
    fields; a pair the method has no record for is abstained, and a field it leaves ``None``
    answers "cannot tell".

    - ``"nonzero"``: whether the effect is nonzero.
    - ``"identifiable"``: whether one adjustment set is valid in every graph allowed. "Not
      identifiable" is an answer, right where the oracle says so.
    - ``"adjustment"``: the sets, for the pairs whose effect the oracle does not call zero. A
      record with no set answers "cannot tell". ``"valid"`` sets are right when each is a valid
      adjustment set in ``true`` (the adjustment criterion: no forbidden variable, and every
      non-causal path blocked); ``"local_parents"`` sets are right when they are the oracle's,
      and incomplete when they hold every oracle set and more, a wider set of possible effects;
      other sets are right when they are the oracle's. The base-rate policy adjusts for nothing.
      Also ``"member_tp"``, ``"member_fp"``, ``"member_fn"``: the variables of one answered set
      that are, are not, and should have been in the oracle's one set, over the pairs where
      both give one set of the same kind.

    Parameters
    ----------
    estimated : StructureOutput or list of dict
        A fit whose ``metadata["effects"]`` holds the records, or the records.
    oracle : StructureOutput or list of dict
        The oracle records, one per pair, such as :func:`effect_oracle`'s, or a fit of the method
        with a d-separation oracle in place of its test. A pair is its treatment, outcome, and
        estimand; the two sides name variables alike.
    true : StructureOutput, GraphStructure, or array-like of shape (n, n), optional
        The true DAG, needed to judge ``"valid"`` sets; records name variables by its labels or
        positions.

    Returns
    -------
    dict
        ``"nonzero"``, ``"identifiable"``, and ``"adjustment"``, each the dict
        :func:`~andrey.metrics.answer_scores` returns.

    Raises
    ------
    ValueError
        If either side breaks the record's rules, names a pair twice, or the method answers a
        pair the oracle lacks; or a ``"valid"`` set comes without ``true``.

    Examples
    --------
    >>> import numpy as np
    >>> dag = np.zeros((4, 4), dtype=int)
    >>> dag[0, 1] = dag[0, 2] = dag[1, 3] = dag[2, 3] = 1  # 1 <- 0 -> 2, 1 -> 3 <- 2
    >>> oracle = [effect_oracle(dag, 1, 3, identified_from="dag")]
    >>> answer = [dict(oracle[0], adjustment_sets=[[2]], set_kind="valid")]
    >>> s = effect_scores(answer, oracle, true=dag)
    >>> s["adjustment"]["correct"], s["identifiable"]["correct"]
    (1, 1)
    """
    est, ora = _keyed(estimated, "estimated"), _keyed(oracle, "oracle")
    fields: dict[str, dict[str, Any]] = {}
    for field in ("nonzero", "identifiable"):
        answers = {k: r[field] for k, r in est.items()}
        fields[field] = answer_scores(
            answers, {k: r[field] for k, r in ora.items()}, choices=(True, False)
        )
    dag = None if true is None else true_dag(true)
    asked = {k: _sets(r) for k, r in ora.items() if r["nonzero"] is not False}
    answers = {k: _sets(r) for k, r in est.items() if k in asked}

    def judge(query: Any, answer: Any, truth: Any) -> str:
        if answer is None:
            return "correct" if truth is None else "incomplete"
        if answer == EMPTY_SET:
            by_validity = dag is not None and query[2] == "total"
            answer = ("valid" if by_validity else "optimal", frozenset({frozenset()}))
        kind, family = answer
        if kind == "valid":
            if dag is None:
                raise ValueError("judging 'valid' sets needs the true DAG: pass true=")
            ok = all(_valid(dag, query, s) for s in family)
            return "correct" if ok else "wrong"
        if truth is None:
            return "wrong"
        if family == truth[1]:
            return "correct"
        return "incomplete" if kind == "local_parents" and family > truth[1] else "wrong"

    scores = tally(answers, asked, judge, [EMPTY_SET] if asked else [])
    tp = fp = fn = 0
    for k, answer in answers.items():
        truth = asked[k]
        if answer is None or truth is None or answer[0] != truth[0]:
            continue
        if len(answer[1]) == 1 and len(truth[1]) == 1:
            (s,), (o,) = answer[1], truth[1]
            tp, fp, fn = tp + len(s & o), fp + len(s - o), fn + len(o - s)
    fields["adjustment"] = {**scores, "member_tp": tp, "member_fp": fp, "member_fn": fn}
    return fields


def _keyed(records: object, what: str) -> dict[tuple[Any, Any, str], dict[str, Any]]:
    """The records of a fit or a list, keyed by (treatment, outcome, estimand)."""
    if isinstance(records, StructureOutput):
        records = records.metadata.get("effects", [])
    check_effects(records)
    assert isinstance(records, list)  # check_effects refuses anything else
    keyed = {}
    for record in records:
        key = (record["treatment"], record["outcome"], record["estimand"])
        if key in keyed:
            raise ValueError(f"{what} names the pair {key} twice")
        keyed[key] = record
    return keyed


def _sets(record: dict[str, Any]) -> tuple[str, frozenset[frozenset[Any]]] | None:
    """A record's sets as (kind, family), ``None`` when it has none."""
    if not record["adjustment_sets"]:
        return None
    return record["set_kind"], frozenset(frozenset(s) for s in record["adjustment_sets"])


def _valid(dag: Any, query: Any, members: frozenset[Any]) -> bool:
    """Whether ``members`` is a valid adjustment set for the query's total effect in ``dag``."""
    treatment, outcome, estimand = query
    if estimand != "total":
        raise ValueError("'valid' sets are judged for total effects only")
    n, labels = dag.n_nodes, dag.labels
    x, y = _position(treatment, labels, n, "treatment"), _position(outcome, labels, n, "outcome")
    z = [_position(v, labels, n, "an adjustment set") for v in members]
    return adj.is_valid(adj.directed(dag.to_numpy()), x, y, z)
