"""Oracle answers of local methods and the scores of answers that may abstain."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from andrey import GraphStructure
from andrey.core.effects import effect_record
from andrey.metrics import (
    answer_scores,
    claim_scores,
    effect_oracle,
    effect_scores,
    pooled_answer_scores,
    target_answer_scores,
    target_oracle,
)
from andrey.metrics.answers import CANNOT_TELL


def _dag(labels: str, edges: str) -> GraphStructure:
    """A labeled DAG from ``"A>B B>C"``."""
    names = labels.split()
    ix = {name: i for i, name in enumerate(names)}
    marks = np.zeros((len(names), len(names)), dtype=int)
    for edge in edges.split():
        a, b = edge.split(">")
        marks[ix[a], ix[b]], marks[ix[b], ix[a]] = 1, 2
    return GraphStructure.from_numpy(marks, kind="dag", labels=tuple(names))


def _set(graph: GraphStructure, target: str) -> set[str]:
    return {graph.labels[j] for j in graph.neighbors(target).tolist()}


def _ends(graph: GraphStructure) -> set[tuple]:
    return set(graph.oriented_edges())


# A collider at the target and a spouse through its child.
COLLIDER = _dag("A B T C S", "A>T B>T T>C S>C")
# A chain: every edge undirected in the class.
CHAIN = _dag("U T W", "U>T T>W")
# A hidden common cause of X and T, each with a parent of its own.
HIDDEN = _dag("A X H T B", "H>X H>T A>X B>T")
# A selection node with two parents.
SELECTED = _dag("X T S", "X>S T>S")


def test_the_oracle_sets_of_a_collider_at_the_target():
    assert _set(target_oracle(COLLIDER, "T", "markov_blanket"), "T") == {"A", "B", "C", "S"}
    assert _set(target_oracle(COLLIDER, "T", "parents_and_children"), "T") == {"A", "B", "C"}
    cpdag = target_oracle(COLLIDER, "T", "cpdag")
    assert cpdag.kind == "cpdag"
    assert _ends(cpdag) == {("A", "T", "directed"), ("B", "T", "directed"), ("T", "C", "directed")}
    # The PAG leaves the far ends of the collider's edges open.
    assert _ends(target_oracle(COLLIDER, "T", "pag")) == {
        ("A", "T", "partially_directed"),
        ("B", "T", "partially_directed"),
        ("T", "C", "directed"),
    }


def test_an_undirected_neighbor_has_no_decided_mark():
    oracle = target_oracle(CHAIN, "T", "cpdag")
    assert _ends(oracle) == {("U", "T", "undirected"), ("T", "W", "undirected")}
    # The true DAG's direction is a mark the class does not decide: wrong, not correct.
    answer = GraphStructure.from_numpy(target_oracle(CHAIN, "T", "dag").to_numpy(), kind="cpdag")
    marks = target_answer_scores(answer, oracle, "T")["marks"]
    assert marks["wrong"] == 4 and marks["correct"] == 0


def test_a_hidden_common_cause():
    assert _set(target_oracle(HIDDEN, "T", "markov_blanket", latent=["H"]), "T") == {
        "A",
        "X",
        "B",
    }
    assert _set(target_oracle(HIDDEN, "T", "parents_and_children", latent=["H"]), "T") == {"X", "B"}
    pag = target_oracle(HIDDEN, "T", "pag", latent=["H"])
    assert pag.labels == ("A", "X", "T", "B")
    assert _ends(pag) == {("X", "T", "bidirected"), ("B", "T", "partially_directed")}
    with pytest.raises(ValueError, match="hidden"):
        target_oracle(HIDDEN, "T", "cpdag", latent=["H"])
    with pytest.raises(ValueError, match="hidden or selection"):
        target_oracle(HIDDEN, "H", "pag", latent=["H"])


def test_a_selection_node_joins_its_parents():
    assert _set(target_oracle(SELECTED, "T", "parents_and_children"), "T") == {"S"}
    assert _set(target_oracle(SELECTED, "T", "parents_and_children", selection=["S"]), "T") == {"X"}
    assert _ends(target_oracle(SELECTED, "T", "pag", selection=["S"])) == {("X", "T", "circle")}


def test_the_oracle_answer_scores_full_marks():
    cases = [
        (COLLIDER, "T", kind, {}) for kind in ("markov_blanket", "parents_and_children", "cpdag")
    ]
    cases += [(HIDDEN, "T", "pag", {"latent": ["H"]}), (CHAIN, "T", "cpdag", {})]
    for graph, target, kind, extra in cases:
        oracle = target_oracle(graph, target, kind, **extra)
        for part in target_answer_scores(oracle, oracle, target).values():
            assert part["wrong"] == part["incomplete"] == part["abstained"] == 0
            assert part["net"] == part["queries"]


def test_a_set_answer_scores_its_missing_and_extra_members_as_wrong():
    oracle = target_oracle(COLLIDER, "T", "parents_and_children")  # {A, B, C}
    star = np.zeros((5, 5), dtype=int)
    for j in (0, 4):  # {A, S}: one right member, one extra, two missing
        star[2, j] = star[j, 2] = 1
    answer = GraphStructure.from_numpy(star, kind="undirected", labels=COLLIDER.labels)
    s = target_answer_scores(answer, oracle, "T")["adjacency"]
    assert (s["queries"], s["correct"], s["wrong"]) == (4, 1, 3)
    # The base-rate policy is the empty set here only when members are the minority; with 3 of 4
    # it is "every other variable is a member".
    assert s["base_rate_answer"] == "True" and s["floor_rate"] == pytest.approx(0.5)
    assert target_answer_scores(answer, oracle, "T")["marks"]["queries"] == 0


def test_always_abstaining_scores_zero_net():
    oracle = {i: v for i, v in enumerate([True, False, False, None])}
    s = answer_scores({}, oracle)
    assert (s["net"], s["abstained"], s["coverage"]) == (0, 4, 0.0)
    assert np.isnan(s["risk"])


@pytest.mark.parametrize("constant", [True, False, None])
def test_the_base_rate_policy_scores_its_definition(constant):
    oracle = dict(enumerate([True, False, False, False, None]))
    s = answer_scores(dict.fromkeys(oracle, constant), oracle)
    hits = sum(v == constant for v in oracle.values())
    expected = hits if constant is None else 2 * hits - len(oracle)
    assert (
        s["net"] == expected == s["policy_net"][CANNOT_TELL if constant is None else str(constant)]
    )
    assert s["base_rate_answer"] == "False" and s["base_rate"] == pytest.approx(3 / 5)
    assert s["floor_rate"] == pytest.approx(max(0, 2 * 3 - 5) / 5)


def test_coverage_and_risk_move_as_answers_are_withheld():
    oracle = dict(enumerate([True, True, False, False]))
    answers = {0: True, 1: True, 2: True, 3: False}  # one wrong of four
    full = answer_scores(answers, oracle)
    assert (full["coverage"], full["risk"]) == (1.0, 0.25)
    no_wrong = answer_scores({k: v for k, v in answers.items() if k != 2}, oracle)
    assert (no_wrong["coverage"], no_wrong["risk"]) == (0.75, 0.0)
    no_right = answer_scores({k: v for k, v in answers.items() if k != 0}, oracle)
    assert (no_right["coverage"], no_right["risk"]) == (0.75, pytest.approx(1 / 3))
    for s in (full, no_wrong, no_right):
        assert s["net_rate"] == pytest.approx(s["coverage"] * (1 - 2 * s["risk"]))


def test_incomplete_is_never_wrong():
    oracle = target_oracle(COLLIDER, "T", "pag")
    marks = oracle.to_numpy()
    vague = np.where(marks != 0, 3, 0)  # every mark a circle
    s = target_answer_scores(GraphStructure.from_numpy(vague, kind="pag"), oracle, "T")["marks"]
    assert s["wrong"] == 0 and s["incomplete"] == 4 and s["correct"] == 2


def test_scores_do_not_depend_on_column_order_or_labels():
    rng = np.random.default_rng(0)
    order = rng.permutation(5)
    names = tuple(f"v{k}" for k in range(5))
    moved = COLLIDER.to_numpy()[np.ix_(order, order)]
    renamed = GraphStructure.from_numpy(moved, kind="dag", labels=names)
    target = names[list(order).index(2)]
    for kind in ("markov_blanket", "cpdag", "pag"):
        # An answer that differs from the oracle: one of the target's edges dropped.
        answer = target_oracle(COLLIDER, "T", kind).to_numpy()
        answer[2, 0] = answer[0, 2] = 0
        moved_answer = answer[np.ix_(order, order)]
        before = target_answer_scores(
            GraphStructure.from_numpy(answer, kind=_answer_kind(kind), labels=COLLIDER.labels),
            target_oracle(COLLIDER, "T", kind),
            "T",
        )
        after = target_answer_scores(
            GraphStructure.from_numpy(moved_answer, kind=_answer_kind(kind), labels=names),
            target_oracle(renamed, target, kind),
            target,
        )
        assert repr(before) == repr(after) and before["adjacency"]["wrong"] == 1


def _answer_kind(answer: str) -> str:
    return "undirected" if answer == "markov_blanket" else answer


def test_pooling_equals_scoring_the_union():
    a_oracle, b_oracle = {0: True, 1: False, 2: None}, {3: False, 4: False}
    a_answers, b_answers = {0: True, 1: True}, {3: False}
    pooled = pooled_answer_scores(
        [answer_scores(a_answers, a_oracle), answer_scores(b_answers, b_oracle)]
    )
    whole = answer_scores({**a_answers, **b_answers}, {**a_oracle, **b_oracle})
    assert pooled == whole


# Pairs. X <- A -> B -> Y and X -> Y: {A}, {B}, {A, B} are valid; the optimal set is {B}.
TWO_SETS = _dag("A X B Y", "A>X A>B B>Y X>Y")


def test_a_pair_with_two_valid_adjustment_sets():
    x, y = 1, 3
    valid = [
        z
        for k in range(3)
        for z in itertools.combinations((0, 2), k)
        if TWO_SETS.is_adjustment_set(x, y, z)
    ]
    assert valid == [(0,), (2,), (0, 2)]
    record = effect_oracle(TWO_SETS, "X", "Y", identified_from="dag")
    assert record["identifiable"] and record["adjustment_sets"] == [["B"]]
    for sets, outcome in (([["A"]], "correct"), ([["B"]], "correct"), ([[]], "wrong")):
        answer = [dict(record, adjustment_sets=sets, set_kind="valid")]
        assert effect_scores(answer, [record], true=TWO_SETS)["adjustment"][outcome] == 1
    optimal_a = [dict(record, adjustment_sets=[["A"]])]
    s = effect_scores(optimal_a, [record])["adjustment"]
    assert s["wrong"] == 1 and (s["member_tp"], s["member_fp"], s["member_fn"]) == (0, 1, 1)


def test_a_pair_whose_effect_is_not_identifiable():
    # From the class: X - Y could point either way.
    lone = _dag("X Y", "X>Y")
    record = effect_oracle(lone, "X", "Y")
    assert record["identifiable"] is False and record["adjustment_sets"] == []
    assert record["nonzero"] is True
    # From the DAG: a hidden common cause no observed set blocks.
    confounded = _dag("X H Y", "H>X H>Y X>Y")
    record = effect_oracle(
        confounded, "X", "Y", identified_from="dag", set_kind="valid", latent=["H"]
    )
    assert record["identifiable"] is False and record["adjustment_sets"] == []
    # "Not identifiable" is an answer: right where the oracle says so; a set there is wrong.
    said = [dict(record, identifiable=False)]
    assert effect_scores(said, [record])["identifiable"]["correct"] == 1
    guessed = [dict(record, identifiable=True, adjustment_sets=[[]], set_kind="valid")]
    s = effect_scores(guessed, [record], true=confounded)
    assert s["identifiable"]["wrong"] == 1 and s["adjustment"]["wrong"] == 1


def test_a_wider_family_of_parent_sets_is_incomplete():
    record = effect_oracle(CHAIN, "T", "W", set_kind="local_parents")
    assert record["adjustment_sets"] == [[], ["U"]]
    narrower = [dict(record, adjustment_sets=[["U"]])]  # leaves out a parent set the class allows
    assert effect_scores(narrower, [record])["adjustment"]["wrong"] == 1
    oracle = effect_oracle(_dag("U T W V", "U>T T>W V>T"), "T", "W", set_kind="local_parents")
    assert oracle["adjustment_sets"] == [["U", "V"]]
    two = [dict(oracle, adjustment_sets=[["U", "V"], ["U"]], identifiable=None)]
    s = effect_scores(two, [oracle])
    assert s["adjustment"]["incomplete"] == 1 and s["identifiable"]["incomplete"] == 1


def test_ancestor_claims_against_their_base_rate():
    dag = _dag("A B C", "A>B B>C")
    pairs = list(itertools.permutations(dag.labels, 2))
    every = [_claim(dag, a, b) for a, b in pairs]
    s = claim_scores(every, dag, pairs=pairs)
    assert (s["true_claims"], s["false_claims"]) == (3, 3)
    assert s["base_rate"] == pytest.approx(0.5) and s["floor_rate"] == 0.0 and s["net"] == 0
    one = [_claim(dag, "A", "C")]
    s = claim_scores(one, dag, pairs=pairs)
    assert (s["net"], s["abstained"], s["coverage"]) == (1, 5, pytest.approx(1 / 6))
    with pytest.raises(ValueError, match="oracle lacks"):
        claim_scores(one, dag, pairs=[("A", "B")])
    # A dataset with no claims pools as abstaining, not as a policy that never answers right.
    parts = [claim_scores(one, dag, pairs=pairs), claim_scores([], dag, pairs=pairs)]
    keys = [k for k in parts[0] if k in pooled_answer_scores(parts)]
    union = answer_scores(
        {("A", "C", 0): True},
        {(a, b, part): truth for part in (0, 1) for (a, b), truth in _ancestry(dag, pairs).items()},
        choices=(True,),
    )
    assert {k: pooled_answer_scores(parts)[k] for k in keys} == {k: union[k] for k in keys}


def _ancestry(dag: GraphStructure, pairs: list) -> dict:
    """Whether the first of each pair is an ancestor of the second."""
    return {(a, b): a in dag.ancestors(b) for a, b in pairs}


def _claim(dag: GraphStructure, a: str, b: str) -> dict:
    """The claim that ``a`` is an ancestor of ``b``."""
    ix = dag.labels.index
    return effect_record(ix(a), ix(b), estimand="total", nonzero=True, name=dag.labels.__getitem__)


def test_an_effects_oracle_scores_full_marks():
    for kind, given in (("optimal", "cpdag"), ("valid", "dag"), ("local_parents", "cpdag")):
        records = [
            effect_oracle(TWO_SETS, a, b, set_kind=kind, identified_from=given)
            for a, b in itertools.permutations(TWO_SETS.labels, 2)
        ]
        for part in effect_scores(records, records, true=TWO_SETS).values():
            assert part["wrong"] == part["incomplete"] == part["abstained"] == 0
