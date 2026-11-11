"""The shared pieces of the local methods: columns, target sets, effects records, and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.api._local import (
    TARGET_SETS,
    pair_columns,
    target_column,
    target_metadata,
    variable_columns,
)
from andrey.core import ARROW, CIRCLE, TAIL, GraphStructure, StructureOutput
from andrey.core.effects import EFFECT_FIELDS, check_effects, effect_record
from andrey.metrics import claim_scores, target_edge_scores, to_cpdag

LABELS = ("a", "b", "c", "d")


def test_columns_by_label_or_position():
    assert target_column("c", LABELS, 4, "m") == target_column(2, None, 4, "m") == 2
    assert pair_columns("d", 0, LABELS, 4, "m") == (3, 0)
    assert variable_columns(["d", "a", 1], LABELS, 4, "m") == [0, 1, 3]
    assert variable_columns([2, 0], None, 4, "m", minimum=2, maximum=2) == [0, 2]


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: target_column(None, LABELS, 4, "m"), ValueError, "m: pass target"),
        (lambda: target_column("z", LABELS, 4, "m"), ValueError, "m: target names column 'z'"),
        (lambda: target_column(1.5, LABELS, 4, "m"), TypeError, "m: target"),
        (lambda: pair_columns("a", None, LABELS, 4, "m"), ValueError, "m: pass treatment"),
        (lambda: pair_columns("a", 0, LABELS, 4, "m"), ValueError, "the same column"),
        (lambda: variable_columns(None, LABELS, 4, "m"), ValueError, "m: pass variables"),
        (lambda: variable_columns("ab", LABELS, 4, "m"), TypeError, "a list of columns"),
        (lambda: variable_columns(["a", 0], LABELS, 4, "m"), ValueError, "a column twice"),
        (lambda: variable_columns(["a"], LABELS, 4, "m"), ValueError, "at least 2"),
        (lambda: variable_columns([0, 1, 2], None, 4, "m", maximum=2), ValueError, "2 columns"),
    ],
)
def test_a_bad_column_argument_says_what_to_pass(call, error, message):
    with pytest.raises(error, match=message):
        call()


def test_target_metadata_lists_each_set_under_its_key():
    out = target_metadata(1, {"parents": [3, 0], "undirected": []}, LABELS.__getitem__)
    assert out == {"target": "b", "parents": ["d", "a"], "undirected": []}
    with pytest.raises(ValueError, match="not a target set"):
        target_metadata(1, {"spouses": [0]}, int)
    with pytest.raises(ValueError, match="holds the target"):
        target_metadata(1, {"parents": [1]}, int)
    assert {"markov_blanket", "parents_and_children", "undetermined"} <= set(TARGET_SETS)


def test_an_effects_record_is_canonical_and_checked():
    record = effect_record(
        3,
        0,
        estimand="total",
        nonzero=True,
        identifiable=True,
        adjustment_sets=[[2, 1], [1], [1, 2]],
        set_kind="valid",
        name=LABELS.__getitem__,
        score=0.9,
    )
    assert list(record) == [*EFFECT_FIELDS, "score"]
    assert record["treatment"] == "d" and record["outcome"] == "a"
    assert record["adjustment_sets"] == [["b"], ["b", "c"]]
    check_effects([record], LABELS)
    empty = effect_record(0, 1, estimand="direct", adjustment_sets=[[]], set_kind="valid")
    assert empty["adjustment_sets"] == [[]]


def _record(**change):
    return {**effect_record(0, 1, estimand="total"), **change}


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ({"a": 1}, "must be a list"),
        ([_record(outcome=0)], "both 0"),
        ([_record(estimand="mean")], "estimand"),
        ([_record(nonzero=1)], "nonzero"),
        ([_record(adjustment_sets=[[0]], set_kind="valid")], "holds the treatment"),
        ([_record(adjustment_sets=[[2], [2]], set_kind="valid")], "listed twice"),
        ([_record(adjustment_sets=[[2]])], "set_kind"),
        ([_record(set_kind="valid")], "set_kind"),
        ([_record(set_kind="best", adjustment_sets=[[2]])], "set_kind"),
        ([{"treatment": 0}], "lacks"),
        ([_record(treatment=True)], "label or position"),
    ],
)
def test_a_broken_effects_record_is_refused(records, message):
    with pytest.raises(ValueError, match=message):
        check_effects(records)


def test_a_column_the_data_lack_is_refused():
    with pytest.raises(ValueError, match="not a column"):
        check_effects([_record()], LABELS)


def _graph(edges: list[tuple[int, int, int, int]], kind: str, n: int = 4) -> GraphStructure:
    """``(i, j, mark at i, mark at j)`` edges over ``n`` labeled nodes."""
    m = np.zeros((n, n), dtype=np.int8)
    for i, j, at_i, at_j in edges:
        m[i, j], m[j, i] = at_i, at_j
    return GraphStructure.from_numpy(m, kind=kind, labels=LABELS[:n])


def test_target_edge_scores_on_a_cpdag():
    """Truth a -> b <- c, b - d; the answer has a -> b, b -> c, b - d, and a - c away from b."""
    truth = _graph([(0, 1, TAIL, ARROW), (2, 1, TAIL, ARROW), (1, 3, TAIL, TAIL)], "cpdag")
    answer = _graph(
        [(0, 1, TAIL, ARROW), (1, 2, TAIL, ARROW), (1, 3, TAIL, TAIL), (0, 2, TAIL, TAIL)], "cpdag"
    )
    s = target_edge_scores(answer, truth, "b")
    assert s["adjacency_f1"] == 1.0
    assert (s["edge_precision"], s["edge_recall"]) == (2 / 3, 2 / 3)
    assert (s["missing"], s["extra"], s["wrong_marks"], s["reversed"], s["shd"]) == (0, 0, 1, 1, 1)
    assert target_edge_scores(answer, truth, 1) == s
    order = [3, 1, 0, 2]
    moved = target_edge_scores(*(_moved(g, order) for g in (answer, truth)), "b")
    assert moved == s


def _moved(graph: GraphStructure, order: list[int]) -> GraphStructure:
    m = graph.to_numpy()[np.ix_(order, order)]
    return GraphStructure.from_numpy(m, kind=graph.kind, labels=tuple(LABELS[i] for i in order))


def test_target_edge_scores_on_a_pag():
    """Truth a o-> b <-> c; the answer has a o-> b, b <-o c, and d o-o b."""
    truth = _graph([(0, 1, CIRCLE, ARROW), (1, 2, ARROW, ARROW)], "pag")
    answer = _graph([(0, 1, CIRCLE, ARROW), (2, 1, CIRCLE, ARROW), (3, 1, CIRCLE, CIRCLE)], "pag")
    s = target_edge_scores(StructureOutput.new(answer), truth, "b")
    assert (s["adjacency_precision"], s["adjacency_recall"]) == (2 / 3, 1.0)
    assert (s["missing"], s["extra"], s["wrong_marks"], s["reversed"], s["shd"]) == (0, 1, 1, 0, 2)
    assert s["edge_recall"] == 0.5


def test_target_edge_scores_wants_the_answer_s_kind():
    dag = np.array([[0, 1], [0, 0]])
    with pytest.raises(ValueError, match="to_cpdag"):
        target_edge_scores(to_cpdag(dag), dag, 0)
    assert target_edge_scores(to_cpdag(dag), to_cpdag(dag), 0)["shd"] == 0


def test_claim_scores_against_the_true_ancestry():
    """Truth a -> b -> c, h -> c with h hidden: a causes c, not directly; c does not cause a."""
    truth = _graph([(0, 1, TAIL, ARROW), (1, 2, TAIL, ARROW), (3, 2, TAIL, ARROW)], "dag")
    records = [
        effect_record(0, 2, estimand="total", nonzero=True, name=LABELS.__getitem__),
        effect_record(0, 2, estimand="direct", nonzero=True, name=LABELS.__getitem__),
        effect_record(2, 0, estimand="total", nonzero=False, name=LABELS.__getitem__),
        effect_record(1, 0, estimand="total", name=LABELS.__getitem__),
    ]
    found = claim_scores(records, truth)
    assert found == {
        "claims": 3,
        "true_claims": 2,
        "false_claims": 1,
        "undecided": 1,
        "precision": 2 / 3,
    }
    empty = GraphStructure.from_numpy(np.zeros((3, 3)), kind="cpdag", labels=LABELS[:3])
    out = StructureOutput.new(empty, metadata={"effects": records[:3]})
    assert claim_scores(out, truth)["claims"] == 3
    assert claim_scores([], truth)["precision"] == 1.0
    with pytest.raises(ValueError, match="DAG"):
        claim_scores(records, to_cpdag(np.array([[0, 1], [0, 0]])))
