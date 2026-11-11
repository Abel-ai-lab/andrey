"""The answers of local methods: how an adapter packs one, what scores it, and the contract.

The checks run on hand-built graphs and need only NumPy: the scores of a target's edges with both
marks, of ancestral claims, and what the empty answer scores, which is what a method has to beat.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pytest

from andrey_bench import local_lane as L

_NULL, _TAIL, _ARROW, _CIRCLE = 0, 1, 2, 3


def _marks(d: int, edges: dict[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    """``edges[(i, j)] = (mark at i, mark at j)``."""
    m = np.zeros((d, d), dtype=np.int8)
    for (i, j), (at_i, at_j) in edges.items():
        m[i, j], m[j, i] = at_i, at_j
    return m


# The truth at node 0: 0 -> 1, 0 o-> 2, 0 <-> 3, and an edge that is not at the target.
TRUTH = _marks(
    5,
    {
        (0, 1): (_TAIL, _ARROW),
        (0, 2): (_CIRCLE, _ARROW),
        (0, 3): (_ARROW, _ARROW),
        (1, 4): (_TAIL, _ARROW),
    },
)


def test_an_answer_packs_with_its_test_count_and_unpacks_unchanged():
    marks = TRUTH.copy()
    packed = L.pack_answer(marks, 1_600_000)
    assert packed.shape == (6, 5)
    got, tests = L.unpack_answer(packed)
    np.testing.assert_array_equal(got, marks)
    assert tests == 1_600_000
    # No count reads back as none, a bare square array is an answer without one.
    assert L.unpack_answer(L.pack_answer(marks))[1] is None
    bare, none = L.unpack_answer(marks)
    assert none is None and bare is marks


def test_a_score_matrix_stays_float_and_a_count_of_zero_is_a_count():
    scores = np.zeros((3, 3))
    scores[0, 2] = 4.5
    packed = L.pack_answer(scores, 0)
    assert packed.dtype == np.float64
    got, tests = L.unpack_answer(packed)
    np.testing.assert_array_equal(got, scores)
    assert tests == 0


def test_an_answer_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="square"):
        L.pack_answer(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="d \\+ 1"):
        L.unpack_answer(np.zeros((5, 3)))


def test_the_nonzero_marks_round_trip_through_a_records_triples():
    entries = L.marks_to_entries(TRUTH)
    assert [0, 2, _CIRCLE] in entries and len(entries) == 8
    np.testing.assert_array_equal(L.entries_to_marks(entries, 5), TRUTH)


def test_the_edges_at_a_target_are_its_row_and_column():
    assert L.target_edges(TRUTH, 0) == {
        1: (_TAIL, _ARROW),
        2: (_CIRCLE, _ARROW),
        3: (_ARROW, _ARROW),
    }
    assert L.target_edges(TRUTH, 4) == {1: (_ARROW, _TAIL)}


def test_the_truth_scores_perfectly_at_its_target():
    s = L.target_edge_scores(TRUTH, TRUTH, 0)
    assert (s["est_edges"], s["true_edges"], s["correct_edges"]) == (3, 3, 3)
    assert (s["shd"], s["extra"], s["missing"], s["mismarked"]) == (0, 0, 0, 0)
    assert s["precision"] == s["recall"] == s["f1"] == 1.0 and s["distance"] == 0.0


def test_an_edge_counts_only_with_both_marks_right():
    """The paper's rule: a circle where the truth has a tail, or the reverse, is a wrong edge."""
    est = _marks(
        5,
        {
            (0, 1): (_CIRCLE, _ARROW),  # the truth's tail is a circle: adjacent, not a true edge
            (0, 2): (_CIRCLE, _ARROW),  # right
            (0, 4): (_TAIL, _ARROW),  # extra
        },
    )
    s = L.target_edge_scores(est, TRUTH, 0)
    assert (s["est_edges"], s["true_edges"], s["correct_edges"], s["adj_correct"]) == (3, 3, 1, 2)
    assert (s["extra"], s["missing"], s["mismarked"], s["shd"]) == (1, 1, 1, 3)
    assert s["precision"] == pytest.approx(1 / 3) and s["recall"] == pytest.approx(1 / 3)
    assert s["adj_precision"] == pytest.approx(2 / 3) and s["adj_recall"] == pytest.approx(2 / 3)
    assert s["distance"] == pytest.approx(math.hypot(2 / 3, 2 / 3))


def test_edges_away_from_the_target_are_not_read():
    est = TRUTH.copy()
    est[1, 4], est[4, 1] = _NULL, _NULL  # the edge 1 -> 4 is gone, and it is not the target's
    assert L.target_edge_scores(est, TRUTH, 0)["shd"] == 0
    assert L.target_edge_scores(est, TRUTH, 1)["missing"] == 1


def test_the_empty_answer_misses_every_edge_and_has_no_precision():
    s = L.target_edge_scores(np.zeros_like(TRUTH), TRUTH, 0)
    assert (s["est_edges"], s["missing"], s["shd"], s["empty_shd"]) == (0, 3, 3, 3)
    assert s["recall"] == 0.0 and math.isnan(s["precision"]) and math.isnan(s["distance"])


def test_a_target_with_no_true_edge_has_no_recall():
    s = L.target_edge_scores(_marks(3, {(0, 1): (_TAIL, _ARROW)}), np.zeros((3, 3), np.int8), 2)
    assert s["true_edges"] == 0 and math.isnan(s["recall"]) and s["shd"] == 0


def test_claims_are_scored_among_the_system_columns_only():
    # Column 3 is the context: its pairs, true or claimed, are outside the score.
    truth = np.zeros((4, 4), dtype=bool)
    truth[0, 1] = truth[1, 2] = truth[0, 2] = truth[3, 0] = True
    scores = np.zeros((4, 4))
    scores[0, 1], scores[1, 2], scores[2, 0], scores[3, 0] = 9.0, 5.0, 1.0, 7.0
    s = L.claim_scores(scores, truth, [0, 1, 2])
    assert (s["claims"], s["tp"], s["fp"]) == (3, 2, 1)
    assert (s["positives"], s["pairs"]) == (3, 6)
    assert s["base_rate"] == pytest.approx(0.5)
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 3)


def test_average_precision_follows_the_ranking_and_counts_ties_as_one_threshold():
    truth = np.zeros((4, 4), dtype=bool)
    truth[0, 1] = truth[0, 2] = True
    # Ranked best first: true, false, true, then the rest unclaimed.
    ranked = np.zeros((4, 4))
    ranked[0, 1], ranked[1, 0], ranked[0, 2] = 3.0, 2.0, 1.0
    s = L.claim_scores(ranked, truth, [0, 1, 2, 3])
    assert s["average_precision"] == pytest.approx(0.5 * 1.0 + 0.5 * 2 / 3)
    # Ties: one threshold holds both claims, so the precision there is 1/2 at recall 1/2.
    tied = np.zeros((4, 4))
    tied[0, 1] = tied[1, 0] = 2.0
    assert L.claim_scores(tied, truth, [0, 1, 2, 3])["average_precision"] == pytest.approx(
        0.5 * 0.5 + 0.5 * (2 / 12)
    )


def test_a_method_that_claims_nothing_scores_the_base_rate_floor():
    truth = np.zeros((3, 3), dtype=bool)
    truth[0, 1] = True
    s = L.claim_scores(np.zeros((3, 3)), truth, [0, 1, 2])
    assert s["claims"] == 0 and math.isnan(s["precision"]) and s["recall"] == 0.0
    assert s["base_rate"] == pytest.approx(1 / 6)
    # With every ordered pair tied the curve is one point: precision at the base rate.
    assert s["average_precision"] == pytest.approx(1 / 6)


def test_no_true_pair_leaves_the_ranking_undefined():
    s = L.claim_scores(np.ones((3, 3)), np.zeros((3, 3), dtype=bool), [0, 1, 2])
    assert s["positives"] == 0 and math.isnan(s["average_precision"]) and math.isnan(s["recall"])


class _StandIn:
    """What a method's adapter provides, with the answer read from the target's row of a graph."""

    name = "stand-in"
    package = "test"
    package_version = "0"
    backend = "numpy"
    mode = "serial"
    algorithm = "stand_in"
    output_type = "pag"
    local = True

    def params(self) -> dict[str, Any]:
        return {"alpha": 0.01}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        t = int(params[L.TARGET_PARAM])
        answer = np.zeros_like(TRUTH)
        answer[t], answer[:, t] = TRUTH[t], TRUTH[:, t]
        return L.pack_answer(answer, 12)

    def to_structure(self, native: np.ndarray) -> Any:
        raise NotImplementedError


class _ClaimStandIn:
    output_type = "claims"

    def params(self) -> dict[str, Any]:
        return {"alpha": 0.01}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        scores = np.zeros((data.shape[1], data.shape[1]))
        scores[0, 1] = 2.0
        return L.pack_answer(scores)


def test_an_adapter_that_fills_the_contract_is_one_and_scores_through_the_packing():
    adapter = _StandIn()
    assert isinstance(adapter, L.LocalAdapter)
    answer, tests = L.unpack_answer(
        adapter.fit(np.zeros((10, 5)), {**adapter.params(), "target": 0})
    )
    assert tests == 12
    assert L.target_edge_scores(answer, TRUTH, 0)["shd"] == 0
    assert (
        L.target_edge_scores(answer, TRUTH, 1)["shd"] == 1
    )  # only the target's edges were answered
    assert isinstance(_ClaimStandIn(), L.ClaimAdapter)
    assert not isinstance(object(), L.LocalAdapter)
