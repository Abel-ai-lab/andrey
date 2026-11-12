"""The campaign rules for local answers: the gate, paired risk, risk-coverage, and Wilson."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.metrics import answer_scores
from andrey.metrics.answers import outcomes
from andrey_bench import local_eval as E
from andrey_bench.pairing import BOOTSTRAP, bootstrap_95, compare

# Ten candidates per dataset, two of them members: the base-rate answer is "not a member".
ORACLE = {j: j < 2 for j in range(10)}


def _answer(found: set[int]) -> dict:
    return answer_scores({j: j in found for j in ORACLE}, ORACLE, choices=(True, False))


def test_the_bootstrap_is_the_release_cards_recipe():
    values = np.random.default_rng(5).normal(size=17)
    picks = np.random.default_rng(0).integers(0, values.size, (BOOTSTRAP, values.size))
    assert bootstrap_95(values) == list(np.percentile(values[picks].mean(axis=1), [2.5, 97.5]))
    assert bootstrap_95(np.zeros(4)) == [0.0, 0.0]
    assert compare([1.0] * 12, [0.0] * 12)["verdict"] == "worse"


def test_an_answer_with_more_right_members_than_wrong_passes():
    good = [_answer({0, 1, 5}) for _ in range(12)]  # 2 right, 1 extra: beats the empty set
    g = E.gate(good)
    assert g["floor_policy"] == "False" and g["verdict"] == "pass"
    assert g["mean_D"] == pytest.approx(2 * (2 - 1) / 10)  # 2 (right - extra) / candidates


def test_the_empty_set_and_a_coin_do_not_pass():
    assert E.gate([_answer(set()) for _ in range(12)])["verdict"] == "fail"
    even = [_answer({0, 5}) for _ in range(12)]  # one right, one wrong: the empty set's net
    assert E.gate(even)["verdict"] == "fail"


def test_a_failed_fit_fails_and_few_datasets_are_reported():
    good = [_answer({0, 1}) for _ in range(12)]
    assert E.gate([*good, None])["verdict"] == "fail"
    assert E.gate(good[:5])["verdict"] == "reported"


def test_abstaining_is_the_floor_when_every_constant_answer_loses():
    oracle = {j: j % 2 == 0 for j in range(10)}
    always = [answer_scores({0: True}, oracle) for _ in range(12)]
    g = E.gate(always)
    assert g["floor_policy"] == E.ABSTAIN and g["verdict"] == "pass"


def test_paired_risk_compares_only_the_queries_both_answered():
    oracle = dict.fromkeys(range(10), True)
    # a answers the five easy queries right; b answers all ten, wrong on the five hard ones.
    a = outcomes({j: True for j in range(5)}, oracle)
    b = outcomes({j: j < 5 for j in range(10)}, oracle)
    p = E.paired_risk([(a, b)] * 12)
    assert p["common"] == 60 and p["risk_a"] == p["risk_b"] == 0.0 and p["verdict"] == "tie"
    p = E.paired_risk([(b, a), (None, a)])
    assert p["failed_a"] == 1 and p["paired"] == 1


def test_risk_coverage_keeps_ties_together():
    oracle = {0: True, 1: True, 2: False, 3: False}
    answers = {0: True, 1: True, 2: True, 3: None}
    rows = E.risk_coverage(answers, oracle, {0: 0.9, 1: 0.5, 2: 0.5})
    assert [r["kept"] for r in rows] == [1, 3]
    assert rows[0]["risk"] == 0.0 and rows[1]["risk"] == pytest.approx(1 / 3)
    assert rows[1]["coverage"] == 0.75 and rows[1]["net_rate"] == pytest.approx(1 / 4)


def test_wilson_interval():
    low, high = E.wilson_95(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4) and high == pytest.approx(0.7634, abs=1e-4)
    assert E.wilson_95(0, 10)[0] == 0.0 and E.wilson_95(0, 0) == [0.0, 1.0]
