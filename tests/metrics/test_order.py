"""Causal-order metrics: ancestral accuracy and Kendall tau."""

from __future__ import annotations

import pytest

from andrey.metrics import (
    causal_order_accuracy,
    causal_order_kendall_tau,
    causal_order_scores,
)

CHAIN = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]  # 0 -> 1 -> 2, ancestral pairs (0,1) (0,2) (1,2)


def test_correct_order_scores_one():
    assert causal_order_accuracy([0, 1, 2], CHAIN) == 1.0
    assert causal_order_kendall_tau([0, 1, 2], CHAIN) == 1.0


def test_reversed_order_scores_zero():
    assert causal_order_accuracy([2, 1, 0], CHAIN) == 0.0
    assert causal_order_kendall_tau([2, 1, 0], CHAIN) == -1.0


def test_partial_order_fraction():
    # order [1, 0, 2]: pair (0,1) reversed, (0,2) ok, (1,2) ok -> 2/3 correct.
    assert causal_order_accuracy([1, 0, 2], CHAIN) == pytest.approx(2 / 3)


def test_scores_report_comparable_pairs():
    report = causal_order_scores([0, 1, 2], CHAIN)
    assert report["n_comparable_pairs"] == 3
    assert report["kendall_tau"] == 2 * report["accuracy"] - 1


def test_empty_dag_has_no_comparable_pairs():
    report = causal_order_scores([0, 1], [[0, 0], [0, 0]])
    assert report == {"accuracy": 1.0, "kendall_tau": 1.0, "n_comparable_pairs": 0}


def test_non_permutation_order_raises():
    with pytest.raises(ValueError):
        causal_order_accuracy([0, 0, 1], CHAIN)
