"""The Andrey GES adapter's parent rule: recorded in ``params`` and passed to the search."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from andrey.search.operators import max_parents_in_class
from andrey_bench.adapters.andrey_ges import AndreyGES, andrey_ges_adapters


def _fan_in(causes: int, n: int = 2000) -> np.ndarray:
    """``causes`` independent standard-normal variables and a last one, their noisy sum."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, causes))
    return np.column_stack([x, x.sum(axis=1) + 0.5 * rng.standard_normal(n)])


def _adapter(parent_rule: str, max_parents: float | None = 1) -> AndreyGES:
    return AndreyGES(
        name="andrey.ges.numpy.serial",
        backend="numpy",
        mode="serial",
        max_parents=max_parents,
        parent_rule=parent_rule,
    )


def test_params_record_the_rule_only_where_a_limit_applies() -> None:
    """A row names the rule next to the limit it governs; no limit leaves nothing to name."""
    assert andrey_ges_adapters()[0].params()["parent_rule"] == "cap"
    assert andrey_ges_adapters(parent_rule="gate")[1].params()["parent_rule"] == "gate"
    assert "parent_rule" not in _adapter("cap", max_parents=None).params()


@pytest.mark.parametrize(("parent_rule", "exceeds"), [("cap", False), ("gate", True)])
def test_the_adapter_applies_the_rule_it_records(parent_rule: str, exceeds: bool) -> None:
    """Two causes of one variable under a limit of 1: the cap guarantees that no DAG of the
    returned class gives a node more than one parent; the gate admits the collider, so the
    variable reaches two."""
    adapter = _adapter(parent_rule, max_parents=1)
    marks = adapter.fit(_fan_in(2), adapter.params())
    assert (max_parents_in_class(marks) > 1).any() == exceeds


def test_the_ges_lane_gives_both_packages_one_limit_and_one_rule() -> None:
    """`ges_cpu.py` limits causal-learn's parents, so each Andrey arm declares the same limit as a
    gate, causal-learn's rule."""
    path = Path(__file__).resolve().parents[1] / "ges_cpu.py"
    spec = importlib.util.spec_from_file_location("ges_cpu_driver", path)
    lane = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lane
    spec.loader.exec_module(lane)
    reference = next(s for s in lane.SOLUTIONS if s.name == "causal-learn.ges").params()
    arms = [s for s in lane.SOLUTIONS if s.name.startswith("andrey.ges")]
    assert arms
    for arm in arms:
        params = arm.params()
        assert (params["max_parents"], params["parent_rule"]) == (
            reference["max_parents"],
            reference["parent_rule"],
        )
