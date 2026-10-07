"""The Andrey FCI adapter's two collider rules: one solution name each, and the rule reaches the fit.

The fit-to-valid-PAG check for both FCI adapters is in :mod:`test_adapters_contract`.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from andrey import GraphStructure
from andrey_bench.adapters.andrey_fci import AndreyFCI


def test_andrey_fci_collider_rules_have_distinct_identities() -> None:
    default, majority = AndreyFCI(), AndreyFCI("majority")
    assert default.name == "andrey.fci.numpy"
    assert majority.name == "andrey.fci.majority.numpy"
    assert default.params()["collider_rule"] == "sepsets"
    assert majority.params()["collider_rule"] == "majority"
    with pytest.raises(ValueError, match="collider_rule"):
        AndreyFCI("invalid")


@pytest.mark.parametrize(
    "rule,params,expected",
    [
        ("sepsets", {}, "sepsets"),
        ("majority", {}, "majority"),
        ("majority", {"collider_rule": "sepsets"}, "sepsets"),
    ],
)
def test_andrey_fci_forwards_collider_rule(rule, params, expected, monkeypatch):
    import andrey

    calls = []
    marks = np.array([[0, 3], [3, 0]], dtype=np.int8)

    def fit(x, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(structure=GraphStructure.from_numpy(marks, kind="pag"))

    monkeypatch.setattr(andrey, "fci", fit)
    result = AndreyFCI(rule).fit(np.zeros((10, 2)), params)
    np.testing.assert_array_equal(result, marks)
    assert calls == [{"alpha": 0.05, "indep_test": "fisherz", "collider_rule": expected}]
