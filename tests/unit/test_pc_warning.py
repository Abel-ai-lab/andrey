"""PC warns about expensive passes without changing the search or warning for other methods."""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path

import numpy as np
import pytest

import andrey
from andrey.core import ci

pc_module = importlib.import_module("andrey.constraint.pc")


@pytest.fixture
def oracle(monkeypatch):
    class ConstantTest:
        pvalue = 0.0
        queries = []

        def __init__(self, data):
            pass

        def __call__(self, x, y, condition_set=()):
            self.queries.append((x, y, tuple(condition_set)))
            return self.pvalue

    monkeypatch.setitem(ci._CI_REGISTRY, "warning-test", ConstantTest)
    return ConstantTest


def test_warning_preserves_queries_and_output_and_occurs_once_per_process(monkeypatch, oracle):
    data = np.zeros((20, 6))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        baseline = andrey.pc(data, indep_test="warning-test")
    assert not caught
    baseline_queries = list(oracle.queries)
    monkeypatch.setattr(pc_module, "_EXPENSIVE_PASS_TESTS", 10)

    for fit in range(2):
        oracle.queries.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = andrey.pc(data, indep_test="warning-test")
        sizes = [str(w.message).split()[3] for w in caught]
        if fit == 0:  # Each expensive conditioning size warns with its own message.
            assert sizes[0] == "2" and len(sizes) == len(set(sizes)) >= 2, sizes
            assert all(w.category is andrey.PerformanceWarning for w in caught)
            assert all(Path(w.filename) == Path(__file__) for w in caught)
        else:  # The same messages do not warn again in this process.
            assert not caught
        assert oracle.queries == baseline_queries
        np.testing.assert_array_equal(result.structure.to_numpy(), baseline.structure.to_numpy())


@pytest.mark.parametrize(("threshold", "expected"), [(90, 1), (91, 0)])
def test_threshold_counts_shared_subsets_once(monkeypatch, oracle, threshold, expected):
    # A complete six-node graph has 15 pairs, each with six distinct size-two subsets.
    monkeypatch.setattr(pc_module, "_EXPENSIVE_PASS_TESTS", threshold)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        andrey.pc(np.zeros((20, 6)), indep_test="warning-test")
    assert len(caught) == expected
    assert sum(len(cond) == 2 for _, _, cond in oracle.queries) == 90


def test_many_variables_pruned_early_do_not_warn(monkeypatch, oracle):
    monkeypatch.setattr(pc_module, "_EXPENSIVE_PASS_TESTS", 1)
    oracle.pvalue = 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = andrey.pc(np.zeros((20, 200)), indep_test="warning-test")
    assert not caught
    assert not np.any(result.structure.to_numpy())


@pytest.mark.parametrize("method", ["fci", "cdnod"])
def test_warning_is_specific_to_pc(monkeypatch, oracle, method):
    monkeypatch.setattr(pc_module, "_EXPENSIVE_PASS_TESTS", 1)
    data = np.zeros((20, 6))
    kwargs = {"c_indx": np.arange(20)} if method == "cdnod" else {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(andrey, method)(data, indep_test="warning-test", **kwargs)
    assert not [w for w in caught if w.category is andrey.PerformanceWarning]
    assert any(len(cond) >= 2 for _, _, cond in oracle.queries)
