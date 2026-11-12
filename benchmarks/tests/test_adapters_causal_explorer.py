"""Causal Explorer's set learners, through MATLAB: the data and settings the adapter hands the
toolbox, how it reads the answers, where the lane puts the arm, and, where MATLAB and the toolbox
are there, the sets of a collider.

The checks that need MATLAB skip without it, as the R checks skip without R; nothing here reaches
the toolbox otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import blanket_cpu  # noqa: E402

from andrey_bench import blanket_lane as lane  # noqa: E402
from andrey_bench.adapters import causal_explorer_blanket as ce  # noqa: E402
from andrey_bench.adapters.causal_explorer_blanket import CausalExplorerBlanket  # noqa: E402

_SETTINGS = blanket_cpu.Settings("gsq", True, 0.05, None)


def test_categorical_columns_are_coded_from_zero_over_their_observed_levels():
    X = np.array([[3, 0.5], [7, 0.5], [3, 2.0], [9, 0.5]])
    codes, counts = ce.encode(X)
    assert codes.tolist() == [[0, 0], [1, 0], [0, 1], [2, 0]] and counts == [3, 2]


@pytest.mark.parametrize(
    ("function", "test", "depth", "want"),
    [
        ("HITON_PC", "gsq", None, {"statistic": "g2", "max_k": 5}),
        ("HITON_MB", "gsq", 3, {"statistic": "g2", "max_k": 3}),
        ("MMMB", "gsq", None, {"statistic": "g2", "max_k": 5, "epc": 5}),
    ],
)
def test_each_setting_reaches_the_toolbox(function, test, depth, want):
    """alpha is the threshold, max_depth the largest conditioning set (d - 1 without a cap), the
    targets 1-based, and the test the toolbox's statistic."""
    adapter = CausalExplorerBlanket(function, indep_test=test, alpha=0.01, max_depth=depth)
    config = adapter.config(6, [0, 4], [2, 3, 2, 2, 2, 2])
    assert config.items() >= {**want, "threshold": 0.01, "targets": [1, 5]}.items()
    assert config["function"] == function and json.loads(json.dumps(config)) == config
    params = adapter.params()
    assert params["max_depth"] == depth and params["statistic"] == want["statistic"]
    assert params["kind"] == ce.FUNCTIONS[function]


def test_a_setting_without_a_counterpart_is_refused():
    with pytest.raises(ValueError, match="takes the tests"):
        CausalExplorerBlanket("HITON_PC", indep_test="chisq")
    for function in ce.FUNCTIONS:
        with pytest.raises(ValueError, match="takes the tests"):
            CausalExplorerBlanket(function, indep_test="fisherz")
    with pytest.raises(ValueError, match="function must be one of"):
        CausalExplorerBlanket("IAMB", indep_test="gsq")


def test_the_answers_are_read_0_based_and_an_error_raises():
    text = "matlab\t9.10\n1\tok\t0.1\t3 2\n2\tok\t0.0\t\n3\tok\t0.2\t1\n"
    assert ce.parse_output(text, 3) == {0: [1, 2], 1: [], 2: [0]}
    with pytest.raises(RuntimeError, match="failed at target 1: bad"):
        ce.parse_output("2\terror\t0.0\tbad\n", 3)
    with pytest.raises(RuntimeError, match="gave target 0"):
        ce.parse_output("1\tok\t0.0\t1\n", 3)


def test_the_and_rule_keeps_the_pairs_in_each_others_set():
    sets = {0: [1, 2], 1: [0], 2: [1], 3: []}
    assert ce.and_rule(sets, [0, 1, 2]) == {0: [1], 1: [0], 2: []}


def test_hiton_pc_is_symmetric_unless_asked_for_the_toolboxs_sets():
    assert CausalExplorerBlanket("HITON_PC", indep_test="gsq").symmetry
    raw = CausalExplorerBlanket("HITON_PC", indep_test="gsq", symmetry=False)
    assert not raw.symmetry and raw.name == "causal-explorer.hiton-pc.raw"
    assert not CausalExplorerBlanket("HITON_MB", indep_test="gsq").symmetry


def test_the_arm_is_absent_without_matlab_or_the_toolbox(monkeypatch, tmp_path):
    monkeypatch.setenv(ce.TOOLBOX_ENV, str(tmp_path))
    assert ce.toolbox_dir() is None and not ce.available()
    monkeypatch.delenv(ce.TOOLBOX_ENV)
    assert not ce.available()
    with pytest.raises(RuntimeError, match="start_causal_explorer.sh"):
        CausalExplorerBlanket("HITON_MB", indep_test="gsq").setup()
    names = {
        a.name for a in blanket_cpu.field(blanket_cpu.Cell.parse("net:asia:200"), _SETTINGS, None)
    }
    assert not {n for n in names if n.startswith("causal-explorer.")}


def test_the_field_pairs_the_hiton_learners_with_the_authors_functions():
    """In target mode, on categorical data, each learner with a Causal Explorer counterpart runs
    beside it; the whole-graph runs and continuous data have none."""
    cell = blanket_cpu.Cell.parse("net:asia:200")
    names = [a.name for a in blanket_cpu.field(cell, _SETTINGS, None, causal_explorer=True)]
    for learner in ("hiton_pc", "hiton_mb"):
        if f"andrey.{learner}" in names:
            assert blanket_cpu.authors_name(f"andrey.{learner}") in names
    assert blanket_cpu.authors_name("andrey.iamb") is None
    assert blanket_cpu.authors_name("andrey.hiton_mb") == "causal-explorer.hiton-mb"
    graph = blanket_cpu.field(cell, _SETTINGS, "skeleton", causal_explorer=True)
    gauss = blanket_cpu.field(blanket_cpu.Cell.parse("gauss:8:200"), _SETTINGS, None, True)
    assert not [a for a in graph + gauss if a.package == "causal-explorer"]


needs_matlab = pytest.mark.skipif(not ce.available(), reason="no matlab or Causal Explorer")


def _collider(n: int = 3000, seed: int = 0) -> np.ndarray:
    """Binary columns ``[a, b, c, e]``: ``c`` a noisy OR of ``a`` and ``b``, ``e`` alone."""
    rng = np.random.default_rng(seed)
    a, b, e = rng.integers(0, 2, (3, n))
    c = (a | b) ^ (rng.random(n) < 0.1)
    return np.column_stack([a, b, c, e])


@needs_matlab
@pytest.mark.parametrize(
    ("function", "want"),
    [
        ("HITON_PC", {0: [2], 1: [2], 2: [0, 1], 3: []}),
        ("HITON_MB", {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: []}),
        ("MMMB", {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: []}),
    ],
)
def test_each_function_finds_the_colliders_sets(function, want):
    adapter = CausalExplorerBlanket(function, indep_test="gsq", max_depth=3)
    adapter.setup()
    native = adapter.fit(_collider(), {**adapter.params(), "targets": [0, 1, 2, 3]})
    answer, counts, total = lane.unpack_answer(native)
    assert lane.sets_of(answer, [0, 1, 2, 3]) == want
    assert (counts == lane.NO_COUNT).all() and total is None
