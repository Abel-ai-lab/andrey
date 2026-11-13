"""The adapter of the authors' LocPC-CDE code (pyCIPHOD), and how it reads the code's graph.

The parent-side parts (the reading of a final graph, the records, the pins of the start script)
need no reference and run everywhere. The tests that run the authors' code skip unless
``start_ciphod.sh`` has fetched it and the variables it prints are set, as the other reference
lanes' do.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from andrey_bench import runner
from andrey_bench.adapters import ciphod_locpc as ciphod
from andrey_bench.adjustment_lane import (
    OUTCOME_PARAM,
    TREATMENT_PARAM,
    PairAdapter,
    answer_records,
    oracle_records,
    pack_effects,
    unpack_effects,
)
from andrey_bench.contracts import SetupAdapter, SolutionAdapter

BENCH = Path(__file__).resolve().parents[1]
#: Fixes the string hash seed in the child of ``run_task``, as the start script's exports do.
HASH_SEED = {"PYTHONHASHSEED": "0"}

#: Planted linear Gaussian case, columns ``(a, x, y, b, c)``: ``a -> x -> y <- b``, ``c`` apart. The
#: collider at ``y`` directs both edges into it, so the direct effect of ``x`` on ``y`` is
#: identified with the set ``{b}``; ``a`` is not adjacent to ``y``.
A, X, Y, B, C = range(5)


def planted(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a, b, c = rng.standard_normal((3, n))
    x = 0.9 * a + rng.standard_normal(n)
    y = 0.9 * x + 0.9 * b + rng.standard_normal(n)
    return np.column_stack([a, x, y, b, c])


def planted_binary(n: int, seed: int) -> np.ndarray:
    """The same graph with logistic arcs, columns ``(a, x, y, b)``."""
    rng = np.random.default_rng(seed)

    def draw(logit: np.ndarray) -> np.ndarray:
        return (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.float64)

    a, b = draw(np.zeros(n)), draw(np.zeros(n))
    x = draw(3.0 * a - 1.5)
    y = draw(3.0 * x + 3.0 * b - 3.0)
    return np.column_stack([a, x, y, b])


def sets_of(record: dict) -> set[tuple[int, ...]]:
    return {tuple(s) for s in record["adjustment_sets"]}


# ----- what the adapter is, and what it needs -------------------------------------------------


def test_the_adapter_fills_the_solution_and_pair_contracts():
    adapter = ciphod.CiphodLocPCCDE()
    assert isinstance(adapter, SolutionAdapter) and isinstance(adapter, SetupAdapter)
    assert isinstance(adapter, PairAdapter)
    assert (adapter.estimand, adapter.set_kind) == ("direct", "parents_of_outcome")
    assert (adapter.identified_from, adapter.unordered) == ("cpdag", False)
    assert adapter.output_type == "effects"
    assert adapter.indep_tests == ("fisherz", "gsq")
    assert adapter.python_env == "ANDREY_BENCH_CIPHOD_PYTHON"


def test_the_name_the_algorithm_and_the_version_are_the_reference():
    adapter = ciphod.CiphodLocPCCDE()
    assert (adapter.name, adapter.package, adapter.algorithm) == (
        "locpc_cde.authors",
        "pyciphod",
        "locpc_cde",
    )
    assert adapter.package_version == ciphod.COMMIT[:12] == "37a62c9f2052"


def test_the_settings_are_the_codes_and_the_cell_chooses_the_test():
    adapter = ciphod.CiphodLocPCCDE()
    assert adapter.params() == {"alpha": 0.05, "indep_test": "fisherz"}
    data = np.zeros((10, 3))
    pair = {TREATMENT_PARAM: 0, OUTCOME_PARAM: 1}
    with pytest.raises(ValueError, match="indep_test"):
        adapter.fit(data, {**adapter.params(), **pair, "indep_test": "chisq"})
    with pytest.raises(ValueError, match="max_depth"):
        adapter.fit(data, {**adapter.params(), **pair, "max_depth": 3})


def test_an_unset_environment_names_the_script_that_builds_it(monkeypatch, tmp_path):
    adapter = ciphod.CiphodLocPCCDE()
    pair = {TREATMENT_PARAM: 0, OUTCOME_PARAM: 1}
    for value in (None, str(tmp_path)):
        if value is None:
            monkeypatch.delenv(ciphod.CODE_ENV, raising=False)
        else:
            monkeypatch.setenv(ciphod.CODE_ENV, value)
        with pytest.raises(FileNotFoundError, match="start_ciphod.sh"):
            adapter.setup()
        with pytest.raises(FileNotFoundError, match="start_ciphod.sh"):
            adapter.fit(np.zeros((10, 3)), {**adapter.params(), **pair})


@pytest.mark.parametrize(
    ("seed", "refused"), [("0", False), ("5", False), ("random", True), (None, True)]
)
def test_a_random_string_hash_seed_is_refused(seed, refused):
    """The code iterates sets of names, so a fit needs a fixed hash seed."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    env["PYTHONPATH"] = str(BENCH)
    if seed is not None:
        env["PYTHONHASHSEED"] = seed
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            "from andrey_bench.adapters import ciphod_locpc as c; c._check_hash_seed()",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert (run.returncode != 0) == refused, run.stderr
    if refused:
        assert "PYTHONHASHSEED" in run.stderr


def test_the_start_script_fetches_the_commit_the_adapter_reads_and_checks_the_archive():
    script = (BENCH / "start_ciphod.sh").read_text()
    assert re.search(rf'^COMMIT="{ciphod.COMMIT}"$', script, re.MULTILINE)
    assert re.search(r'^SHA256="[0-9a-f]{64}"$', script, re.MULTILINE)
    assert "codeload.github.com/CIPHOD/pyCIPHOD/tar.gz/$COMMIT" in script
    assert "export PYTHONHASHSEED=0" in script
    lock = [
        line
        for line in (BENCH / "requirements-ciphod.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.]*", line) for line in lock)
    pinned = {line.split("==")[0] for line in lock}
    assert {
        "networkx",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    } <= pinned


def test_the_columns_are_named_so_that_sorted_names_are_in_column_order():
    for d in (3, 10, 100, 1000, 1001):
        names = ciphod.column_names(d)
        assert len(set(names)) == d and names == sorted(names)
    assert ciphod.column_names(10)[7] == "V007"


# ----- the reading of the code's graph ----------------------------------------------------------


def dag_of(d: int, edges: list[tuple[int, int]]) -> np.ndarray:
    dag = np.zeros((d, d), dtype=int)
    for a, b in edges:
        dag[a, b] = 1
    return dag


def test_an_oriented_graph_gives_the_outcomes_other_parents_as_the_set():
    record, reason = ciphod.read_graph(X, Y, {(X, Y), (B, Y), (A, X)}, {X, B})
    assert reason == "oriented"
    assert (record["nonzero"], record["identifiable"]) == (True, True)
    assert record["adjustment_sets"] == [[B]] and record["set_kind"] == "parents_of_outcome"
    truth = oracle_records(
        dag_of(5, [(A, X), (X, Y), (B, Y)]),
        [],
        (X, Y),
        estimand="direct",
        set_kind="parents_of_outcome",
        identified_from="cpdag",
    )[0]
    assert (record["nonzero"], record["identifiable"], sets_of(record)) == (
        truth["nonzero"],
        truth["identifiable"],
        sets_of(truth),
    )


def test_an_undirected_edge_at_the_outcome_is_not_identifiable():
    record, reason = ciphod.read_graph(X, Y, set(), {X})
    assert reason == "not orientable"
    assert (record["nonzero"], record["identifiable"]) == (None, False)
    assert record["adjustment_sets"] == [] and record["set_kind"] is None
    truth = oracle_records(
        dag_of(3, [(A, X), (X, Y)]),
        [],
        (X, Y),
        estimand="direct",
        set_kind="parents_of_outcome",
        identified_from="cpdag",
    )[0]
    assert truth["identifiable"] is False and truth["adjustment_sets"] == []


def test_a_directed_edge_from_the_treatment_is_nonzero_even_when_another_edge_is_not():
    record, reason = ciphod.read_graph(X, Y, {(X, Y)}, {X, B})
    assert reason == "not orientable"
    assert (record["nonzero"], record["identifiable"]) == (True, False)
    assert record["adjustment_sets"] == []


@pytest.mark.parametrize(
    ("directed", "adjacent", "reason"),
    [
        ({(B, Y)}, {B}, "not adjacent"),
        ({(Y, X)}, {X}, "treatment is a child"),
        ({(Y, X), (B, Y)}, {X, B}, "treatment is a child"),
    ],
)
def test_a_zero_stop_says_the_effect_is_zero_and_decides_identifiability_by_the_edges(
    directed, adjacent, reason
):
    record, got = ciphod.read_graph(X, Y, directed, adjacent)
    assert got == reason and record["nonzero"] is False
    assert (record["identifiable"], record["set_kind"]) == (True, "parents_of_outcome")
    assert record["adjustment_sets"] == [sorted(a for a, b in directed if b == Y and a != X)]


def test_a_zero_stop_with_an_edge_left_undirected_decides_nothing_about_identifiability():
    record, reason = ciphod.read_graph(X, Y, {(B, Y)}, {B, C})
    assert reason == "not adjacent" and record["nonzero"] is False
    assert record["identifiable"] is None
    assert record["adjustment_sets"] == [] and record["set_kind"] is None


def test_a_directed_edge_counts_as_oriented_whatever_else_the_graph_stores_for_the_pair():
    """The code's final check asks only for a directed edge, so the reading does too: the
    adjacency set of the outcome holds the pair whatever store gave it."""
    record, reason = ciphod.read_graph(X, Y, {(X, Y), (B, Y)}, {X, B})
    assert reason == "oriented" and record["identifiable"] is True


def test_the_records_carry_the_reason_and_the_codes_own_flag():
    adapter = ciphod.CiphodLocPCCDE()
    record, reason = ciphod.read_graph(X, Y, {(X, Y), (B, Y)}, {X, B})
    native = pack_effects([record], 5, 9)
    native[-1, 1] = 1
    native[-1, 2] = ciphod.REASONS.index(reason)
    [read], tests = answer_records(adapter, native, {})
    assert tests == 9
    assert read == {**record, "reason": "oriented", "authors_identifiability": True}
    [plain], _ = unpack_effects(native)
    assert plain == record  # the packed record is the lane's seven fields
    assert adapter.to_structure(native).n_nodes == 5


# ----- the authors' code, run -------------------------------------------------------------------

CODE = bool(os.environ.get(ciphod.CODE_ENV))
READY = CODE and bool(os.environ.get(ciphod.CiphodLocPCCDE.python_env))
needs_code = pytest.mark.skipif(not READY, reason="the authors' code is absent (start_ciphod.sh)")


def run_here(data: np.ndarray, pair: tuple[int, int], **settings):
    """One fit in the reference environment, through the runner, and the records read as the
    driver reads them."""
    adapter = ciphod.CiphodLocPCCDE()
    params = {
        **adapter.params(),
        **settings,
        TREATMENT_PARAM: pair[0],
        OUTCOME_PARAM: pair[1],
    }
    [run] = runner.run_task(
        adapter, data, params, cap_wall_s=300, repeats=1, warmup=0, extra_env=HASH_SEED
    )
    assert run["status"] == "ok", run
    return answer_records(adapter, run["adj"], params)


@needs_code
def test_the_authors_code_finds_the_oriented_effect_and_its_set():
    """At 8000 rows, ``a -> x -> y <- b``: x on y is identified with the set ``[b]``; the code's own
    flag agrees."""
    (record,), n_tests = run_here(planted(8000, 0), (X, Y))
    assert record["reason"] == "oriented" and record["authors_identifiability"] is True
    assert (record["nonzero"], record["identifiable"]) == (True, True)
    assert record["adjustment_sets"] == [[B]] and record["set_kind"] == "parents_of_outcome"
    assert n_tests and n_tests > 0
    assert set(record) == {
        "treatment",
        "outcome",
        "estimand",
        "nonzero",
        "identifiable",
        "adjustment_sets",
        "set_kind",
        "reason",
        "authors_identifiability",
    }


@needs_code
def test_the_authors_code_stops_with_a_zero_effect_when_the_treatment_is_not_adjacent():
    (record,), _ = run_here(planted(8000, 0), (A, Y))
    assert record["reason"] == "not adjacent" and record["nonzero"] is False
    assert record["authors_identifiability"] is True  # the code's flag: a zero effect is decided


@needs_code
def test_the_authors_code_runs_the_g_square_test_on_binary_data():
    (record,), n_tests = run_here(planted_binary(8000, 0), (X, Y), indep_test="gsq")
    assert record["reason"] == "oriented" and record["adjustment_sets"] == [[B]]
    assert (record["nonzero"], record["identifiable"]) == (True, True) and n_tests


@needs_code
def test_the_answer_does_not_depend_on_where_the_pair_sits_in_the_columns():
    """Zero-padded names keep the code's sorted-name order equal to the column order, so permuting
    the columns and the pair with them leaves the record's meaning unchanged: same reason, and the
    set maps to the same variables."""
    data = planted(8000, 1)
    order = np.array([C, B, Y, A, X])  # new column j holds old column order[j]
    position = {old: new for new, old in enumerate(order)}
    (here,), _ = run_here(data, (X, Y))
    (there,), _ = run_here(data[:, order], (position[X], position[Y]))
    assert here["reason"] == there["reason"] == "oriented"
    assert [order[v] for v in there["adjustment_sets"][0]] == here["adjustment_sets"][0]
