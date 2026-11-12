"""The adapters of the authors' SNAP and LOAD code, and the Fisher-z test they are called with.

The parent-side parts (the test callable, the marks of SNAP's graph, the records read from a graph
or from LOAD's output) need no reference and run everywhere. The tests that run the authors' code
skip unless ``start_schubert.sh`` has fetched it and the variables it prints are set, as the other
reference lanes' do.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest

from andrey_bench import runner
from andrey_bench.adapters import schubert_load as load
from andrey_bench.adapters import schubert_snap as snap
from andrey_bench.adapters.schubert_fisherz import FisherZ
from andrey_bench.adjustment_lane import (
    OPTIMAL_OR_LOCAL,
    VARIABLES_PARAM,
    PairAdapter,
    answer_records,
    oracle_records,
    pack_effects,
    unpack_effects,
)
from andrey_bench.contracts import SetupAdapter, SolutionAdapter
from andrey_bench.local_lane import pack_answer

BENCH = Path(__file__).resolve().parents[1]
TAIL, ARROW = 1, 2
ADAPTERS = (snap.SchubertSNAP, load.SchubertLOAD)

#: Planted linear Gaussian cases: ``(columns, edges, pair)``. The first three are the DAGs of LOAD's
#: README (its stated answers: effect identified with the set {3, 4}; not identified, with the sets
#: {} and {3}; neither direction decided); the fourth gives ``x`` two non-adjacent parents, so its
#: v-structure lets the class identify the effect of ``x`` on ``y`` with the optimal set {c}.
CASES = {
    "readme-identified": (5, [(0, 1), (1, 2), (4, 0), (4, 1), (3, 0), (3, 2)], (0, 2)),
    "readme-local": (5, [(0, 1), (1, 2), (4, 1), (3, 0), (3, 2)], (0, 2)),
    "readme-undecided": (4, [(0, 1), (1, 2), (3, 0), (3, 1), (3, 2)], (0, 2)),
    # columns (d, c, x, m, y): d -> x <- c, c -> y, x -> m -> y.
    "collider": (5, [(0, 2), (1, 2), (1, 4), (2, 3), (3, 4)], (2, 4)),
}
#: Further graphs the parent-side reading is checked on, among them ones the class leaves
#: unidentified.
MORE = {
    "chain": (3, [(0, 1), (1, 2)], (0, 2)),
    "unshielded-collider": (3, [(0, 1), (2, 1)], (0, 2)),
    "one-edge": (2, [(0, 1)], (0, 1)),
    "undirected-then-directed": (4, [(0, 1), (1, 3), (2, 3)], (0, 3)),
    "outcome-is-a-parent": (4, [(3, 0), (1, 0), (2, 0), (1, 3)], (0, 3)),
}


def dag_of(d: int, edges: list[tuple[int, int]]) -> np.ndarray:
    dag = np.zeros((d, d), dtype=int)
    for a, b in edges:
        dag[a, b] = 1
    return dag


def planted(d: int, edges: list[tuple[int, int]], n: int, seed: int) -> np.ndarray:
    """``n`` rows of the linear Gaussian model with every weight 0.9 and standard noise."""
    parents = {v: [a for a, b in edges if b == v] for v in range(d)}
    order: list[int] = []
    while len(order) < d:
        order.append(next(v for v in range(d) if v not in order and set(parents[v]) <= set(order)))
    rng = np.random.default_rng(seed)
    X = np.zeros((n, d))
    for v in order:
        X[:, v] = rng.standard_normal(n) + sum(0.9 * X[:, p] for p in parents[v])
    return X


def truth_of(dag: np.ndarray, pair: tuple[int, int]) -> list[dict]:
    """The records of both directions the method's definition gives on the true graph."""
    return oracle_records(
        dag,
        [],
        pair,
        estimand="total",
        set_kind=OPTIMAL_OR_LOCAL,
        identified_from="cpdag",
        unordered=True,
    )


def sets_of(record: dict) -> set[tuple[int, ...]]:
    return {tuple(s) for s in record["adjustment_sets"]}


def same_answer(record: dict, truth: dict) -> bool:
    """Whether a record is the oracle's: the class's own answer, with ``nonzero`` allowed to say
    "cannot tell" where the class does not identify the effect."""
    nonzero_ok = (
        record["nonzero"] == truth["nonzero"]
        if truth["identifiable"]
        else record["nonzero"] in (None, truth["nonzero"])
    )
    return (
        record["identifiable"] == truth["identifiable"]
        and sets_of(record) == sets_of(truth)
        and record["set_kind"] == truth["set_kind"]
        and nonzero_ok
    )


# ----- what the adapters are, and what they need ----------------------------------------------


@pytest.mark.parametrize("adapter_class", ADAPTERS)
def test_the_adapters_fill_the_solution_and_pair_contracts(adapter_class):
    adapter = adapter_class()
    assert isinstance(adapter, SolutionAdapter) and isinstance(adapter, SetupAdapter)
    assert isinstance(adapter, PairAdapter)
    assert (adapter.estimand, adapter.set_kind) == ("total", OPTIMAL_OR_LOCAL)
    assert (adapter.identified_from, adapter.unordered) == ("cpdag", True)
    assert adapter.output_type == "effects"
    assert adapter.python_env == "ANDREY_BENCH_SCHUBERT_PYTHON"


def test_the_names_the_algorithms_and_the_versions_are_the_references():
    snap_, load_ = snap.SchubertSNAP(), load.SchubertLOAD()
    assert (snap_.name, snap_.package, snap_.algorithm) == ("snap.authors", "snap", "snap")
    assert (load_.name, load_.package, load_.algorithm) == (
        "load.authors",
        "load",
        "local_optimal_adjustment",
    )
    assert snap_.package_version == snap.COMMIT[:12] == "432ab5c96904"
    assert load_.package_version == load.COMMIT[:12] == "7febfc68d477"


def test_the_settings_are_the_authors_and_the_pair_is_passed_by_position():
    assert snap.SchubertSNAP().params() == {
        "alpha": 0.05,
        "indep_test": "fisherz",
        "max_order": None,
    }
    assert load.SchubertLOAD().params() == {"alpha": 0.01, "indep_test": "fisherz"}
    assert VARIABLES_PARAM == "variables"


@pytest.mark.parametrize("adapter_class", ADAPTERS)
def test_a_test_other_than_fisher_z_or_a_depth_cap_is_refused(adapter_class):
    adapter, X = adapter_class(), np.zeros((10, 3))
    pair = {VARIABLES_PARAM: [0, 1]}
    with pytest.raises(ValueError, match="fisherz"):
        adapter.fit(X, {**adapter.params(), **pair, "indep_test": "gsq"})
    with pytest.raises(ValueError, match="max_depth"):
        adapter.fit(X, {**adapter.params(), **pair, "max_depth": 3})


@pytest.mark.parametrize("adapter_class", ADAPTERS)
def test_an_unset_environment_names_the_script_that_builds_it(adapter_class, monkeypatch):
    adapter = adapter_class()
    monkeypatch.delenv(snap.CODE_ENV, raising=False)
    monkeypatch.delenv(load.CODE_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="start_schubert.sh"):
        adapter.setup()
    with pytest.raises(FileNotFoundError, match="start_schubert.sh"):
        adapter.fit(np.zeros((10, 3)), {**adapter.params(), VARIABLES_PARAM: [0, 1]})


def test_a_directory_without_the_code_names_the_script_too(tmp_path, monkeypatch):
    monkeypatch.setenv(snap.CODE_ENV, str(tmp_path))
    monkeypatch.setenv(load.CODE_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError, match="start_schubert.sh"):
        snap.SchubertSNAP().setup()
    with pytest.raises(FileNotFoundError, match="start_schubert.sh"):
        load.SchubertLOAD().setup()


def test_the_start_script_fetches_the_commits_the_adapters_read_and_checks_each_archive():
    script = (BENCH / "start_schubert.sh").read_text()
    for name, module in (("SNAP", snap), ("LOAD", load)):
        assert re.search(rf'^{name}_COMMIT="{module.COMMIT}"$', script, re.MULTILINE), name
        assert re.search(rf'^{name}_SHA256="[0-9a-f]{{64}}"$', script, re.MULTILINE), name
    assert "codeload.github.com/Matyasch/$name/tar.gz/$commit" in script
    lock = (BENCH / "requirements-schubert.txt").read_text()
    assert [line.split("==")[0] for line in lock.split()] == ["numpy"]


# ----- the Fisher-z test ----------------------------------------------------------------------


def test_fisher_z_is_the_two_sided_normal_tail_of_the_partial_correlation():
    stats = pytest.importorskip("scipy.stats")
    X = planted(4, [(0, 1), (1, 2), (3, 2)], 600, 0)
    test = FisherZ(X)
    for x, y, given in [(0, 2, ()), (0, 2, (1,)), (0, 3, (1, 2)), (1, 3, (0,)), (2, 3, ())]:
        # The partial correlation from the residuals of the regressions on the given columns.
        design = np.column_stack([np.ones(len(X)), X[:, list(given)]])
        resid = [X[:, v] - design @ np.linalg.lstsq(design, X[:, v], rcond=None)[0] for v in (x, y)]
        r = np.corrcoef(resid)[0, 1]
        z = np.sqrt(len(X) - len(given) - 3) * abs(np.arctanh(r))
        assert test(x, y, given) == pytest.approx(2 * stats.norm.sf(z), rel=1e-9, abs=1e-300)


def test_fisher_z_agrees_with_andreys_own_test():
    ci = pytest.importorskip("andrey.core.ci")
    X = planted(5, [(0, 1), (1, 2), (3, 2), (2, 4)], 800, 2)
    ours, theirs = FisherZ(X), ci.FisherZ(X)
    for x, y, given in [(0, 2, ()), (0, 4, (2,)), (1, 3, (0, 2)), (0, 3, (1, 2, 4)), (4, 1, (2,))]:
        assert ours(x, y, given) == pytest.approx(theirs(x, y, given), abs=1e-12)


def test_the_count_is_the_distinct_queries_whatever_their_order_or_container():
    test = FisherZ(planted(4, [(0, 1), (1, 2), (3, 2)], 300, 1))
    p = test(0, 2, (1, 3))
    assert (
        test(2, 0, {3, 1}) == p and test(0, 2, frozenset((1, 3))) == p and test(0, 2, [3, 1]) == p
    )
    assert (test.n_tests, test.calls) == (1, 4)
    test(0, 2, (1,))
    test(np.int64(0), np.int64(1), ())
    assert (test.n_tests, test.calls) == (3, 6)


def test_a_sample_too_small_for_the_set_and_a_singular_matrix_are_refused():
    X = planted(4, [(0, 1)], 5, 0)
    with pytest.raises(ValueError, match="too few"):
        FisherZ(X)(0, 1, (2, 3))
    Y = planted(3, [(0, 1)], 200, 0)
    with pytest.raises(ValueError, match="singular"):
        FisherZ(np.column_stack([Y, Y[:, 0]]))(1, 2, (0, 3))


# ----- SNAP: the graph and its records ---------------------------------------------------------


def test_the_codes_adjacency_becomes_andrey_marks():
    amat = np.zeros((4, 4), dtype=int)
    amat[0, 1] = 1  # 0 -> 1
    amat[1, 2] = amat[2, 1] = 1  # 1 - 2
    marks = snap.andrey_marks(amat)
    assert (marks[0, 1], marks[1, 0]) == (TAIL, ARROW)
    assert (marks[1, 2], marks[2, 1]) == (TAIL, TAIL)
    assert marks.dtype == np.int8 and np.count_nonzero(marks) == 4 and not marks[3].any()


def _native(marks: np.ndarray) -> np.ndarray:
    return pack_answer(np.asarray(marks), 7)


@pytest.mark.parametrize("case", [*CASES, *MORE])
def test_snap_reads_the_class_of_the_true_graph_as_the_oracle_does(case):
    """The records read from the CPDAG of the true graph are the oracle's, the PC arm's reading."""
    from andrey.metrics import to_cpdag

    d, edges, pair = {**CASES, **MORE}[case]
    dag = dag_of(d, edges)
    adapter = snap.SchubertSNAP()
    records, n_tests = adapter.records(
        _native(to_cpdag(dag).to_numpy()), {VARIABLES_PARAM: list(pair)}
    )
    assert n_tests == 7
    assert [(r["treatment"], r["outcome"]) for r in records] == [pair, pair[::-1]]
    for record, truth in zip(records, truth_of(dag, pair), strict=True):
        assert same_answer(record, truth), (case, record, truth)


def test_a_graph_that_is_not_a_completed_cpdag_is_read_from_its_completion():
    pair = {VARIABLES_PARAM: [0, 1]}
    directed, undirected = np.zeros((2, 2), dtype=np.int8), np.zeros((2, 2), dtype=np.int8)
    directed[0, 1], directed[1, 0] = TAIL, ARROW  # a lone edge the class leaves undirected
    undirected[0, 1] = undirected[1, 0] = TAIL
    adapter = snap.SchubertSNAP()
    assert adapter.records(_native(directed), pair) == adapter.records(_native(undirected), pair)


def test_a_graph_without_a_consistent_dag_extension_decides_nothing():
    cycle = np.zeros((3, 3), dtype=np.int8)
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        cycle[a, b], cycle[b, a] = TAIL, ARROW
    records, n_tests = snap.SchubertSNAP().records(_native(cycle), {VARIABLES_PARAM: [0, 2]})
    assert n_tests == 7
    for record in records:
        assert record["nonzero"] is None and record["identifiable"] is None
        assert record["adjustment_sets"] == [] and record["set_kind"] is None


def test_a_zero_effect_the_graph_shows_is_said_zero_where_it_is_not_identified():
    """The outcome is a parent of the treatment: no set blocks that edge, and the effect is zero."""
    d, edges, pair = MORE["outcome-is-a-parent"]
    from andrey.metrics import to_cpdag

    records, _ = snap.SchubertSNAP().records(
        _native(to_cpdag(dag_of(d, edges)).to_numpy()), {VARIABLES_PARAM: list(pair)}
    )
    assert (records[0]["identifiable"], records[0]["nonzero"]) == (False, False)


def test_snap_draws_no_graph_to_score():
    adapter = snap.SchubertSNAP()
    structure = adapter.to_structure(_native(np.zeros((4, 4), dtype=np.int8)))
    assert structure.n_nodes == 4 and not structure.to_numpy().any()


# ----- LOAD: the records of what it returns ----------------------------------------------------


READMES = {
    # The three outputs the README prints, for its three DAGs, with the targets [2, 0] it passes.
    "readme-identified": {"adj_sets": {(0, 2): [{3, 4}]}, "identifiable": True},
    "readme-local": {"adj_sets": {(0, 2): [set(), {3}]}, "identifiable": False},
    "readme-undecided": {
        "adj_sets": {
            (2, 0): [set(), {1}, {3}, {1, 3}],
            (0, 2): [set(), {1}, {3}, {1, 3}],
        },
        "identifiable": False,
    },
}


@pytest.mark.parametrize("case", READMES)
def test_loads_readme_outputs_become_the_records_the_oracle_gives_on_its_graphs(case):
    d, edges, pair = CASES[case]
    records = load.effect_records(READMES[case], pair)
    assert [(r["treatment"], r["outcome"]) for r in records] == [pair, pair[::-1]]
    for record, truth in zip(records, truth_of(dag_of(d, edges), pair), strict=True):
        assert same_answer(record, truth), (case, record, truth)


def test_the_records_of_loads_three_kinds_of_answer():
    identified, local, undecided = (load.effect_records(READMES[k], (0, 2)) for k in READMES)
    assert identified[0] == {
        "treatment": 0,
        "outcome": 2,
        "estimand": "total",
        "nonzero": True,
        "identifiable": True,
        "adjustment_sets": [[3, 4]],
        "set_kind": "optimal",
    }
    # No set for a direction: the code's "identified as zero".
    assert (identified[1]["nonzero"], identified[1]["identifiable"]) == (False, True)
    assert identified[1]["adjustment_sets"] == [] and identified[1]["set_kind"] is None
    assert local[0]["adjustment_sets"] == [[], [3]] and local[0]["set_kind"] == "local_parents"
    assert (local[0]["nonzero"], local[0]["identifiable"]) == (None, False)
    assert (local[1]["nonzero"], local[1]["identifiable"]) == (False, True)
    assert [r["nonzero"] for r in undecided] == [None, None]
    assert [r["identifiable"] for r in undecided] == [False, False]


def test_an_empty_optimal_set_is_a_set_and_the_outcome_is_never_in_one():
    answer = {"adj_sets": {(0, 2): [set()]}, "identifiable": True}
    assert load.effect_records(answer, (0, 2))[0]["adjustment_sets"] == [[]]
    answer = {"adj_sets": {(0, 2): [{1, 2}, {1}]}, "identifiable": False}
    assert load.effect_records(answer, (0, 2))[0]["adjustment_sets"] == [[1]]


def test_load_draws_no_graph_to_score():
    adapter = load.SchubertLOAD()
    native = pack_effects(load.effect_records(READMES["readme-local"], (0, 2)), 5, 9)
    assert adapter.to_structure(native).n_nodes == 5
    assert unpack_effects(native)[1] == 9


# ----- the authors' code, run -------------------------------------------------------------------


def _code(adapter_class) -> bool:
    module = snap if adapter_class is snap.SchubertSNAP else load
    return bool(os.environ.get(module.CODE_ENV))


needs_code = pytest.mark.parametrize(
    "adapter_class",
    [
        pytest.param(
            cls,
            marks=pytest.mark.skipif(
                not _code(cls), reason="the authors' code is absent (start_schubert.sh)"
            ),
        )
        for cls in ADAPTERS
    ],
)


@needs_code
@pytest.mark.parametrize("case", CASES)
def test_the_authors_code_gives_the_class_answer_on_planted_cases(adapter_class, case):
    """At 20000 rows the answer is the oracle's on at least three of four draws: the code's tests
    are tests, each wrong at its level on some draw."""
    d, edges, pair = CASES[case]
    truth = truth_of(dag_of(d, edges), pair)
    adapter = adapter_class()
    params = {**adapter.params(), VARIABLES_PARAM: list(pair)}
    right = 0
    for seed in range(4):
        native = adapter.fit(planted(d, edges, 20000, seed), params)
        records, n_tests = answer_records(adapter, native, params)
        assert n_tests is not None and n_tests > 0
        assert [(r["treatment"], r["outcome"]) for r in records] == [pair, pair[::-1]]
        right += all(same_answer(r, t) for r, t in zip(records, truth, strict=True))
    assert right >= 3, (case, right)


@needs_code
def test_the_authors_code_finds_the_confounded_effect_and_the_zero_reverse(adapter_class):
    """``d -> x <- c``, ``c -> y``, ``x -> m -> y``: x on y, identified, with the set [c]; y on x
    is zero. The pair is given in either column order and the sets are the same."""
    d, edges, pair = CASES["collider"]
    adapter = adapter_class()
    X = planted(d, edges, 20000, 0)
    for variables in (list(pair), list(pair[::-1])):
        params = {**adapter.params(), VARIABLES_PARAM: variables}
        records, n_tests = answer_records(adapter, adapter.fit(X, params), params)
        by_pair = {(r["treatment"], r["outcome"]): r for r in records}
        effect, reverse = by_pair[pair], by_pair[pair[::-1]]
        assert (effect["nonzero"], effect["identifiable"]) == (True, True)
        assert effect["adjustment_sets"] == [[1]] and effect["set_kind"] == "optimal"
        assert (reverse["nonzero"], reverse["identifiable"]) == (False, True)
        assert reverse["adjustment_sets"] == [] and n_tests


@needs_code
def test_the_count_is_the_distinct_queries_the_code_asked(adapter_class, monkeypatch):
    """The reported count is the callable's distinct queries, which no more than its calls."""
    created: list[FisherZ] = []
    original = FisherZ.__init__

    def record(self, data):
        original(self, data)
        created.append(self)

    monkeypatch.setattr(FisherZ, "__init__", record)
    d, edges, pair = CASES["collider"]
    adapter = adapter_class()
    params = {**adapter.params(), VARIABLES_PARAM: list(pair)}
    _, n_tests = answer_records(adapter, adapter.fit(planted(d, edges, 3000, 0), params), params)
    [test] = created
    assert n_tests == test.n_tests <= test.calls


def _own_environment_ready(adapter_class) -> bool:
    return _code(adapter_class) and bool(os.environ.get(adapter_class.python_env))


@pytest.mark.parametrize(
    "adapter_class",
    [
        pytest.param(
            cls,
            marks=pytest.mark.skipif(
                not _own_environment_ready(cls), reason="the reference environment is absent"
            ),
        )
        for cls in ADAPTERS
    ],
)
def test_the_driver_runs_the_authors_code_in_its_own_environment(adapter_class):
    d, edges, pair = CASES["collider"]
    adapter = adapter_class()
    X = planted(d, edges, 8000, 0)
    params = {**adapter.params(), VARIABLES_PARAM: list(pair)}
    [run] = runner.run_task(adapter, X, params, cap_wall_s=300, repeats=1, warmup=0)
    assert run["status"] == "ok", run
    records, n_tests = answer_records(adapter, run["adj"], params)
    here, here_tests = answer_records(adapter, adapter.fit(X, params), params)
    assert records == here and n_tests == here_tests
    assert {k for r in records for k in r} == {
        "treatment",
        "outcome",
        "estimand",
        "nonzero",
        "identifiable",
        "adjustment_sets",
        "set_kind",
    }
