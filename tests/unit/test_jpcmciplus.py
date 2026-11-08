"""J-PCMCI+: the stacked design, the node test, the facade's checks, and its metadata."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.core import ci
from andrey.core.ci import SingularCorrelationError
from andrey.core.knowledge import Knowledge, KnowledgeError
from andrey.core.score import CopiedColumnsError
from andrey.data import sample_context_series
from andrey.temporal import jpcmciplus as engine
from andrey.temporal import pcmciplus as pcmciplus_engine
from andrey.temporal.jpcmciplus import ContextTest, Nodes, one_hot, pooled_design


def _datasets(n_datasets: int = 3, n_steps: int = 40, seed: int = 0) -> list[np.ndarray]:
    return list(np.random.default_rng(seed).standard_normal((n_datasets, n_steps, 2)))


def test_one_hot_marks_each_level():
    assert one_hot(np.array([2, 0, 2])).tolist() == [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]


def test_the_design_keeps_every_lag_window_inside_its_dataset():
    first = np.arange(8.0).reshape(4, 2)
    second = 100.0 + np.arange(8.0).reshape(4, 2)
    nodes = Nodes(2, 0, 1, True)
    design, fixed = pooled_design([first, second], np.array([[5.0], [7.0]]), nodes, 1)
    # Two rows per dataset, of lags 0 to 2 of two series, one space context, and two dummies.
    assert design.shape == (4, 6 + 1 + 2 + 2)
    assert design[:, 0].tolist() == [4.0, 6.0, 104.0, 106.0]
    assert design[:, 4].tolist() == [0.0, 2.0, 100.0, 102.0]
    assert fixed == ((6,), (7, 8), (9, 10))
    assert design[:, 6].tolist() == [5.0, 5.0, 7.0, 7.0]
    assert design[:, 7].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert design[:, 9].tolist() == [1.0, 0.0, 1.0, 0.0]


class _Fixed:
    """Answers every query with one p-value, and records the queries."""

    def __init__(self, p_value: float) -> None:
        self.p_value = p_value
        self.queries: list[tuple[int, int, tuple[int, ...]]] = []

    def statistic_pvalues(self, x, y, conditions):
        self.queries += [
            (int(a), int(b), tuple(int(c) for c in row)) for a, b, row in zip(x, y, conditions)
        ]
        return np.full(len(x), 0.5), np.full(len(x), self.p_value)


def _context_test(answer: _Fixed, rows: int = 100) -> ContextTest:
    """Two system variables, a time context, a space context, and both dummies, at ``tau_max=1``.

    The design columns are the 9 lagged series, the space context 9, the three levels of the
    space dummy 10 to 12, and the two of the time dummy 13 and 14; the design has ``rows`` rows.
    """
    return ContextTest(answer, Nodes(2, 1, 1, True), ((9,), (10, 11, 12), (13, 14)), 1, rows)


def _ask(test: ContextTest, x: int, y: int, conditions: list[int]) -> tuple[float, float]:
    statistics, p_values = test.statistic_pvalues(
        np.array([x]), np.array([y]), np.array([conditions], dtype=np.intp).reshape(1, -1)
    )
    return float(statistics[0]), float(p_values[0])


def test_a_dummy_is_tested_level_by_level_with_bonferroni():
    answer = _Fixed(0.01)
    assert _ask(_context_test(answer), 4, 0, []) == (0.5, 0.03)
    assert [query[0] for query in answer.queries] == [10, 11, 12]


def test_two_levels_are_one_test():
    answer = _Fixed(0.01)
    assert _ask(_context_test(answer), 5, 0, []) == (0.5, 0.01)
    assert [query[0] for query in answer.queries] == [13]


def test_a_dummy_conditions_on_all_its_levels_but_one():
    answer = _Fixed(0.5)
    _ask(_context_test(answer), 0, 1, [4])
    assert answer.queries == [(0, 1, (10, 11))]


def test_a_context_leaves_the_conditions_given_its_dummy():
    answer = _Fixed(0.5)
    # The space context (node 3), the time context at lag 1 (node 2, column 6 + 2), and the dummies.
    _ask(_context_test(answer), 0, 1, [3, 6 + 2, 4, 5])
    assert answer.queries == [(0, 1, (10, 11, 13))]


def test_a_context_given_its_dummy_separates_nothing():
    answer = _Fixed(0.9)
    assert _ask(_context_test(answer), 3, 0, [4]) == (1.0, 0.0)
    assert _ask(_context_test(answer), 2, 0, [5]) == (1.0, 0.0)
    assert not answer.queries


def test_lagged_columns_of_a_node_without_lags_are_its_own():
    answer = _Fixed(0.5)
    _ask(_context_test(answer), 0, 1, [3, 6 + 3])
    assert answer.queries == [(0, 1, (9,))]


def test_a_test_needs_a_degree_of_freedom_beyond_its_own_conditions():
    # Given the space dummy, two columns, a test runs on 6 rows and not on 5; with fewer
    # conditions, 5 rows are enough.
    _ask(_context_test(_Fixed(0.5), rows=6), 0, 1, [4])
    _ask(_context_test(_Fixed(0.5), rows=5), 0, 1, [3])
    with pytest.raises(ValueError, match="at least 6 rows"):
        _ask(_context_test(_Fixed(0.5), rows=5), 0, 1, [4])


@pytest.mark.parametrize(
    ("data", "kwargs", "message"),
    [
        (_datasets(1), {}, "at least two datasets"),
        ([np.zeros(5), np.zeros(5)], {}, "2-D table"),
        ([np.ones((9, 2)), np.ones((9, 3))], {}, "same columns"),
        (_datasets(), {"time_contexts": [0, 1]}, "at least 2 system variables"),
        (_datasets(), {"time_contexts": [0], "space_contexts": [0]}, "both a time context"),
        (_datasets(), {"time_contexts": [0]}, "at least 2 system variables"),
        (_datasets(n_steps=2), {"tau_max": 1}, "more than 2 \\* tau_max"),
        (_datasets(2, 3), {"tau_max": 1}, "at least 4 rows after the first"),
        (_datasets(), {"time_contexts": [5]}, "time_contexts"),
    ],
)
def test_it_refuses_what_it_cannot_read(data, kwargs, message):
    with pytest.raises(ValueError, match=message):
        andrey.jpcmciplus(data, **kwargs)


def test_a_time_context_must_agree_across_datasets():
    data = [np.column_stack([d, np.arange(40.0) + k]) for k, d in enumerate(_datasets())]
    with pytest.raises(ValueError, match="time context 2 must hold the same values"):
        andrey.jpcmciplus(data, time_contexts=[2])


def test_a_space_context_must_be_constant_within_a_dataset():
    data = [np.column_stack([d, np.arange(40.0)]) for d in _datasets()]
    with pytest.raises(ValueError, match="space context 2 varies"):
        andrey.jpcmciplus(data, space_contexts=[2])


def test_a_context_list_must_be_a_list():
    with pytest.raises(TypeError, match="time_contexts must be a list"):
        andrey.jpcmciplus(_datasets(), time_contexts=0)


def test_nan_is_refused():
    data = _datasets()
    data[0][3, 0] = np.nan
    with pytest.raises(ValueError, match="jpcmciplus"):
        andrey.jpcmciplus(data)


@pytest.mark.parametrize("indep_test", [*ci.DISCRETE_INDEP_TESTS, *ci.MIXED_INDEP_TESTS])
def test_a_categorical_or_mixed_data_test_is_refused(indep_test):
    with pytest.raises(ValueError, match="indep_test must be one of"):
        andrey.jpcmciplus(_datasets(), indep_test=indep_test)


def test_a_registered_test_is_taken(monkeypatch):
    monkeypatch.setitem(ci._CI_REGISTRY, "delegate", lambda data, **_: ci.FisherZ(data))
    out = andrey.jpcmciplus(_datasets(), indep_test="delegate", time_dummy=False)
    assert out.metadata["indep_test"] == "delegate"


def test_the_time_dummy_is_used_when_the_datasets_allow_it():
    uneven = _datasets()
    uneven[0] = uneven[0][:30]
    assert andrey.jpcmciplus(uneven).metadata["time_dummy"] is False
    assert andrey.jpcmciplus(_datasets()).metadata["time_dummy"] is True
    assert andrey.jpcmciplus(_datasets(), time_dummy=True).metadata["time_dummy"] is True
    assert andrey.jpcmciplus(_datasets(), time_dummy=False).metadata["time_dummy"] is False
    with pytest.raises(ValueError, match="time_dummy=True needs datasets of one length"):
        andrey.jpcmciplus(uneven, time_dummy=True)
    with pytest.raises(ValueError, match="time_dummy=True needs more than one step"):
        andrey.jpcmciplus(_datasets(4, 3), time_dummy=True)


def test_names_reach_the_structure_and_the_metadata():
    rng = np.random.default_rng(1)
    frames = []
    for level in rng.standard_normal(6):
        x = level + 0.5 * rng.standard_normal(80)
        frames.append(
            pd.DataFrame(
                {"x": x, "site": level, "y": rng.standard_normal(80)}, columns=["x", "site", "y"]
            )
        )
    out = andrey.jpcmciplus(frames, space_contexts=["site"], time_dummy=False)
    assert out.structure.lag(0).labels == ("x", "y")
    assert out.metadata["context_links"] == [["site", 0, "x"]]


def test_the_data_check_names_the_columns():
    frames = [pd.DataFrame(d, columns=["x", "y"]).assign(site=1.0) for d in _datasets()]
    with pytest.raises(ValueError, match="'site' is constant"):
        andrey.jpcmciplus(frames, space_contexts=["site"])


def test_a_context_that_tells_the_datasets_apart_does_not_break_a_test():
    # With two datasets, the one level of the space dummy repeats the space context.
    rng = np.random.default_rng(2)
    data = []
    for level in (0.0, 1.0):
        x = level + rng.standard_normal(100)
        data.append(np.column_stack([x, rng.standard_normal(100), np.full(100, level)]))
    out = andrey.jpcmciplus(data, space_contexts=[2], time_dummy=False)
    assert out.metadata["context_links"] == [[2, 0, 0]]
    assert not out.metadata["index_edges"]


def _with_spaces(values: list[np.ndarray], n_steps: int = 60) -> list[np.ndarray]:
    """Noise over two system variables, then a space context for each array of ``values``."""
    return [
        np.column_stack([block, *[np.full(n_steps, value[k]) for value in values]])
        for k, block in enumerate(_datasets(len(values[0]), n_steps))
    ]


def test_copied_space_contexts_are_refused():
    levels = np.arange(4.0) ** 2
    with pytest.raises(CopiedColumnsError, match="3 repeats 2"):
        andrey.jpcmciplus(
            _with_spaces([levels, 3 * levels + 1]), space_contexts=[2, 3], time_dummy=False
        )


def test_two_datasets_tell_only_one_space_context():
    data = _with_spaces([np.array([0.0, 1.0]), np.array([5.0, -1.0])])
    with pytest.raises(CopiedColumnsError, match="3 repeats 2"):
        andrey.jpcmciplus(data, space_contexts=[2, 3], time_dummy=False)


def test_an_empty_dataset_is_refused_before_its_contexts_are_read():
    data = _with_spaces([np.arange(2.0)])
    with pytest.raises(ValueError, match="more than 2 \\* tau_max"):
        andrey.jpcmciplus([np.zeros((0, 3)), *data], space_contexts=[2])


def test_a_system_variable_that_sums_others_raises():
    rng = np.random.default_rng(0)
    data = []
    for _ in range(4):
        a, b, c = rng.standard_normal((3, 200))
        data.append(np.column_stack([a, b, a + b, a + 0.3 * c]))
    with pytest.raises(SingularCorrelationError, match="span"):
        andrey.jpcmciplus(data, time_dummy=False)


def test_a_series_every_dataset_shares_is_a_time_context():
    shared = np.random.default_rng(1).standard_normal(60)
    data = [np.column_stack([block, shared]) for block in _datasets(4, 60)]
    with pytest.raises(SingularCorrelationError) as caught:
        andrey.jpcmciplus(data)
    assert "as a time context" in caught.value.__notes__[0]
    assert andrey.jpcmciplus(data, time_contexts=[2]).metadata["time_dummy"]


def test_a_pulse_is_a_time_context_beside_the_time_dummy():
    pulse = np.zeros(40)
    pulse[20] = 1.0
    data = [np.column_stack([block, pulse]) for block in _datasets(3, 40)]
    out = andrey.jpcmciplus(data, time_contexts=[2])
    assert out.metadata["time_dummy"]


def test_normal_scores_reads_two_datasets():
    out = andrey.jpcmciplus(_datasets(2, 100), indep_test="normal_scores", time_dummy=False)
    assert out.metadata["indep_test"] == "normal_scores"


def test_the_rows_are_checked_against_the_conditions_each_test_uses():
    # Five datasets of ten steps at tau_max=2 leave 30 rows, fewer than the largest conditioning
    # set the design allows, and every test the fit runs still fits in them.
    series = sample_context_series(
        n_system=5, n_time=2, n_space=2, n_datasets=5, n_steps=10, tau_max=2, observed=0.5, seed=0
    )
    out = andrey.jpcmciplus(
        list(series.datasets),
        time_contexts=list(series.time_contexts),
        space_contexts=list(series.space_contexts),
        tau_max=2,
        alpha=0.05,
    )
    assert out.metadata["time_dummy"]


def test_knowledge_reads_the_system_variables():
    data = _datasets(4, 60)
    required = Knowledge(required=[(0, 1, 1)], forbidden=[(1, 0)])
    out = andrey.jpcmciplus(data, knowledge=required, time_dummy=False)
    assert out.structure.lag(1).to_numpy()[1, 0] == andrey.core.ARROW
    assert out.metadata["p_values"][1][0][1] == 0.0
    for lag in out.structure.lags:
        assert out.structure.lag(lag).to_numpy()[0, 1] != andrey.core.ARROW
    spaces = [np.column_stack([block, np.full(60, float(k))]) for k, block in enumerate(data)]
    for knowledge in (Knowledge(forbidden=[(2, 0)]), Knowledge(tiers=[[0], [-1]])):
        with pytest.raises(KnowledgeError, match="the 2 system variables"):
            andrey.jpcmciplus(spaces, space_contexts=[2], knowledge=knowledge)


@pytest.mark.parametrize("kind", ["time", "space"])
def test_knowledge_that_names_a_context_is_refused(kind):
    frames = [
        pd.DataFrame(d, columns=["x", "y"]).assign(c=float(k)) for k, d in enumerate(_datasets())
    ]
    if kind == "time":
        frames = [frame.assign(c=np.arange(40.0)) for frame in frames]
    with pytest.raises(KnowledgeError, match=f"'c', a {kind} context"):
        andrey.jpcmciplus(
            frames, knowledge=Knowledge(forbidden=[("c", "x")]), **{f"{kind}_contexts": ["c"]}
        )


def test_caps_bound_the_conditions(monkeypatch):
    """Under ``max_depth=0`` the lagged phase tests without conditions, and under
    ``max_p_global=0`` the later passes search no subset beyond the default conditions."""
    series = sample_context_series(n_system=3, n_time=1, n_space=1, n_steps=60, seed=3)
    lagged: list[int] = []
    subsets: list[int] = []
    pc1, search = engine.pc1, pcmciplus_engine._test_subsets

    class Recorded:
        def __init__(self, test):
            self.test = test

        def statistic_pvalues(self, x, y, conditions):
            lagged.append(np.shape(conditions)[1])
            return self.test.statistic_pvalues(x, y, conditions)

    def tested(test, n_vars, parents, i, lag, j, found):
        subsets.append(len(found[0]))
        return search(test, n_vars, parents, i, lag, j, found)

    monkeypatch.setattr(engine, "pc1", lambda test, *args: pc1(Recorded(test), *args))
    monkeypatch.setattr(pcmciplus_engine, "_test_subsets", tested)

    def largest(**caps) -> tuple[int, int]:
        lagged.clear()
        subsets.clear()
        andrey.jpcmciplus(
            list(series.datasets),
            time_contexts=list(series.time_contexts),
            space_contexts=list(series.space_contexts),
            time_dummy=False,
            **caps,
        )
        return max(lagged), max(subsets)

    assert min(largest()) > 0
    assert largest(max_depth=0, max_p_global=0) == (0, 0)


def test_a_nonlinear_test_runs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = andrey.jpcmciplus(
            _datasets(2, 12), indep_test="gpdc", time_dummy=False, max_p_global=0, seed=0
        )
    assert out.metadata["indep_test"] == "gpdc"
    assert out.metadata["seed"] == 0


def test_the_generator_holds_its_links_and_contexts():
    series = sample_context_series(
        n_system=4, n_time=2, n_space=1, n_datasets=3, n_steps=50, observed=2 / 3, seed=0
    )
    width = 4 + len(series.time_contexts) + len(series.space_contexts)
    assert width == 6 and all(d.shape == (50, width) for d in series.datasets)
    assert len(series.context_links) == 2 and len(series.hidden_links) == 1
    for column in series.time_contexts:
        assert all(
            np.array_equal(d[:, column], series.datasets[0][:, column]) for d in series.datasets
        )
    for column in series.space_contexts:
        assert all(np.ptp(d[:, column]) == 0 for d in series.datasets)
    pooled = np.vstack([d[:, :4] for d in series.datasets])
    assert np.allclose(pooled.std(axis=0), 1.0)


def test_the_generator_refuses_what_it_cannot_draw():
    with pytest.raises(ValueError, match="contexts"):
        sample_context_series(n_system=2, n_time=2, n_space=1)
    with pytest.raises(ValueError, match="links"):
        sample_context_series(n_system=3, n_time=1, n_space=0, n_links=4)
