"""Unit tests for ``locpc_cde``: the record, the arguments, the test count, and the knowledge."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from andrey import Knowledge, locpc_cde
from andrey.constraint.blanket import LocalSearch
from andrey.constraint.locpc import Memory, locpc, non_orientable
from andrey.constraint.locpc import locpc_cde as engine
from andrey.core.ci import counting_tests
from andrey.core.effects import check_effects
from andrey.data.latent import DSepOracle

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")


def frame(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """a -> x <- c, x -> y <- b, y -> m: the effect of x on y is identifiable (a -> x <- c
    directs x -> y by Meek's R1, b -> y collides with x) and the set is {b}."""
    rng = np.random.default_rng(seed)
    a, b, c = rng.standard_normal((3, n))
    x = a + c + rng.standard_normal(n)
    y = x + b + rng.standard_normal(n)
    m = y + rng.standard_normal(n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "m": m, "x": x, "y": y})


def test_the_record_holds_the_seven_fields_and_the_reason():
    out = locpc_cde(frame(), treatment="x", outcome="y")
    records = out.metadata["effects"]
    check_effects(records, columns=list(out.structure.labels))
    (record,) = records
    assert record["estimand"] == "direct" and record["reason"] == "oriented"
    assert record["identifiable"] is True and record["nonzero"] is True
    assert record["adjustment_sets"] == [["b"]] and record["set_kind"] == "parents_of_outcome"
    assert out.structure.kind == "cpdag"
    assert out.metadata["visited"][0] == "y"
    assert isinstance(out.metadata["hop"], int) and out.metadata["n_tests"] > 0
    assert all(len(pair) == 2 for pair in out.metadata["double_bar"])


def test_a_treatment_that_is_the_outcomes_child_or_not_adjacent_is_a_zero_effect():
    data = frame()
    child = locpc_cde(data, treatment="m", outcome="y").metadata["effects"][0]
    assert child["reason"] == "treatment is a child" and child["nonzero"] is False
    apart = locpc_cde(data, treatment="a", outcome="y").metadata["effects"][0]
    assert apart["reason"] == "not adjacent" and apart["nonzero"] is False


def test_positions_and_labels_give_the_same_answer():
    data = frame()
    by_label = locpc_cde(data, treatment="x", outcome="y").metadata["effects"][0]
    by_position = locpc_cde(data.to_numpy(), treatment=4, outcome=5).metadata["effects"][0]
    assert by_position["adjustment_sets"] == [[1]]
    assert by_position["reason"] == by_label["reason"]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"outcome": "y"}, ValueError),
        ({"treatment": "x"}, ValueError),
        ({"treatment": "x", "outcome": "x"}, ValueError),
        ({"treatment": "q", "outcome": "y"}, ValueError),
    ],
)
def test_the_pair_is_required_and_distinct(kwargs, error):
    with pytest.raises(error):
        locpc_cde(frame(500), **kwargs)


def test_each_query_is_tested_once_and_counted():
    data = frame()
    out = locpc_cde(data, treatment="x", outcome="y")
    dag = nx.DiGraph([(0, 4), (2, 4), (4, 5), (1, 5), (5, 3)])
    search = LocalSearch(DSepOracle(dag, nodes=list(range(6))), list(range(6)), 0.5)
    with counting_tests() as tests:
        answer = engine(search, 4, 5)
    assert answer.reason == "oriented"
    assert tests.n == search.counts.issued
    assert out.metadata["n_tests"] > 0


def test_a_rerun_with_kept_memory_runs_no_new_test():
    dag = nx.DiGraph([(0, 4), (2, 4), (4, 5), (1, 5), (5, 3)])
    search = LocalSearch(DSepOracle(dag, nodes=list(range(6))), list(range(6)), 0.5)
    memory = Memory()
    first = locpc(search, 5, 2, memory)
    with counting_tests() as tests:
        again = locpc(search, 5, 2, memory)
    assert tests.n == 0
    assert (first.marks == again.marks).all() and first.double_bar == again.double_bar


def test_max_depth_caps_every_conditioning_set():
    seen = []

    class Spy(DSepOracle):
        def __call__(self, x, y, condition_set=()):
            seen.append(len(tuple(condition_set)))
            return super().__call__(x, y, condition_set)

    dag = nx.DiGraph([(0, 4), (2, 4), (4, 5), (1, 5), (5, 3), (0, 1)])
    search = LocalSearch(Spy(dag, nodes=list(range(6))), list(range(6)), 0.5, max_depth=1)
    engine(search, 4, 5)
    assert seen and max(seen) <= 1


def test_forbidden_pairs_never_join_and_required_edges_are_never_tested():
    data = frame()
    knowledge = Knowledge(forbidden=[("b", "y"), ("y", "b")], required=[("x", "y")])
    out = locpc_cde(data, treatment="x", outcome="y", knowledge=knowledge)
    marks = out.structure.to_numpy()
    labels = list(out.structure.labels)
    assert marks[labels.index("b"), labels.index("y")] == 0
    assert marks[labels.index("x"), labels.index("y")] != 0


def test_tiers_direct_the_edges_they_decide():
    data = frame()
    knowledge = Knowledge(tiers=[["a", "b", "c"], ["x"], ["y"], ["m"]])
    out = locpc_cde(data, treatment="x", outcome="y", knowledge=knowledge)
    record = out.metadata["effects"][0]
    assert record["reason"] == "oriented" and record["adjustment_sets"] == [["b"]]


def test_the_criterion_reads_undirected_and_double_bar_edges_leaving_the_set():
    """Definition 12 on a hand-built graph: an undirected edge leaving the set fails it, one
    double-bar edge per member passes, two fail."""
    from andrey.constraint.locpc import LocalEssentialGraph

    marks = np.zeros((4, 4), dtype=np.int8)
    for i, j in ((0, 1), (1, 2), (1, 3)):
        marks[i, j] = marks[j, i] = 1
    one = LocalEssentialGraph(marks, frozenset({(1, 2)}), (0, 1), (0, 1), 1)
    assert not non_orientable(one, [0, 1])
    both = LocalEssentialGraph(marks, frozenset({(1, 2), (1, 3)}), (0, 1), (0, 1), 1)
    assert not non_orientable(both, [0, 1])
    marks[1, 3] = marks[3, 1] = 0
    single = LocalEssentialGraph(marks, frozenset({(1, 2)}), (0, 1), (0, 1), 1)
    assert non_orientable(single, [0, 1])


def test_units_shifts_and_reruns_do_not_change_the_answer():
    data = frame()
    scaled = data * np.array([3.0, 0.01, 7.0, 2.0, 1e3, 0.5]) + 11.0
    first = locpc_cde(data, treatment="x", outcome="y").metadata["effects"]
    second = locpc_cde(scaled, treatment="x", outcome="y").metadata["effects"]
    again = locpc_cde(data, treatment="x", outcome="y").metadata["effects"]
    assert first == second == again


def test_missing_values_are_deleted_test_by_test():
    data = frame().copy()
    data.iloc[::17, 0] = np.nan
    out = locpc_cde(data, treatment="x", outcome="y")
    assert out.metadata["effects"][0]["reason"] == "oriented"
    assert "rows_tested" in out.metadata
