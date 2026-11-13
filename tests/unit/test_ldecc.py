"""Unit tests for ``ldecc``: the record and the graph it returns, its arguments, and the order,
units, and reruns it must not depend on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from andrey import ldecc
from andrey.constraint import ldecc as engine
from andrey.constraint.blanket import local_search
from andrey.core.ci import counting_tests
from andrey.core.effects import check_effects

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")


def frame(n: int = 3000, seed: int = 3) -> pd.DataFrame:
    """Two independent causes ``a`` and ``b`` of ``x`` that also cause ``y``, a mediator ``m``, a
    child ``k`` of ``x`` with another parent ``s`` (a spouse of ``x``), and an unrelated ``u``."""
    rng = np.random.default_rng(seed)
    a, b, s, u = rng.standard_normal((4, n))
    x = a + b + rng.standard_normal(n)
    m = x + rng.standard_normal(n)
    k = x + s + rng.standard_normal(n)
    y = m + a + b + rng.standard_normal(n)
    return pd.DataFrame({"a": a, "b": b, "s": s, "u": u, "x": x, "m": m, "k": k, "y": y})


def answer(out) -> tuple:
    """The record, its sets as sets, and the edges, so the column order does not show."""
    record = out.metadata["effects"][0]
    return (
        record["nonzero"],
        record["identifiable"],
        frozenset(frozenset(s) for s in record["adjustment_sets"]),
        record["set_kind"],
        frozenset(out.structure.oriented_edges()),
    )


def test_the_edges_at_the_treatment_and_one_record():
    df = frame()
    out = ldecc(df, treatment="x", outcome="y")
    records = out.metadata["effects"]
    check_effects(records, list(df.columns))
    assert records == [
        {
            "treatment": "x",
            "outcome": "y",
            "estimand": "total",
            "nonzero": None,
            "identifiable": True,
            "adjustment_sets": [["a", "b"]],
            "set_kind": "local_parents",
        }
    ]
    assert out.structure.kind == "cpdag"
    assert set(out.structure.oriented_edges()) == {
        ("a", "x", "directed"),
        ("b", "x", "directed"),
        ("x", "m", "directed"),
        ("x", "k", "directed"),
    }
    assert isinstance(out.metadata["n_tests"], int) and out.metadata["n_tests"] > 0


def test_undirected_neighbors_give_one_set_each():
    """In a chain ``a - x - y`` the edges cannot be directed: the sets are those of the two
    DAGs where ``a`` is a parent or not."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal(3000)
    x = a + rng.standard_normal(3000)
    y = x + rng.standard_normal(3000)
    out = ldecc(pd.DataFrame({"a": a, "x": x, "y": y}), treatment="x", outcome="y")
    record = out.metadata["effects"][0]
    assert record["identifiable"] is None and record["nonzero"] is None
    assert record["adjustment_sets"] == [[], ["a"]]
    assert set(out.structure.oriented_edges()) == {
        ("a", "x", "undirected"),
        ("x", "y", "undirected"),
    }


def test_positions_name_the_columns_of_an_array():
    df = frame()
    out = ldecc(df.to_numpy(), treatment=4, outcome=7)
    assert out.metadata["effects"][0]["adjustment_sets"] == [[0, 1]]


def test_the_answer_does_not_depend_on_the_column_order():
    df = frame()
    base = answer(ldecc(df, treatment="x", outcome="y"))
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(df.columns)
        assert answer(ldecc(df[order], treatment="x", outcome="y")) == base


def test_the_answer_does_not_depend_on_the_units():
    df = frame()
    factors = np.geomspace(1e-6, 1e6, df.shape[1])
    base = ldecc(df, treatment="x", outcome="y").metadata
    scaled = ldecc(df * factors + 3.0, treatment="x", outcome="y").metadata
    assert scaled["effects"] == base["effects"]
    assert scaled["n_tests"] == base["n_tests"]


def test_a_rerun_repeats_the_fit():
    df = frame(seed=2)
    first = ldecc(df, treatment="x", outcome="y").metadata
    assert ldecc(df, treatment="x", outcome="y").metadata == first


def test_the_pair_is_required_and_two_columns():
    df = frame(n=200)
    with pytest.raises(ValueError, match="pass treatment and outcome"):
        ldecc(df, treatment="x")
    with pytest.raises(ValueError, match="same column"):
        ldecc(df, treatment="x", outcome=4)
    with pytest.raises(ValueError):
        ldecc(df, treatment="x", outcome="nope")


def test_each_query_is_tested_once():
    """After the blanket step, the steps of LDECC ask each query once."""
    X = frame().to_numpy()
    with counting_tests() as tests:
        search = local_search(X, alpha=0.05, indep_test="fisherz", trace=True)
        search.local("iamb", 4)
        before = tests.n
        search.counts.issued, search.counts.queries = 0, set()
        engine.ldecc(search, 4)
    assert search.counts.queries is not None
    assert search.counts.issued == len(search.counts.queries) == tests.n - before


@pytest.mark.parametrize("blanket", ["gs", "inter_iamb", "total_conditioning"])
def test_another_blanket_learner_finds_the_same_answer(blanket):
    df = frame()
    base = answer(ldecc(df, treatment="x", outcome="y"))
    assert answer(ldecc(df, treatment="x", outcome="y", blanket=blanket)) == base


def test_max_depth_caps_every_test():
    df = frame()
    X = df.to_numpy()
    for depth in (0, 1):
        search = local_search(X, alpha=0.05, indep_test="fisherz", max_depth=depth, trace=True)
        engine.ldecc(search, 4)
        assert search.counts.queries
        assert max(len(s) for _, _, s in search.counts.queries) <= depth


def test_check_marks_parents_only_when_the_separators_agree():
    """Fig. 4, ECCParents: for two variables that are not neighbors of the treatment, ``check``
    marks their minimal neighbor separators only when the two are equal; a neighbor of the pair
    is always marked, with the other's separator."""
    search = local_search(frame(n=200).to_numpy(), alpha=0.05, indep_test="fisherz")
    tests = engine.Tests(search)
    ne = [1, 2, 3]
    cache = {5: (1,), 6: (2,), 7: (1,)}
    assert engine.ecc_parents(tests, 0, 5, 6, ne, True, cache) == set()
    assert engine.ecc_parents(tests, 0, 5, 6, ne, False, cache) == {1, 2}
    assert engine.ecc_parents(tests, 0, 5, 7, ne, True, cache) == {1}
    assert engine.ecc_parents(tests, 0, 3, 6, ne, True, cache) == {2, 3}
    assert engine.ecc_parents(tests, 0, 3, 6, ne, True, {6: None}) == set()


def test_missing_values_are_deleted_test_by_test():
    df = frame()
    df.iloc[::7, 0] = np.nan
    out = ldecc(df, treatment="x", outcome="y")
    assert out.metadata["missing"] == "testwise"
    assert out.metadata["rows_tested"]["min"] < len(df)


def test_conflicting_marks_give_no_answer():
    """Two parents marked as a non-collider at the treatment leave no locally valid parent set:
    the record has no set and does not decide."""
    from andrey.api.adjustment import _ldecc_record

    result = engine.Result(
        blanket=(1, 2, 3),
        neighbors=(1, 2, 3),
        parents=frozenset({1, 2}),
        children=frozenset({3}),
        unoriented=frozenset(),
        non_colliders=frozenset({frozenset({1, 2})}),
        parent_sets=(),
    )
    record = _ldecc_record(result, 0, 3, int)
    assert record["adjustment_sets"] == [] and record["set_kind"] is None
    assert record["identifiable"] is None and record["nonzero"] is None
