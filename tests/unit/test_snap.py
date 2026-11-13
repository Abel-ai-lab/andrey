"""``snap``: the answer's shape, the arguments, and the properties a fit must keep (the column
order, the columns' units, a rerun, one test per query, the cap on the conditioning sets)."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from andrey import snap
from andrey.constraint import snap as engine
from andrey.constraint.blanket import local_search
from andrey.core.ci import counting_tests
from andrey.core.effects import check_effects

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")


def frame(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Causes ``a`` and ``b`` of the treatment ``x``, a confounder ``a`` and a cause ``c`` of the
    outcome ``y``, and a child ``z`` of the outcome."""
    rng = np.random.default_rng(seed)
    a, b, c = rng.standard_normal((3, n))
    x = a + b + rng.standard_normal(n)
    y = x + 0.5 * a + c + rng.standard_normal(n)
    z = y + rng.standard_normal(n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "x": x, "y": y, "z": z})


def answer(out) -> tuple:
    """The records, the columns kept, and the edges by label, so the column order does not show."""
    records = {
        (r["treatment"], r["outcome"]): (
            r["nonzero"],
            r["identifiable"],
            frozenset(frozenset(s) for s in r["adjustment_sets"]),
            r["set_kind"],
        )
        for r in out.metadata["effects"]
    }
    return (
        records,
        frozenset(out.metadata["possible_ancestors"]),
        frozenset(
            (a, b, kind) if kind == "directed" else (frozenset((a, b)), kind)
            for a, b, kind in out.structure.oriented_edges()
        ),
    )


def test_a_record_per_ordered_pair_and_the_kept_columns():
    df = frame()
    out = snap(df, variables=["y", "x"])
    records = out.metadata["effects"]
    check_effects(records, list(df.columns))
    assert [(r["treatment"], r["outcome"]) for r in records] == [("x", "y"), ("y", "x")]
    forward, backward = records
    assert forward["identifiable"] and forward["nonzero"]
    assert forward["adjustment_sets"] == [["a", "c"]] and forward["set_kind"] == "optimal"
    # x -> y: the effect of y on x is zero, and no set adjusts for it, since none blocks the edge.
    assert backward["identifiable"] is False and backward["nonzero"] is False
    assert backward["adjustment_sets"] == [["a", "c"]] and backward["set_kind"] == "local_parents"
    # z is a descendant of both: it is not kept and has no edge.
    assert out.metadata["possible_ancestors"] == ["a", "b", "c", "x", "y"]
    assert not any("z" in e[:2] for e in out.structure.oriented_edges())
    assert out.structure.kind == "cpdag"
    assert isinstance(out.metadata["n_tests"], int) and out.metadata["n_tests"] > 0


def test_three_variables_give_six_records():
    out = snap(frame(), variables=["a", "x", "y"])
    pairs = [(r["treatment"], r["outcome"]) for r in out.metadata["effects"]]
    assert sorted(pairs) == sorted(permutations(["a", "x", "y"], 2))


def test_positions_name_the_columns_of_an_array():
    df = frame()
    out = snap(df.to_numpy(), variables=[4, 3])
    assert [(r["treatment"], r["outcome"]) for r in out.metadata["effects"]] == [(3, 4), (4, 3)]
    assert out.metadata["effects"][0]["adjustment_sets"] == [[0, 2]]


def test_the_answer_does_not_depend_on_the_column_order():
    df = frame()
    base = snap(df, variables=["x", "y"])
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(df.columns)
        out = snap(df[order], variables=["y", "x"])
        assert answer(out) == answer(base)
        assert out.metadata["n_tests"] == base.metadata["n_tests"]


def test_the_answer_does_not_depend_on_the_units():
    df = frame()
    factors = np.geomspace(1e-6, 1e6, df.shape[1])
    base = snap(df, variables=["x", "y"])
    scaled = snap(df * factors + 3.0, variables=["x", "y"])
    assert answer(scaled) == answer(base)


def test_a_rerun_repeats_the_fit():
    df = frame(seed=2)
    first = snap(df, variables=["x", "y"]).metadata
    assert snap(df, variables=["x", "y"]).metadata == first


def test_the_variables_are_required_and_distinct():
    df = frame(n=200)
    with pytest.raises(ValueError, match="pass variables"):
        snap(df)
    with pytest.raises(ValueError):
        snap(df, variables=["x"])
    with pytest.raises(ValueError, match="twice"):
        snap(df, variables=["x", 3])
    with pytest.raises(ValueError):
        snap(df, variables=["x", "nope"])
    with pytest.raises(TypeError):
        snap(df, variables="xy")


def test_each_query_is_tested_once():
    X = frame().to_numpy()
    with counting_tests() as tests:
        search = local_search(X, alpha=0.05, indep_test="fisherz", trace=True)
        engine.snap(search, [3, 4])
    assert search.counts.queries is not None
    assert tests.n == search.counts.issued == len(search.counts.queries)


def test_a_cap_the_search_does_not_reach_changes_nothing():
    df = frame()
    full = snap(df, variables=["x", "y"])
    assert snap(df, variables=["x", "y"], max_depth=len(df.columns)).metadata == full.metadata


def test_a_cap_that_stops_the_search_keeps_a_prefilter():
    """With unconditional tests only, the search stops with pairs left to test. The columns kept
    hold every possible ancestor of the pair, and here the outcome's child too, which a larger
    conditioning set drops. Only a zero effect is decided."""
    df = frame()
    capped = snap(df, variables=["x", "y"], max_depth=0)
    kept = set(capped.metadata["possible_ancestors"])
    assert {"a", "b", "c", "x", "y", "z"} <= kept
    for record in capped.metadata["effects"]:
        assert record["identifiable"] is None and record["adjustment_sets"] == []
        assert record["nonzero"] in (None, False)


def test_missing_values_are_deleted_test_by_test():
    df = frame()
    df.iloc[::7, 0] = np.nan
    out = snap(df, variables=["x", "y"])
    assert out.metadata["missing"] == "testwise"
    assert out.metadata["rows_tested"]["min"] < len(df)
