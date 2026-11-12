"""``y_structures``: the claim records, ``variables``, column order, per-column rescaling, reruns,
and the test count."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from andrey import y_structures
from andrey.constraint.blanket import LocalSearch
from andrey.constraint.ystructures import y_structures as engine
from andrey.core.effects import check_effects
from andrey.data.latent import DSepOracle

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")


def selected(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """The paper's fixed graph (Figure 5a): c -> x1, c -> x5 <- x4, x5 -> x6, x3 -> x2, with the
    rows kept when x1 + x2 > 1, a selection both depend on."""
    rng = np.random.default_rng(seed)
    m = 6 * n
    c, x3, x4 = rng.standard_normal((3, m))
    x1 = c + rng.standard_normal(m)
    x2 = x3 + rng.standard_normal(m)
    x5 = c + x4 + rng.standard_normal(m)
    x6 = x5 + rng.standard_normal(m)
    data = pd.DataFrame({"c": c, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "x6": x6})
    return data[x1 + x2 > 1].head(n).reset_index(drop=True)


def claims(out, *, ordered: bool = True) -> dict:
    """``{(treatment, outcome): (score, patterns)}``, patterns as a set of pairs."""
    return {
        (e["treatment"], e["outcome"]): (
            e["score"],
            {tuple(p) if ordered else frozenset(p) for p in e["patterns"]},
        )
        for e in out.metadata["effects"]
    }


@pytest.mark.parametrize("extended", [False, True])
def test_each_claim_is_an_effects_record_and_the_graph_has_no_edges(extended):
    data = selected()
    out = y_structures(data, extended=extended)
    records = out.metadata["effects"]
    check_effects(records, list(data.columns))
    assert ("x5", "x6") in claims(out)
    assert not out.structure.to_numpy().any()
    for e in records:
        assert (e["estimand"], e["nonzero"], e["identifiable"]) == ("total", True, True)
        assert (e["adjustment_sets"], e["set_kind"]) == ([[]], "valid")
        assert e["patterns"] and all(len(p) == 2 for p in e["patterns"])
    scores = [e["score"] for e in records]
    assert scores == sorted(scores, reverse=True)
    assert isinstance(out.metadata["n_tests"], int) and out.metadata["n_tests"] >= 21


def test_variables_keep_the_claims_about_their_pairs():
    """``variables`` answers only for pairs of them, and a pair gets the same claim either way."""
    data = selected()
    every = claims(y_structures(data))
    some = claims(y_structures(data, variables=["x6", "x5", "x1"]))
    assert some == {k: v for k, v in every.items() if set(k) <= {"x1", "x5", "x6"}}
    assert some


def test_the_y_structure_lists_each_pattern_once_and_the_extended_one_in_order():
    """The Y-structure's two variables play one role, so a pattern is one pair; the extended
    Y-structure's are ordered, the first separated from the effect by the cause."""
    found = claims(y_structures(selected()))["x5", "x6"][1]
    wider = claims(y_structures(selected(), extended=True))["x5", "x6"][1]
    assert all(v < w for v, w in found)
    assert {frozenset(p) for p in found} <= {frozenset(p) for p in wider}
    assert ("c", "x4") in wider and ("x4", "c") in wider


@pytest.mark.parametrize("extended", [False, True])
def test_the_claims_do_not_depend_on_the_column_order(extended):
    data = selected()
    want = claims(y_structures(data, extended=extended), ordered=extended)
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(data.columns)
        got = claims(y_structures(data[order], extended=extended), ordered=extended)
        assert got.keys() == want.keys()
        for key, (score, patterns) in got.items():
            assert patterns == want[key][1]
            assert score == pytest.approx(want[key][0], rel=1e-9)


def test_the_claims_do_not_depend_on_the_columns_units():
    data = selected()
    want = claims(y_structures(data))
    factors = 10.0 ** np.array([-6, 3, 0, 9, -2, 5, -8])
    got = claims(y_structures(data * factors))
    assert got.keys() == want.keys()
    for key, (score, patterns) in got.items():
        assert patterns == want[key][1]
        assert score == pytest.approx(want[key][0], rel=1e-9)


def test_a_rerun_gives_the_same_result():
    data = selected()
    assert y_structures(data).metadata == y_structures(data).metadata


def test_fewer_than_four_columns_claim_nothing():
    out = y_structures(selected()[["c", "x5", "x6"]])
    assert out.metadata["effects"] == []


@pytest.mark.parametrize(
    ("variables", "error"),
    [
        ("x5", TypeError),
        (["x5"], ValueError),
        (["x5", "x5"], ValueError),
        (["x9", "x5"], ValueError),
    ],
)
def test_variables_are_checked(variables, error):
    with pytest.raises(error):
        y_structures(selected(n=200), variables=variables)


def test_each_test_given_x_runs_once_for_a_pattern_it_can_serve():
    """Under d-separation on v -> x <- w, x -> y: the 6 marginal tests, then given each of the 4
    variables only the pairs a pattern there could use, each once."""
    dag = nx.DiGraph([(0, 2), (1, 2), (2, 3)])
    oracle = DSepOracle(dag)
    search = LocalSearch(oracle, [0, 1, 2, 3], 0.5, trace=True)
    found = engine(search, range(4))
    assert [(c.cause, c.effect, c.patterns) for c in found] == [(2, 3, ((0, 1),))]
    asked = search.counts.issued
    # Given 2: (0, 3) and (1, 3) separated, then (0, 1) connected. Given 3: (0, 2) and (1, 2), the
    # dependent pairs of a variable with an independent partner, neither separated. Given 0 or 1
    # no pair is independent, so no pattern there can use a test.
    assert asked == 6 + 3 + 2
