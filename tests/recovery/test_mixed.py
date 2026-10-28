"""Recovery on mixed data: the degenerate and conditional Gaussian tests and scores.

The fixed case has one answer: the network ``x0 -> x2 <- x1``, ``x2 -> x3``, ``x0 -> x4``, with
``x0`` and ``x4`` categorical, whose CPDAG orients the collider at ``x2`` and ``x2 -> x3`` and
leaves ``x0 - x4`` undirected. The ground-truth case is a random conditional linear Gaussian
network from :func:`andrey.data.sample_mixed`, scored against its CPDAG with a margin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.core import ARROW
from andrey.data import sample_mixed
from andrey.metrics import shd, to_cpdag

CATEGORICAL = [0, 4]
DIRECTED = {(0, 2), (1, 2), (2, 3)}
UNDIRECTED = {(0, 4)}


def _network(seed: int = 20261012, n: int = 3000) -> np.ndarray:
    """``x0`` (3 categories) and ``x1`` cause ``x2``, which causes ``x3``; ``x0`` causes ``x4``."""
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 3, n)
    x1 = rng.standard_normal(n)
    x2 = np.array([-1.5, 0.0, 1.5])[x0] + x1 + rng.standard_normal(n)
    x3 = x2 + rng.standard_normal(n)
    x4 = np.where(rng.random(n) < 0.8, x0 == 2, rng.integers(0, 2, n))
    return np.column_stack([x0, x1, x2, x3, x4]).astype(np.float64)


def _independent(seed: int = 20261013, n: int = 3000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.integers(0, 3, n),
            rng.standard_normal(n),
            rng.integers(0, 2, n),
            rng.standard_normal(n),
        ]
    ).astype(np.float64)


# Each fit that learns a CPDAG from mixed data, by its mixed-data setting.
CPDAG_FITS = {
    "pc-degen_gauss": lambda x, c: andrey.pc(x, indep_test="degen_gauss", categorical=c),
    "pc-cond_gauss": lambda x, c: andrey.pc(x, indep_test="cond_gauss", categorical=c),
    "ges-degen_gauss": lambda x, c: andrey.ges(x, score_func="degen_gauss", categorical=c),
    "ges-cond_gauss": lambda x, c: andrey.ges(x, score_func="cond_gauss", categorical=c),
    "hc-degen_gauss": lambda x, c: andrey.hc(x, score_func="degen_gauss", categorical=c),
    "hc-cond_gauss": lambda x, c: andrey.hc(x, score_func="cond_gauss", categorical=c),
    "tabu-degen_gauss": lambda x, c: andrey.tabu(x, score_func="degen_gauss", categorical=c),
    "tabu-cond_gauss": lambda x, c: andrey.tabu(x, score_func="cond_gauss", categorical=c),
    "boss-degen_gauss": lambda x, c: andrey.boss(x, score_func="degen_gauss", categorical=c),
    "boss-cond_gauss": lambda x, c: andrey.boss(x, score_func="cond_gauss", categorical=c),
    "grasp-degen_gauss": lambda x, c: andrey.grasp(x, score_func="degen_gauss", categorical=c),
    "grasp-cond_gauss": lambda x, c: andrey.grasp(x, score_func="cond_gauss", categorical=c),
    "cdnod-cond_gauss": lambda x, c: andrey.cdnod(
        x, np.zeros(len(x)), indep_test="cond_gauss", categorical=c
    ),
}
PAG_FITS = {
    "fci-degen_gauss": lambda x, c: andrey.fci(x, indep_test="degen_gauss", categorical=c),
    "fci-cond_gauss": lambda x, c: andrey.fci(x, indep_test="cond_gauss", categorical=c),
    "gfci-degen_gauss": lambda x, c: andrey.gfci(
        x, score_func="degen_gauss", indep_test="degen_gauss", categorical=c
    ),
    "gfci-cond_gauss": lambda x, c: andrey.gfci(
        x, score_func="cond_gauss", indep_test="cond_gauss", categorical=c
    ),
}


def _edges(out, kinds):
    return {(a, b) for a, b, kind in out.structure.oriented_edges() if kind in kinds}


@pytest.mark.parametrize("fit", list(CPDAG_FITS))
def test_cpdag_methods_recover_the_mixed_network(fit):
    out = CPDAG_FITS[fit](_network(), CATEGORICAL)
    assert out.structure.kind == "cpdag"
    assert _edges(out, {"directed"}) == DIRECTED
    assert _edges(out, {"undirected"}) == UNDIRECTED


@pytest.mark.parametrize("fit", list(PAG_FITS))
def test_pag_methods_find_the_skeleton_and_the_collider(fit):
    out = PAG_FITS[fit](_network(), CATEGORICAL)
    assert out.structure.kind == "pag"
    assert {(a, b) for a, b, _ in out.structure.oriented_edges()} == DIRECTED | UNDIRECTED
    marks = out.structure.to_numpy()
    assert marks[2, 0] == ARROW and marks[2, 1] == ARROW  # M[i, j] is the mark at i


@pytest.mark.parametrize("score_func", ["degen_gauss", "cond_gauss"])
def test_exact_search_finds_a_dag_in_the_true_class(score_func):
    out = andrey.exact_search(_network(), score_func=score_func, categorical=CATEGORICAL)
    assert out.structure.kind == "dag"
    cpdag = to_cpdag(out.structure)
    assert {(a, b) for a, b, k in cpdag.oriented_edges() if k == "directed"} == DIRECTED
    assert {(a, b) for a, b, k in cpdag.oriented_edges() if k == "undirected"} == UNDIRECTED


@pytest.mark.parametrize("fit", [*CPDAG_FITS, *PAG_FITS])
def test_independent_columns_give_no_edges(fit):
    out = {**CPDAG_FITS, **PAG_FITS}[fit](_independent(), [0, 2])
    assert out.structure.oriented_edges() == []


@pytest.mark.parametrize(
    "fit", ["pc-degen_gauss", "pc-cond_gauss", "ges-degen_gauss", "ges-cond_gauss"]
)
def test_the_graph_does_not_depend_on_codes_or_units(fit):
    """Each distinct value is a category, and a continuous column's units do not matter."""
    x = _network()
    recoded = x * np.array([3.0, 0.01, -2.0, 50.0, -1.0]) + np.array([-7.0, 4.0, 0.0, 11.0, 100.0])
    assert np.array_equal(
        CPDAG_FITS[fit](recoded, CATEGORICAL).structure.to_numpy(),
        CPDAG_FITS[fit](x, CATEGORICAL).structure.to_numpy(),
    )


def test_a_dataframe_marks_its_categorical_columns_by_dtype():
    x = _network()
    frame = pd.DataFrame(
        {
            "group": pd.Categorical(np.array(["low", "mid", "high"])[x[:, 0].astype(int)]),
            "u": x[:, 1],
            "v": x[:, 2],
            "w": x[:, 3],
            "flag": x[:, 4].astype(bool),
        }
    )
    out = andrey.pc(frame, indep_test="degen_gauss")
    assert set(out.structure.oriented_edges()) == {
        ("group", "v", "directed"),
        ("u", "v", "directed"),
        ("v", "w", "directed"),
        ("group", "flag", "undirected"),
    }
    named = andrey.pc(
        pd.DataFrame(x, columns=list("abcde")), indep_test="degen_gauss", categorical=["a", "e"]
    )
    assert np.array_equal(named.structure.to_numpy(), out.structure.to_numpy())


def test_cdnod_with_a_mixed_test_finds_a_shift_in_one_domain_of_three():
    """The index is categorical under a mixed-data test, so a shift in the middle domain orients
    ``x0 -> x1``, though every data column is continuous."""
    rng = np.random.default_rng(20261014)
    n = 3000
    domain = np.repeat([0, 1, 2], n // 3)
    x0 = np.where(domain == 1, 1.5, 0.0) + rng.standard_normal(n)
    x1 = x0 + rng.standard_normal(n)
    x = np.column_stack([x0, x1])
    for test in ("degen_gauss", "cond_gauss"):
        assert andrey.cdnod(x, domain, indep_test=test).structure.oriented_edges() == [
            (0, 1, "directed")
        ]


# A random network: 10 variables, 10 edges, half of them categorical, 5000 rows. On this draw every
# fit below finds the true CPDAG (SHD 0); the margin allows for a change in tie-breaking without
# hiding a broken build, which scores near the empty graph's 10.
_RANDOM = {"d": 10, "n": 5000, "seed": 4}
_MARGIN = 2


@pytest.fixture(scope="module")
def random_network():
    return sample_mixed(**_RANDOM)


@pytest.mark.parametrize(
    "fit",
    [
        "pc-degen_gauss",
        "pc-cond_gauss",
        "ges-degen_gauss",
        "ges-cond_gauss",
        "boss-degen_gauss",
        "boss-cond_gauss",
    ],
)
def test_a_random_network_is_recovered_within_the_margin(random_network, fit):
    truth = to_cpdag(random_network.graph)
    assert shd(np.zeros((_RANDOM["d"],) * 2), truth) == 10
    categorical = random_network.provenance["config"]["categorical"]
    assert shd(CPDAG_FITS[fit](random_network.data, categorical).structure, truth) <= _MARGIN


@pytest.mark.parametrize("score_func", ["degen_gauss", "cond_gauss"])
def test_hc_gives_no_categorical_node_a_continuous_parent(score_func):
    """Hill climbing keeps to conditional linear Gaussian networks, as the data were drawn."""
    ds = sample_mixed(d=6, n=1000, seed=2)
    categorical = ds.provenance["config"]["categorical"]
    out = andrey.hc(ds.data, score_func=score_func, categorical=categorical)
    into = [
        (a, b)
        for a, b, kind in out.structure.oriented_edges()
        if kind == "directed" and a not in categorical and b in categorical
    ]
    assert into == []
