"""Recovery on categorical data: the chi-square and G-square tests, BDeu, and the multinomial BIC.

The fixed cases have one answer: ``discrete_5v`` is the network ``x0 -> x2 <- x1``, ``x2 -> x3 ->
x4``, whose CPDAG has every edge oriented, and ``discrete_independent_4v`` has no edges. The
ground-truth case is a random network from :func:`andrey.data.sample_discrete`, scored against its
CPDAG with a margin, and the permutation searches are checked against exact search's optimum on
draws where that optimum's class is unique. Edges are named by variable and mapped to the case's
columns.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.core import ARROW, TAIL
from andrey.core.orient import pdag2dag as to_dag
from andrey.core.score import make_score
from andrey.data import sample_discrete
from andrey.metrics import shd, to_cpdag
from tests.recovery.helpers import datagen
from tests.recovery.helpers.project import found_dag

CASE = "discrete_5v"
COLUMN = datagen.column_of(CASE)  # the column of each variable
TRUTH = datagen.to_columns(CASE, {(0, 2), (1, 2), (2, 3), (3, 4)})

# Each fit that learns a CPDAG from categorical data, by its categorical setting.
CPDAG_FITS = {
    "pc-chisq": lambda x: andrey.pc(x, indep_test="chisq"),
    "pc-gsq": lambda x: andrey.pc(x, indep_test="gsq"),
    "ges-bdeu": lambda x: andrey.ges(x, score_func="bdeu"),
    "hc-bdeu": lambda x: andrey.hc(x, score_func="bdeu"),
    "ges-bic": lambda x: andrey.ges(x, score_func="discrete_bic"),
    "gies-bdeu": lambda x: andrey.gies(x, score_func="bdeu"),
    "gies-bic": lambda x: andrey.gies(x, score_func="discrete_bic"),
    "xges-bdeu": lambda x: andrey.xges(x, score_func="bdeu"),
    "xges-bic": lambda x: andrey.xges(x, score_func="discrete_bic"),
    "hc-bic": lambda x: andrey.hc(x, score_func="discrete_bic"),
    "tabu-bdeu": lambda x: andrey.tabu(x, score_func="bdeu"),
    "tabu-bic": lambda x: andrey.tabu(x, score_func="discrete_bic"),
    "boss-bdeu": lambda x: andrey.boss(x, score_func="bdeu"),
    "grasp-bdeu": lambda x: andrey.grasp(x, score_func="bdeu"),
    "boss-bic": lambda x: andrey.boss(x, score_func="discrete_bic"),
    "grasp-bic": lambda x: andrey.grasp(x, score_func="discrete_bic"),
    "cdnod-chisq": lambda x: andrey.cdnod(x, np.zeros(len(x)), indep_test="chisq"),
}
PAG_FITS = {
    "fci-chisq": lambda x: andrey.fci(x, indep_test="chisq"),
    "fci-gsq": lambda x: andrey.fci(x, indep_test="gsq"),
    "gfci-bdeu-chisq": lambda x: andrey.gfci(x, score_func="bdeu", indep_test="chisq"),
    "gfci-bic-gsq": lambda x: andrey.gfci(x, score_func="discrete_bic", indep_test="gsq"),
}

# Each exact search, which returns one DAG of the optimal class.
DAG_FITS = {
    f"exact-{method}-{name}": partial(andrey.exact_search, search_method=method, score_func=score)
    for name, score in (("bdeu", "bdeu"), ("bic", "discrete_bic"))
    for method in ("astar", "dp")
}


def _edges(out, kinds):
    return {(a, b) for a, b, kind in out.structure.oriented_edges() if kind in kinds}


@pytest.mark.parametrize("fit", list(CPDAG_FITS))
def test_cpdag_methods_recover_the_categorical_network(fit):
    out = CPDAG_FITS[fit](datagen.generate(CASE))
    assert out.structure.kind == "cpdag"
    assert _edges(out, {"directed"}) == TRUTH
    assert _edges(out, {"undirected"}) == set()


@pytest.mark.parametrize("fit", list(PAG_FITS))
def test_pag_methods_find_the_skeleton_and_the_collider(fit):
    out = PAG_FITS[fit](datagen.generate(CASE))
    assert out.structure.kind == "pag"
    marks = out.structure.to_numpy()
    skeleton = {tuple(sorted(edge)) for edge in TRUTH}
    assert {tuple(sorted((a, b))) for a, b, _ in out.structure.oriented_edges()} == skeleton
    # Arrowheads at x2 from both parents: the collider every method must see.
    x0, x1, x2 = COLUMN[:3]
    assert marks[x2, x0] == ARROW and marks[x2, x1] == ARROW  # M[i, j] is the mark at i


@pytest.mark.parametrize("fit", [*CPDAG_FITS, *PAG_FITS])
def test_independent_columns_give_no_edges(fit):
    out = {**CPDAG_FITS, **PAG_FITS}[fit](datagen.generate("discrete_independent_4v"))
    assert out.structure.oriented_edges() == []


@pytest.mark.parametrize("fit", list(CPDAG_FITS))
def test_the_codes_need_not_start_at_zero(fit):
    """Each distinct value is a category, so recoding a column leaves the graph unchanged."""
    x = datagen.generate(CASE)
    recoded = x * np.array([3.0, -1.0, 2.0, 5.0, 1.0]) + np.array([-7.0, 4.0, 0.0, 11.0, 100.0])
    assert np.array_equal(
        CPDAG_FITS[fit](recoded).structure.to_numpy(), CPDAG_FITS[fit](x).structure.to_numpy()
    )


def test_a_dataframe_names_the_nodes():
    x = datagen.generate(CASE)
    frame = pd.DataFrame(x, columns=["abcde"[v] for v in datagen.column_order(CASE)])
    out = andrey.pc(frame, indep_test="chisq")
    assert set(out.structure.oriented_edges()) == {
        ("a", "c", "directed"),
        ("b", "c", "directed"),
        ("c", "d", "directed"),
        ("d", "e", "directed"),
    }


@pytest.mark.parametrize("d", [1, 2])
@pytest.mark.parametrize("fit", ["ges", "gies", "xges", "hc", "boss", "grasp"])
def test_bdeu_fits_one_or_two_columns(fit, d):
    """With two or fewer variables the structure prior is left out, so the score stays finite."""
    x = datagen.generate(CASE)[:, COLUMN[[2, 3][:d]]]  # x2, and x3 with its parent x2
    out = getattr(andrey, fit)(x, score_func="bdeu")
    assert np.isfinite(out.metadata["score"])
    assert len(out.structure.oriented_edges()) == d - 1


def test_cdnod_chisq_finds_a_shift_in_one_domain_of_three():
    """The categorical index finds a shift in the middle domain, which Fisher-Z's linear read of the
    codes ``0, 1, 2`` misses; the shift orients ``x0 -> x1``."""
    rng = np.random.default_rng(20240621)
    n = 3000
    domain = np.repeat([0, 1, 2], n // 3)
    x0 = (rng.random(n) < np.where(domain == 1, 0.8, 0.3)).astype(int)
    x1 = np.where(rng.random(n) < 0.8, x0, rng.integers(0, 2, n))
    x = np.column_stack([x0, x1])
    assert andrey.cdnod(x, domain, indep_test="chisq").structure.oriented_edges() == [
        (0, 1, "directed")
    ]
    assert andrey.cdnod(x, domain).structure.oriented_edges() == [(0, 1, "undirected")]


# A random network: 10 variables, 10 edges, 5000 rows. On this draw every method listed finds the
# true CPDAG (SHD 0); the margin allows for a change in tie-breaking without hiding a broken build,
# which scores near the empty graph's 10. GRaSP with BDeu stops at a worse-scoring DAG on this draw,
# so it is checked against exact search instead.
_RANDOM = {"d": 10, "n": 5000, "seed": 5}
_MARGIN = 2


@pytest.fixture(scope="module")
def random_network():
    return sample_discrete(**_RANDOM)


@pytest.mark.parametrize(
    "fit",
    [
        *("pc-chisq", "pc-gsq", "ges-bdeu", "ges-bic", "gies-bdeu", "gies-bic"),
        *("xges-bdeu", "xges-bic", "boss-bdeu", "boss-bic", "grasp-bic"),
    ],
)
def test_a_random_network_is_recovered_within_the_margin(random_network, fit):
    truth = to_cpdag(random_network.graph)
    assert shd(np.zeros((_RANDOM["d"],) * 2), truth) == 10
    assert shd(CPDAG_FITS[fit](random_network.data), truth) <= _MARGIN


def test_a_fractional_value_is_refused_by_the_categorical_bic():
    x = datagen.generate(CASE)
    x[0, 0] = 0.5
    with pytest.raises(ValueError, match="whole number"):
        andrey.ges(x, score_func="discrete_bic")


@pytest.mark.parametrize("fit", list(DAG_FITS))
def test_exact_search_returns_the_categorical_network(fit):
    """Every edge of the network's CPDAG is oriented, so its class has the network alone."""
    out = DAG_FITS[fit](datagen.generate("discrete_5v"))
    assert out.structure.kind == "cpdag"
    assert _edges(out, {"directed"}) == TRUTH
    assert {(a, b) for a, b, _ in found_dag(out).oriented_edges()} == TRUTH


@pytest.mark.parametrize("score_func", ["bdeu", "discrete_bic"])
def test_the_searches_order_by_score_on_one_categorical_score(score_func):
    """Exact search finds the lowest total score, and tabu search never ends above hill climbing.
    On this draw, dense and with few rows, hill climbing stops short of the optimum."""
    data = sample_discrete(d=6, n=200, seed=8, density=4.0).data
    score = make_score(score_func, data)

    def total(structure):
        dag = to_dag(structure.to_numpy())
        return sum(score.score(v, np.flatnonzero(dag[:, v] == TAIL).tolist()) for v in range(6))

    hc = andrey.hc(data, score_func=score_func)
    tabu = andrey.tabu(data, score_func=score_func)
    exact = andrey.exact_search(data, score_func=score_func)
    assert hc.metadata["score"] == pytest.approx(total(hc.structure), rel=1e-12)
    assert tabu.metadata["score"] <= hc.metadata["score"]
    assert total(found_dag(exact)) <= tabu.metadata["score"] + 1e-9
    assert total(found_dag(exact)) < hc.metadata["score"] - 1.0


# Draws of sample_discrete, as (d, n, seed), whose optimal class is unique under BDeu and under the
# multinomial BIC: the next-best class scores far above the search tolerance. Hill climbing stops
# above the optimum on each.
_UNIQUE_OPTIMA = [(6, 5000, 2), (7, 2000, 1), (8, 5000, 2)]
_SCORES = {"bdeu": "bdeu", "bic": "discrete_bic"}


@pytest.mark.parametrize("fit", ["boss-bdeu", "grasp-bdeu", "boss-bic", "grasp-bic"])
@pytest.mark.parametrize("draw", _UNIQUE_OPTIMA)
def test_the_permutation_searches_find_the_exact_optimum(fit, draw):
    """BOSS and GRaSP return the class of exact search's DAG, at its score."""
    d, n, seed = draw
    score_func = _SCORES[fit.split("-")[1]]
    data = sample_discrete(d=d, n=n, seed=seed).data
    score = make_score(score_func, data)
    exact = found_dag(andrey.exact_search(data, score_func=score_func))
    optimum = sum(
        score.score(v, np.flatnonzero(exact.to_numpy()[:, v] == TAIL).tolist()) for v in range(d)
    )
    out = CPDAG_FITS[fit](data)
    np.testing.assert_array_equal(out.structure.to_numpy(), to_cpdag(exact).to_numpy())
    assert out.metadata["score"] == pytest.approx(optimum, abs=1e-6)
    assert andrey.hc(data, score_func=score_func).metadata["score"] > optimum + 1.0
