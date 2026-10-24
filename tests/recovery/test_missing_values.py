"""Recovery with values missing completely at random, each test deleting the rows it cannot read.

The fixed cases are colliders whose complete data separate the independent pair clearly, with a
p-value above 0.5, and keep the graph that complete data give at every rate of missing values. On
seeded ground truth, a method's errors with values missing stay within ``ceil(lost * E)`` of its
errors on the complete data, and below the empty graph's ``E``: ``lost`` is the share of the rows
its median test deleted, read from ``metadata["rows_tested"]``. A test that reads fewer rows has
less power, and the margin allows one error more for each edge in ``1 / lost``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey import data
from andrey.core import ARROW, GraphStructure, TemporalStructure
from andrey.core.ci import make_indep_test
from andrey.data import functional, graphs, noise
from andrey.data.scm import SCM
from andrey.metrics import shd, temporal_scores
from andrey.metrics.graph import to_cpdag

RATES = [0.05, 0.1, 0.2]
IID = ["pc", "fci", "cdnod", "iamb", "inter_iamb", "mmpc"]
SERIES = ["pcmci", "pcmciplus", "lpcmci"]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        warnings.simplefilter("ignore", andrey.DataTypeWarning)
        yield


def _gaps(table, rate: float, seed: int = 7):
    """``table`` with each cell missing completely at random with probability ``rate``."""
    cells = np.random.default_rng(seed).random(np.shape(table)) < rate
    if isinstance(table, pd.DataFrame):
        return table.mask(cells)
    out = np.array(table, dtype=np.float64)
    out[cells] = np.nan
    return out


def _fit(name: str, table, **kwargs):
    if name == "cdnod":
        return andrey.cdnod(table, np.repeat([0.0, 1.0], len(table) // 2), **kwargs)
    return getattr(andrey, name)(table, **kwargs)


# --- fixed cases ----------------------------------------------------------------------------------


def _collider(n: int = 1000) -> pd.DataFrame:
    """x -> z <- y."""
    rng = np.random.default_rng(4)
    x, y, e = rng.standard_normal((3, n))
    return pd.DataFrame({"x": x, "y": y, "z": x + y + 0.5 * e})


def test_the_continuous_collider_separates_its_pair_clearly():
    assert make_indep_test("fisherz", _collider().to_numpy())(0, 1) > 0.5


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("name", ["pc", "fci", "cdnod"])
def test_a_collider_keeps_its_graph_with_missing_values(name, rate):
    complete = _fit(name, _collider())
    assert len(complete.structure.oriented_edges()) == 2
    gappy = _fit(name, _gaps(_collider(), rate))
    assert gappy.structure.oriented_edges() == complete.structure.oriented_edges()


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize(("name", "want"), [("iamb", {"x", "y"}), ("mmpc", {"x", "y"})])
def test_a_collider_keeps_its_sets_with_missing_values(name, want, rate):
    key = "markov_blanket" if name == "iamb" else "parents_and_children"
    out = _fit(name, _gaps(_collider(), rate), target="z")
    assert set(out.metadata[key]) == want


def _categorical_collider() -> pd.DataFrame:
    """a -> c <- b."""
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 2, (2, 2000))
    c = np.where(rng.random(2000) < 0.8, a + b, rng.integers(0, 3, 2000))
    return pd.DataFrame({"a": a, "b": b, "c": c}).astype(float)


def _mixed_collider() -> pd.DataFrame:
    """g -> v <- u, g categorical."""
    rng = np.random.default_rng(4)
    g = rng.integers(0, 3, 2000)
    u = rng.standard_normal(2000)
    v = np.array([-1.0, 0.0, 1.0])[g] + u + 0.5 * rng.standard_normal(2000)
    return pd.DataFrame({"g": g.astype(float), "u": u, "v": v})


@pytest.mark.parametrize("indep_test", ["chisq", "gsq"])
def test_the_categorical_collider_separates_its_pair_clearly(indep_test):
    assert make_indep_test(indep_test, _categorical_collider().to_numpy())(0, 1) > 0.5


@pytest.mark.parametrize("indep_test", ["degen_gauss", "cond_gauss"])
def test_the_mixed_collider_separates_its_pair_clearly(indep_test):
    table = _mixed_collider().to_numpy()
    mask = np.array([True, False, False])
    assert make_indep_test(indep_test, table, categorical=mask)(0, 1) > 0.5


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("indep_test", ["chisq", "gsq"])
def test_a_categorical_collider_keeps_its_graph_with_missing_values(indep_test, rate):
    want = [("a", "c", "directed"), ("b", "c", "directed")]
    table = _categorical_collider()
    assert andrey.pc(table, indep_test=indep_test).structure.oriented_edges() == want
    out = andrey.pc(_gaps(table, rate), indep_test=indep_test)
    assert out.structure.oriented_edges() == want


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("indep_test", ["degen_gauss", "cond_gauss"])
def test_a_mixed_collider_keeps_its_graph_with_missing_values(indep_test, rate):
    want = [("g", "v", "directed"), ("u", "v", "directed")]
    table = _mixed_collider()
    fit = andrey.pc(table, indep_test=indep_test, categorical=["g"])
    assert fit.structure.oriented_edges() == want
    out = andrey.pc(_gaps(table, rate), indep_test=indep_test, categorical=["g"])
    assert out.structure.oriented_edges() == want


def _series() -> np.ndarray:
    """x drives y one step later; each series depends on its own past."""
    rng = np.random.default_rng(0)
    e = rng.standard_normal((1500, 2))
    series = np.zeros((1500, 2))
    for t in range(1, 1500):
        series[t, 0] = 0.5 * series[t - 1, 0] + e[t, 0]
        series[t, 1] = 0.5 * series[t - 1, 1] + 0.6 * series[t - 1, 0] + e[t, 1]
    return series


def _lagged_links(out) -> dict[tuple[int, int, int], str]:
    """Every link ``(cause, effect, lag)`` with its marks; LPCMCI's lagged graphs keep only the
    definite causes, so its links come from ``metadata["links"]``."""
    if "links" in out.metadata:
        return {
            (i, j, lag): link
            for lag, rows in enumerate(out.metadata["links"])
            for i, row in enumerate(rows)
            for j, link in enumerate(row)
            if link
        }
    return {
        (int(i), int(j), lag): "-->"
        for lag in out.structure.lags
        for j, i in np.argwhere(out.structure.lag(lag).to_numpy() == ARROW)
    }


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("name", SERIES)
def test_planted_links_survive_missing_steps(name, rate):
    complete = _lagged_links(getattr(andrey, name)(_series(), tau_max=1))
    assert {link[:2] for link in complete} == {(0, 0), (1, 1), (0, 1)}
    assert _lagged_links(getattr(andrey, name)(_gaps(_series(), rate), tau_max=1)) == complete


# --- ground truth ---------------------------------------------------------------------------------


def _target(name: str, truth: GraphStructure) -> np.ndarray:
    """The adjacencies a method estimates: the moral graph for a Markov blanket, else the
    skeleton."""
    parents = truth.to_numpy() == ARROW  # [child, parent]
    target = parents | parents.T
    if name in ("iamb", "inter_iamb"):
        for child in range(parents.shape[0]):
            spouses = np.flatnonzero(parents[child])
            target[np.ix_(spouses, spouses)] = True
        np.fill_diagonal(target, False)
    return target


def _errors(name: str, out, truth: GraphStructure) -> int:
    """PC's errors against the true CPDAG; every other method's against its adjacencies."""
    if name == "pc":
        return shd(out.structure, to_cpdag(truth))
    adjacent = out.structure.to_numpy() != 0
    adjacent = adjacent | adjacent.T
    return int(np.triu(adjacent ^ _target(name, truth), 1).sum())


@pytest.fixture(scope="module")
def sample():
    return data.sample_scm(d=20, n=1000, seed=1, density=2.0, scale="standardize")


def _bound(complete: int, edges: int, out, rows: int) -> int:
    """The most errors a fit with missing values may make: ``complete``, plus one per edge in
    ``1 / lost``, ``lost`` the share of the ``rows`` its median test deleted."""
    lost = 1.0 - out.metadata["rows_tested"]["median"] / rows
    return min(complete + math.ceil(lost * edges), edges - 1)


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("name", IID)
def test_missing_values_cost_a_bounded_number_of_errors(sample, name, rate):
    truth = sample.observed_graph
    edges = int(np.triu(_target(name, truth), 1).sum())
    complete = _errors(name, _fit(name, sample.data), truth)
    out = _fit(name, _gaps(sample.data, rate))
    bound = _bound(complete, edges, out, len(sample.data))
    assert _errors(name, out, truth) <= bound, (complete, edges)


@pytest.mark.parametrize("rate", RATES)
def test_missing_categories_cost_a_bounded_number_of_errors(rate):
    sample = data.sample_discrete(d=10, n=3000, seed=1, density=2.0)
    truth = sample.observed_graph
    edges = int(np.triu(_target("pc", truth), 1).sum())
    complete = _errors("pc", andrey.pc(sample.data, indep_test="chisq"), truth)
    out = andrey.pc(_gaps(sample.data, rate), indep_test="chisq")
    bound = _bound(complete, edges, out, len(sample.data))
    assert _errors("pc", out, truth) <= bound, (complete, edges)


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("name", SERIES)
def test_missing_steps_cost_a_bounded_number_of_lagged_errors(name, rate):
    scm = SCM(
        graph=graphs.FixedDAG(nodes=tuple(f"x{i}" for i in range(5)), arcs=()),
        functional=functional.var(n_lags=2, lag_density=0.3),
        noise=noise.gaussian(),
    )
    sample = scm.sample(1500, 0, scale="raw")
    nothing = GraphStructure.from_numpy(np.zeros((5, 5), np.int8), kind="digraph")
    empty = TemporalStructure.from_lag_graphs([nothing, nothing], lags=[1, 2])
    edges = temporal_scores(empty, sample.graph)["lagged"]["shd"]

    def errors(out) -> int:
        return temporal_scores(out, sample.graph)["lagged"]["shd"]

    complete = errors(getattr(andrey, name)(sample.data, tau_max=2))
    out = getattr(andrey, name)(_gaps(sample.data, rate), tau_max=2)
    bound = _bound(complete, edges, out, len(sample.data) - 2 * 2)  # the lagged design's rows
    assert errors(out) <= bound, (complete, edges)
