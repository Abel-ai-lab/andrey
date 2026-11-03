"""The search bounds: ``max_depth`` caps every conditioning set a method tests, ``max_parents``
every parent set it returns (gfci's: its GES phase's), and stopping on an iteration cap warns
``SearchLimitWarning``.

Each method that takes a bound is found from the registry, so a method that gains one is checked
here. A run without the cap first shows that the data make the method exceed it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

import andrey
from andrey.core import ci, warning_policy
from andrey.fcm import camuv as camuv_engine
from andrey.irregular import rcd as rcd_engine
from andrey.search import gfci as gfci_engine
from andrey.search.operators import max_parents_in_class
from andrey.spec import REGISTRY
from andrey.temporal import lpcmci as lpcmci_engine
from andrey.temporal import pcmci as pcmci_engine
from andrey.temporal import pcmciplus as pcmciplus_engine
from andrey.temporal import varma as varma_engine

CAPS = (0, 1, 2)


def methods_with(concept: str) -> list[str]:
    return sorted(
        m for m, spec in REGISTRY.items() if any(p.concept == concept for p in spec.params)
    )


def linear_data(seed: int, *, d: int = 8, n: int = 600) -> np.ndarray:
    """A dense linear model: each variable depends on most earlier ones, so sets grow large."""
    rng = np.random.default_rng(seed)
    weights = np.triu(rng.uniform(0.5, 1.0, (d, d)) * (rng.random((d, d)) < 0.7), 1)
    return rng.standard_normal((n, d)) @ np.linalg.inv(np.eye(d) - weights)


def series(seed: int, *, n_vars: int = 5, steps: int = 400) -> np.ndarray:
    """A lagged linear process in which every variable drives every other one step later."""
    rng = np.random.default_rng(seed)
    x = np.zeros((steps, n_vars))
    for t in range(1, steps):
        x[t] = 0.4 * x[t - 1] + 0.1 * x[t - 1].sum() + rng.standard_normal(n_vars)
    return x


# --- max_depth: recorded conditioning-set sizes -----------------------------------------------


def ci_search_sizes(method: str, cap: int | None, monkeypatch) -> list[int]:
    """Every test of a CI search, through a registered Fisher-z test that only defines ``__call__``,
    so no batched route can bypass the record."""
    sizes: list[int] = []

    class Recorded:
        def __init__(self, data, **_):
            self._test = ci.FisherZ(data)

        def __call__(self, x, y, condition_set=()):
            condition_set = list(condition_set)
            sizes.append(len(condition_set))
            return self._test(x, y, condition_set)

    monkeypatch.setitem(ci._CI_REGISTRY, "recorded", Recorded)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(andrey, method)(*inputs(method), indep_test="recorded", max_depth=cap)
    return sizes


def inputs(method: str) -> list:
    """The data, and a two-context index for a method that also requires one (cdnod)."""
    data = linear_data(0)
    extra = [inp for inp in REGISTRY[method].data.inputs[1:] if inp.required]
    return [data] + [np.repeat([0.0, 1.0], len(data) // 2).reshape(-1, 1) for _ in extra]


def pc1_sizes(method: str, cap: int | None, monkeypatch) -> list[int]:
    """Every test of the PC1 step, the step ``max_depth`` caps; MCI and the contemporaneous phases
    condition on the parents of both ends as well."""
    sizes: list[int] = []
    original = pcmci_engine.pc1

    class Recorded:
        def __init__(self, test):
            self._test = test

        def statistic_pvalues(self, x, y, conditions):
            sizes.extend([np.shape(conditions)[1]] * len(np.atleast_1d(x)))
            return self._test.statistic_pvalues(x, y, conditions)

    def pc1(test, *args, **kwargs):
        return original(Recorded(test), *args, **kwargs)

    monkeypatch.setattr(pcmci_engine, "pc1", pc1)
    monkeypatch.setattr(pcmciplus_engine, "pc1", pc1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(andrey, method)(series(0), tau_max=2, max_depth=cap)
    return sizes


def rcd_sizes(method: str, cap: int | None, monkeypatch) -> list[int]:
    """Every ancestor test of RCD: the variables regressed out together."""
    sizes: list[int] = []
    original = rcd_engine._is_independent_of_residual

    def recorded(Y, xi, exog, *args):
        sizes.append(len(exog))
        return original(Y, xi, exog, *args)

    monkeypatch.setattr(rcd_engine, "_is_independent_of_residual", recorded)
    # Two hidden common causes of every variable: no ancestor test succeeds, so the sets grow.
    rng = np.random.default_rng(0)
    hidden = rng.exponential(size=(500, 2)) - 1
    data = hidden @ rng.uniform(0.5, 1.0, (2, 5)) + 0.3 * (rng.exponential(size=(500, 5)) - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        andrey.rcd(data, max_depth=data.shape[1] if cap is None else max(cap, 1))
    return sizes


def camuv_sizes(method: str, cap: int | None, monkeypatch) -> list[int]:
    """Every sink test of CAM-UV: the other members of the set, tested together.

    The tests are stubbed to find every pair dependent and no sink, so the search tries every set
    of every size the cap allows.
    """
    sizes: list[int] = []

    def sink_test(self, child, regressors, y):
        sizes.append(len(y))
        return -1e6

    monkeypatch.setattr(camuv_engine._Tests, "log_p_fresh", sink_test)
    monkeypatch.setattr(camuv_engine._Tests, "log_p", lambda self, x, y: -1e6)
    data = np.random.default_rng(0).standard_normal((200, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        andrey.camuv(data, max_depth=data.shape[1] if cap is None else max(cap, 1))
    return sizes


RECORDERS = {"pcmci": pc1_sizes, "pcmciplus": pc1_sizes, "rcd": rcd_sizes, "camuv": camuv_sizes}


@pytest.mark.parametrize("method", methods_with("max_depth"))
def test_max_depth_caps_every_recorded_conditioning_set(method, monkeypatch):
    record = RECORDERS.get(method, ci_search_sizes)
    assert max(record(method, None, monkeypatch)) > max(CAPS)  # the data reach past every cap
    lowest = next(p.minimum for p in REGISTRY[method].params if p.concept == "max_depth")
    for cap in CAPS:
        if lowest is not None and cap < lowest:
            continue
        sizes = record(method, cap, monkeypatch)
        assert sizes and max(sizes) <= cap, (method, cap, max(sizes))


# --- max_p_global: recorded sizes of the searched subsets ---------------------------------------


def same_step_series(seed: int, *, n_vars: int = 6, steps: int = 500) -> np.ndarray:
    """A lagged process whose noise is mixed along a dense same-step graph, so the same-step
    neighbor sets stay large."""
    rng = np.random.default_rng(seed)
    weights = np.tril(rng.uniform(0.5, 1.0, (n_vars, n_vars)), -1)
    mixing = np.linalg.inv(np.eye(n_vars) - weights)
    x = np.zeros((steps, n_vars))
    for t in range(1, steps):
        x[t] = 0.3 * x[t - 1] + mixing @ rng.standard_normal(n_vars)
    return x


def subset_sizes(method: str, cap: int | None, monkeypatch) -> list[int]:
    """The size of every subset of same-step neighbors a test searches, in whichever phase."""
    sizes: list[int] = []
    if method == "pcmciplus":
        original = pcmciplus_engine._test_subsets

        def tested(test, n_vars, parents, i, lag, j, subsets):
            sizes.append(len(subsets[0]))
            return original(test, n_vars, parents, i, lag, j, subsets)

        monkeypatch.setattr(pcmciplus_engine, "_test_subsets", tested)
    else:
        search = lpcmci_engine._Search.separating_subsets

        def searched(self, a, b, found, sdef, p):
            sizes.append(p)
            return search(self, a, b, found, sdef, p)

        monkeypatch.setattr(lpcmci_engine._Search, "separating_subsets", searched)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(andrey, method)(same_step_series(0), tau_max=1, max_p_global=cap)
    return sizes


@pytest.mark.parametrize("method", methods_with("max_p_global"))
def test_max_p_global_caps_every_searched_subset(method, monkeypatch):
    assert max(subset_sizes(method, None, monkeypatch)) > max(CAPS)  # the data reach past the caps
    for cap in CAPS:
        sizes = subset_sizes(method, cap, monkeypatch)
        assert sizes and max(sizes) <= cap, (method, cap, max(sizes))


def test_a_max_p_global_above_every_search_returns_the_uncapped_pcmciplus_graph():
    data = same_step_series(1)
    uncapped = andrey.pcmciplus(data, tau_max=1)
    capped = andrey.pcmciplus(data, tau_max=1, max_p_global=50)
    for lag in (0, 1):
        assert (
            capped.structure.lag(lag).oriented_edges()
            == uncapped.structure.lag(lag).oriented_edges()
        )
    assert capped.metadata["p_values"] == uncapped.metadata["p_values"]
    assert capped.metadata["ambiguous_triples"] == uncapped.metadata["ambiguous_triples"]


# --- max_parents: the returned graph -----------------------------------------------------------


def capped_graph(method: str, monkeypatch):
    """A function that fits ``method`` and returns the marks of the graph its cap bounds: gfci's
    GES CPDAG, recorded before the tests prune and orient it, else the returned graph."""
    fit = getattr(andrey, method)
    if method != "gfci":
        return lambda data, **kw: fit(data, **kw).structure.to_numpy()
    found = []
    ges = gfci_engine.ges

    def recorded(*args, **kwargs):
        out = ges(*args, **kwargs)
        found.append(out[0].to_numpy())
        return out

    monkeypatch.setattr(gfci_engine, "ges", recorded)

    def graph(data, **kw):
        found.clear()
        fit(data, **kw)
        (marks,) = found
        return marks

    return graph


@pytest.mark.parametrize("method", methods_with("max_parents"))
def test_max_parents_caps_every_dag_the_result_represents(method, monkeypatch):
    data = linear_data(1, d=7, n=500)
    graph = capped_graph(method, monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uncapped = max_parents_in_class(graph(data)).max()
        assert uncapped > max(CAPS)
        for cap in CAPS:
            marks = graph(data, max_parents=cap)
            assert max_parents_in_class(marks).max() <= cap, (method, cap)


@pytest.mark.parametrize("method", ["gies", "xges", "gfci"])
def test_max_parents_holds_from_a_start_of_required_edges(method, monkeypatch):
    """The equivalence-class searches start from the class of the required edges; the cap holds
    there and after every move, and the required edges stay."""
    data = pd.DataFrame(linear_data(3, d=7, n=500), columns=[f"x{i}" for i in range(7)])
    knowledge = andrey.Knowledge(required=[("x0", "x6"), ("x5", "x6")], forbidden=[("x6", "x1")])
    graph = capped_graph(method, monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cap in (2, 3):
            marks = graph(data, max_parents=cap, knowledge=knowledge)
            assert max_parents_in_class(marks).max() <= cap, (method, cap)
            assert marks[0, 6] != 0 and marks[5, 6] != 0


def test_max_parents_caps_the_interventional_essential_graph():
    data = linear_data(4, d=7, n=600)
    interventions = np.zeros(data.shape, dtype=bool)
    interventions[400:, 2] = True
    data[400:, 2] = np.random.default_rng(4).standard_normal(200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uncapped = andrey.gies(data, interventions).structure.to_numpy()
        assert max_parents_in_class(uncapped).max() > max(CAPS)
        for cap in CAPS:
            marks = andrey.gies(data, interventions, max_parents=cap).structure.to_numpy()
            assert max_parents_in_class(marks).max() <= cap, cap


@pytest.mark.parametrize("method", ["gies", "xges"])
def test_a_cap_no_graph_can_reach_returns_the_uncapped_graph(method):
    data = linear_data(1, d=7, n=500)
    fit = getattr(andrey, method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert fit(data, max_parents=6).structure == fit(data).structure


@pytest.mark.parametrize("method", methods_with("max_parents"))
def test_required_edges_above_the_parent_cap_are_refused(method):
    data = pd.DataFrame(linear_data(2, d=4, n=200), columns=["x0", "x1", "x2", "x3"])
    knowledge = andrey.Knowledge(required=[("x0", "x3"), ("x1", "x3")])
    with pytest.raises(ValueError, match="max_parents"):
        getattr(andrey, method)(data, max_parents=1, knowledge=knowledge)


# --- iteration caps --------------------------------------------------------------------------


def test_a_varma_fit_that_exhausts_the_evaluation_limit_warns_as_a_search_limit(monkeypatch):
    """The optimizer's evaluation limit grows with ``max_iter``; a fit that ends on it is a cap."""
    monkeypatch.setattr(varma_engine, "_EVALUATIONS_PER_ITERATION", 1)
    warning_policy._warned.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        andrey.varma_lingam(series(0, n_vars=3), max_iter=100)
    assert andrey.SearchLimitWarning in {w.category for w in caught}


def test_a_scikit_learn_convergence_warning_is_raised_again_as_a_search_limit():
    warning_policy._warned.clear()

    def fit():
        warnings.warn("did not converge", ConvergenceWarning, stacklevel=1)
        warnings.warn("kept as it is", RuntimeWarning, stacklevel=1)
        return 7

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert warning_policy.convergence_as_search_limit("source", fit) == 7
    kinds = {w.category: str(w.message) for w in caught}
    assert ConvergenceWarning not in kinds
    assert kinds[andrey.SearchLimitWarning] == "source: did not converge"
    assert kinds[RuntimeWarning] == "kept as it is"
