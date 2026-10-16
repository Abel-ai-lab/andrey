"""The paper simulations: each draws the links its paper describes, at the same density at every d.

Each test checks a property of the drawn model against the paper's setup as
``andrey_bench.temporal_models`` records it: the number of cross-links per variable, the share
within a time step, the lag range, one autodependency per variable, an acyclic same-step graph, a
stationary series, and the unobserved share. The stored truth and the key's identity are checked
through the dataset store, as a campaign reads them.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from andrey_bench import datasets, temporal_models
from andrey_bench.integration import _truth
from andrey_bench.window import ARROW, TAIL, WindowGraph

MODELS = sorted(temporal_models.MODELS)


def _links(marks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The ``(lag, cause, effect)`` cross-links and the ``(lag, variable)`` autodependencies."""
    lagged = np.argwhere(marks[1:] == TAIL) + [1, 0, 0]
    same = np.argwhere(marks[0] == TAIL)
    same = np.column_stack([np.zeros(len(same), int), same])
    links = np.concatenate([same, lagged])
    cross = links[links[:, 1] != links[:, 2]]
    auto = links[links[:, 1] == links[:, 2]][:, :2]
    return cross, auto


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("d", [10, 40, 160])
def test_each_model_draws_its_papers_links_at_every_size(name: str, d: int) -> None:
    model = temporal_models.MODELS[name]
    _, marks = temporal_models.sample(name, d, n=200, seed=0)
    cross, auto = _links(marks)
    n_links = math.floor(model.cross_degree * d)
    assert len(cross) == n_links  # the same cross-links per variable at every d
    assert (cross[:, 0] == 0).sum() == round(model.same_step * n_links)
    lags = cross[cross[:, 0] > 0, 0]
    assert lags.min() >= model.lags[0] and lags.max() <= model.lags[1]
    assert set(auto[:, 0]) <= {1}  # autodependencies act at lag 1 only
    assert marks.shape == (model.max_lag + 1, d, d)


@pytest.mark.parametrize("name", MODELS)
def test_the_same_step_links_form_a_dag_with_an_arrowhead_at_each_effect(name: str) -> None:
    _, marks = temporal_models.sample(name, 40, n=200, seed=1)
    lag0 = marks[0]
    np.testing.assert_array_equal(lag0 == TAIL, (lag0 == ARROW).T)
    adjacency = (lag0 == TAIL).astype(int)
    # A DAG's adjacency matrix is nilpotent.
    assert not np.linalg.matrix_power(adjacency, len(adjacency)).any()


@pytest.mark.parametrize("name", MODELS)
def test_a_series_is_stationary_and_the_same_for_the_same_seed(name: str) -> None:
    data, marks = temporal_models.sample(name, 20, n=2000, seed=3)
    again, marks_again = temporal_models.sample(name, 20, n=2000, seed=3)
    np.testing.assert_array_equal(data, again)
    np.testing.assert_array_equal(marks, marks_again)
    assert np.isfinite(data).all()
    # A stationary series keeps its spread: the second half varies about as much as the first.
    first, second = data[:1000].std(axis=0), data[1000:].std(axis=0)
    assert (second < 3 * first).all() and (first < 3 * second).all()
    assert not np.array_equal(data, temporal_models.sample(name, 20, n=2000, seed=4)[0])


def test_the_pcmci_model_uses_one_coupling_strength_and_the_seed_halves_of_its_ensemble() -> None:
    model = temporal_models.MODELS["pcmci2019"]
    assert model.coefficient == (0.2, 0.2) and model.same_step == 0.0
    weak, strong = model.autocorrelation_sets
    assert max(weak) == 0.9 and min(strong) == 0.6


@pytest.mark.parametrize("d, hidden", [(10, 3), (15, 4), (20, 6)])
def test_the_lpcmci_model_leaves_thirty_percent_unobserved(d: int, hidden: int) -> None:
    assert temporal_models.hidden_count("lpcmci2020", d) == hidden
    data, marks = temporal_models.sample("lpcmci2020", d, n=300, seed=0)
    assert data.shape == (300, d - hidden)
    assert marks.shape[1:] == (d, d)  # the truth keeps the unobserved series, last


# --- the dataset store -------------------------------------------------------------------------


def _key(regime: str, d: int = 10, n: int = 300, seed: int = 0) -> datasets.DatasetKey:
    topology, functional, noise, standardize = datasets.REGIMES[regime]
    return datasets.DatasetKey(
        topology=topology,
        functional=functional,
        noise=noise,
        standardize=standardize,
        density=datasets.TEMPORAL_REGIMES[regime],
        d=d,
        n=n,
        seed=seed,
        latents=datasets.latents_for(regime, d),
    )


def test_a_paper_key_is_a_call_of_its_own_that_carries_the_whole_setup() -> None:
    key = _key("pcmciplus_paper")
    call = datasets.generation_kwargs(key)
    assert call["model"] == "pcmciplus2020"
    assert call["setup"]["cross_degree"] == key.density == 1.5
    assert key.label == "random-pcmciplus2020-gaussian-deg1.5-d10-n300-s0-raw"
    assert len({_key(r).digest for r in ("pcmci_paper", "pcmciplus_paper", "lpcmci_paper")}) == 3
    with pytest.raises(ValueError, match="density"):
        datasets.generation_kwargs(dataclasses.replace(key, density=2.0))


def test_each_paper_key_stores_the_truth_its_method_is_scored_against(tmp_path) -> None:
    keys = [_key("pcmci_paper"), _key("pcmciplus_paper"), _key("lpcmci_paper")]
    registry = datasets.materialize(keys, tmp_path)
    kinds = [registry["keys"][k.label]["graph_kind"] for k in keys]
    assert kinds == ["temporal", datasets.WINDOW_KIND, datasets.WINDOW_KIND]

    pcmci, plus, lp = (datasets.load(tmp_path, k) for k in keys)
    lagged = _truth(pcmci[1], "temporal")
    assert lagged.lags == (1, 2)  # lag-only: the PCMCI model has no same-step links
    window = _truth(plus[1], datasets.WINDOW_KIND)
    assert isinstance(window, WindowGraph) and window.latents == 0 and window.tau_max == 5
    hidden = _truth(lp[1], datasets.WINDOW_KIND, keys[2].latents)
    assert hidden.latents == 3 and lp[0].shape == (300, 7) and hidden.n_nodes == 10
