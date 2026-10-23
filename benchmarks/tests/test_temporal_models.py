"""The paper simulations: each draws the links its paper describes, at the same density at every d.

Each test checks a property of the drawn model against the paper's setup as
``andrey_bench.temporal_models`` records it: the number of cross-links per variable, the share
within a time step, the lag range, one autodependency per variable, an acyclic same-step graph, a
stationary series, and the unobserved share. The stored truth and the key's identity are checked
through the dataset store, as a campaign reads them. A model none of whose draws is stationary is
rescaled to be, and a model whose draws are keeps them exactly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import zlib

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


#: SHA-256 of each model's coefficient stacks, marks, and series over ``d`` in (10, 40, 160) and
#: seeds 0 to 2, where a stationary draw comes within ``MAX_DRAWS``. A change to any drawn value
#: changes it, and with it the datasets a campaign has stored.
DRAWS = {
    "lpcmci2020": "2657a1f494ca4e84481baf473f6ba0f97265b529db52489ad2a14c2aa8b6c140",
    "pcmci2019": "646386ff9ec1116131205bd28b2c7c253f0f97d3cee52851bf450edc4517fd72",
    "pcmciplus2020": "b55800c6691f0917f60201fadb712df0dc7e6b57391f7004e0e9396f35e3e1f6",
}


def _stack(name: str, d: int, seed: int) -> np.ndarray:
    """The coefficient stack ``sample`` draws, from the stream ``sample`` seeds."""
    rng = np.random.default_rng([zlib.crc32(name.encode()), d, seed])
    return temporal_models._draw_model(temporal_models.MODELS[name], d, seed, rng)


def _digest(name: str) -> str:
    digest = hashlib.sha256()
    for d in (10, 40, 160):
        for seed in range(3):
            data, marks = temporal_models.sample(name, d, n=200, seed=seed)
            # The stack is random draws and exact; the series goes through BLAS, so it is rounded.
            for part in (_stack(name, d, seed), np.round(data, 5) + 0.0, marks):
                digest.update(np.ascontiguousarray(part).tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("name", MODELS)
def test_a_model_with_a_stationary_draw_keeps_its_draws(name: str) -> None:
    assert _digest(name) == DRAWS[name]


@pytest.mark.parametrize(
    ("name", "seed"), [("pcmci2019", 1), ("pcmciplus2020", 1), ("lpcmci2020", 2)]
)
def test_a_model_with_no_stationary_draw_is_rescaled_to_be_stationary(
    name: str, seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At d = 240 none of the 1000 draws of PCMCI+ at seed 1 is stationary, and checking them takes
    a minute; the first three draws stand in for them."""
    monkeypatch.setattr(temporal_models, "MAX_DRAWS", 3)
    first = []
    rescale = temporal_models._rescale

    def record(stack: np.ndarray) -> np.ndarray:
        first.append(stack)
        return rescale(stack)

    monkeypatch.setattr(temporal_models, "_rescale", record)
    d = 240
    stack = _stack(name, d, seed)
    (drawn,) = first
    assert temporal_models._spectral_radius(drawn[0], drawn[1:]) >= 1
    radius = temporal_models._spectral_radius(stack[0], stack[1:])
    assert radius == pytest.approx(temporal_models.STATIONARY_TARGET) and radius < 1
    # Only the lagged coefficients change: the same-step part and every link stay as drawn.
    np.testing.assert_array_equal(stack[0], drawn[0])
    np.testing.assert_array_equal(stack != 0, drawn != 0)
    assert not np.linalg.matrix_power((stack[0] != 0).astype(float), d).any()  # still a DAG
    model = temporal_models.MODELS[name]
    n_links = math.floor(model.cross_degree * d)
    assert np.count_nonzero(stack[0]) == round(model.same_step * n_links)
    assert np.count_nonzero(stack) - np.count_nonzero(np.diagonal(stack[1])) == n_links
    np.testing.assert_array_equal(stack, _stack(name, d, seed))  # the same for the same seed


def test_a_rescaled_series_is_stationary_and_the_same_for_the_same_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(temporal_models, "MAX_DRAWS", 3)
    data, marks = temporal_models.sample("pcmciplus2020", 240, n=2000, seed=1)
    again, marks_again = temporal_models.sample("pcmciplus2020", 240, n=2000, seed=1)
    np.testing.assert_array_equal(data, again)
    np.testing.assert_array_equal(marks, marks_again)
    assert np.isfinite(data).all()
    first, second = data[:1000].std(axis=0), data[1000:].std(axis=0)
    assert (second < 3 * first).all() and (first < 3 * second).all()
