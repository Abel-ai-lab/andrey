"""Recovery and behavior checks for tabu search.

Tabu search takes hill climbing's moves until hill climbing stops, then keeps searching and returns
the best DAG it visited. So on the same data and settings, its BIC score lies between the global
optimum (:func:`andrey.exact_search`) and hill climbing's. The seeded Erdos-Renyi cases are
linear-Gaussian, where BIC is consistent; the pinned seeds are ones where hill climbing stops at a
local optimum and tabu search reaches the true equivalence class.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey import data
from andrey.core import backend
from andrey.core.score import BICScore
from andrey.metrics import markov_equivalent, to_cpdag
from andrey.search import _parallel_hc as phc
from tests.recovery.helpers import datagen

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


def _er_sample(d: int, seed: int):
    """A standardized linear-Gaussian sample on an Erdos-Renyi DAG of average degree 2."""
    n = 1000 if d <= 8 else 2000
    return data.sample_scm(
        graph="erdos_renyi", d=d, n=n, seed=seed, density=2.0, scale="standardize"
    )


def _bic(x: np.ndarray, dag) -> float:
    """The total BIC of a DAG, summed as :func:`andrey.tabu` reports it."""
    parents: dict[int, list[int]] = {i: [] for i in range(x.shape[1])}
    for i, j, kind in dag.oriented_edges():
        assert kind == "directed"
        parents[j].append(i)
    score = BICScore(x)
    return float(sum(score.score(i, sorted(parents[i])) for i in range(x.shape[1])))


def test_tabu_recovers_gauss_5v():
    """The CPDAG of ``gauss_5v``: 0 -> 1, 0 -> 2, 1 -> 3 <- 2, 3 -> 4, with the collider at 3."""
    out = andrey.tabu(datagen.generate("gauss_5v"))
    assert out.structure.kind == "cpdag"
    assert out.structure.oriented_edges() == [
        (0, 1, "undirected"),
        (0, 2, "undirected"),
        (1, 3, "directed"),
        (2, 3, "directed"),
        (3, 4, "directed"),
    ]


def test_tabu_finds_no_edges_between_independent_columns():
    """Every move from the empty graph worsens the score, so the best DAG visited is empty."""
    x = np.random.default_rng(0).standard_normal((500, 6))
    out = andrey.tabu(x)
    assert out.structure.oriented_edges() == []
    assert out.metadata["score"] == _bic(x, out.structure)


@pytest.mark.parametrize("seed", range(12))
def test_tabu_score_lies_between_the_optimum_and_hill_climbing(seed):
    x = _er_sample(8, seed).data
    found = andrey.tabu(x).metadata["score"]
    optimum = _bic(x, andrey.exact_search(x).structure)
    assert optimum <= found + 1e-6 * abs(optimum)
    assert found <= andrey.hc(x).metadata["score"]


@pytest.mark.parametrize("seed", [4, 9, 10])
def test_tabu_recovers_the_class_where_hill_climbing_stops_short(seed):
    sample = _er_sample(10, seed)
    truth = to_cpdag(sample.graph)
    assert not markov_equivalent(andrey.hc(sample.data).structure, truth)
    assert markov_equivalent(andrey.tabu(sample.data).structure, truth)


def test_longer_searches_never_score_worse():
    """A larger ``max_tabu`` or ``max_iter`` follows the same path further, keeping the best."""
    x = _er_sample(10, 9).data
    by_max_tabu = [andrey.tabu(x, max_tabu=k).metadata["score"] for k in range(1, 11)]
    assert by_max_tabu == sorted(by_max_tabu, reverse=True)
    with pytest.warns(andrey.SearchLimitWarning):
        short = andrey.tabu(x, max_iter=12).metadata["score"]
    assert andrey.tabu(x).metadata["score"] <= short


def test_tabu_is_deterministic_across_runs_and_worker_counts(monkeypatch):
    """The worker pool reproduces the in-process search exactly, non-improving moves included."""
    x = _er_sample(10, 9).data
    serial = andrey.tabu(x)
    assert np.array_equal(andrey.tabu(x).structure.to_numpy(), serial.structure.to_numpy())
    pools = []
    create_pool = phc._create_pool

    def counting(*args, **kwargs):
        pools.append(1)
        return create_pool(*args, **kwargs)

    monkeypatch.setattr(phc, "_create_pool", counting)
    monkeypatch.setenv("ANDREY_HC_PARALLEL_MIN_WORK", "0")  # every pass fans out to the workers
    for workers in (2, 3):
        with backend.config(num_workers=workers):
            pooled = andrey.tabu(x)
        assert np.array_equal(pooled.structure.to_numpy(), serial.structure.to_numpy())
        assert pooled.metadata["score"] == serial.metadata["score"]
    assert len(pools) == 2  # one worker set per fit, kept across its passes


def test_tabu_sets_no_move_limit_by_default():
    # 200 variables need over 400 moves: a limit of 200 stops the search, the default does not.
    x = data.sample_scm(d=200, n=2000, seed=0, density=2.0).data
    with pytest.warns(andrey.SearchLimitWarning):
        andrey.tabu(x, max_iter=200)
    with warnings.catch_warnings():
        warnings.simplefilter("error", andrey.SearchLimitWarning)
        andrey.tabu(x)
