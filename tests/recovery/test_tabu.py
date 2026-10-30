"""Recovery and behavior checks for tabu search.

Tabu search takes hill climbing's moves until hill climbing stops, then keeps searching and returns
the best DAG it visited. So on the same data and settings, its BIC score lies between the global
optimum (:func:`andrey.exact_search`) and hill climbing's. The seeded Erdos-Renyi cases are
linear-Gaussian, where BIC is consistent; the pinned seeds are ones where hill climbing stops at a
local optimum and tabu search reaches the true equivalence class.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import andrey
from andrey import data
from andrey.core import backend
from andrey.core.score import BICScore
from andrey.metrics import markov_equivalent, to_cpdag
from tests.recovery.helpers import datagen
from tests.recovery.helpers.project import found_dag
from tests.recovery.helpers.slice_fixture import assert_recorded_output

# The extended-BIC weight that tabu search and exact search score with by default.
_EBIC_GAMMA = 1.0


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
    score = BICScore(x, ebic_gamma=_EBIC_GAMMA)
    return float(sum(score.score(i, sorted(parents[i])) for i in range(x.shape[1])))


def test_tabu_reaches_the_score_optimum_on_gauss_5v():
    """Tabu search returns the CPDAG of the best-scoring DAG, which exact search finds, and that
    DAG's score. The graph and score match the recorded output."""
    x = datagen.generate("gauss_5v")
    out = andrey.tabu(x)
    optimum = andrey.exact_search(x)
    assert out.structure.kind == "cpdag"
    assert out.structure.oriented_edges() == optimum.structure.oriented_edges()
    assert math.isclose(
        out.metadata["score"], _bic(x, found_dag(optimum)), rel_tol=0.0, abs_tol=1e-6
    )
    assert_recorded_output(out, "Tabu", case="gauss_5v")


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
    optimum = _bic(x, found_dag(andrey.exact_search(x)))
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


def test_tabu_is_deterministic_across_runs_and_worker_counts():
    """Repeated fits, and fits under different worker counts, agree exactly, non-improving moves
    included."""
    x = _er_sample(10, 9).data
    first = andrey.tabu(x)
    for workers in (1, 2, 3):
        with backend.config(n_jobs=workers):
            again = andrey.tabu(x)
        assert np.array_equal(again.structure.to_numpy(), first.structure.to_numpy())
        assert again.metadata["score"] == first.metadata["score"]


def test_tabu_sets_no_move_limit_by_default():
    # 200 variables need over 400 moves: a limit of 200 stops the search, the default does not.
    x = data.sample_scm(d=200, n=2000, seed=0, density=2.0, scale="raw").data
    with pytest.warns(andrey.SearchLimitWarning):
        andrey.tabu(x, max_iter=200)
    with warnings.catch_warnings():
        warnings.simplefilter("error", andrey.SearchLimitWarning)
        andrey.tabu(x)
