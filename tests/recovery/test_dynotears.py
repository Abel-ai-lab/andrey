"""DYNOTEARS recovery: planted same-step and lagged links, generated truth, and the null cases.

The planted series has unit-variance Gaussian noise, the case in which the paper shows the model
identifiable, and stays in the data's units, the setting the method claims.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey.data import functional, graphs, noise, temporal
from andrey.data.scm import SCM
from andrey.metrics import temporal_scores
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output

# W[cause, effect]: X0 -> X1 -> X2 <- X3 within each step.
W = np.zeros((4, 4))
W[0, 1], W[1, 2], W[3, 2] = 1.0, -0.8, 0.7
# A_k[cause, effect]: every series depends on its own past; X1 -> X3 at lag 1, X2 -> X0 at lag 2.
LAG_1 = np.diag([0.5, 0.4, 0.3, 0.4])
LAG_1[1, 3] = 0.5
LAG_2 = np.zeros((4, 4))
LAG_2[2, 0] = -0.4
SAME_STEP = {(0, 1), (1, 2), (3, 2)}
LAGGED = {(i, i, 1) for i in range(4)} | {(1, 3, 1), (2, 0, 2)}


def _fit(data: np.ndarray, **kwargs) -> andrey.StructureOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return andrey.dynotears(data, **kwargs)


def _same_step(out: andrey.StructureOutput) -> set[tuple[int, int]]:
    return {(i, j) for i, j, _ in out.structure.lag(0).oriented_edges()}


def _lagged(out: andrey.StructureOutput) -> set[tuple[int, int, int]]:
    s = out.structure
    return {(i, j, lag) for lag in s.lags if lag > 0 for i, j, _ in s.lag(lag).oriented_edges()}


@pytest.fixture(scope="module")
def planted() -> np.ndarray:
    """2,000 steps of the SVAR above; ``generate`` takes the matrices as ``[effect, cause]``."""
    innovations = np.random.default_rng(0).standard_normal((2200, 4))
    ib_inv = np.linalg.inv(np.eye(4) - W.T)
    return temporal.generate(ib_inv, np.stack([LAG_1.T, LAG_2.T]), None, innovations, burn_in=200)


@pytest.fixture(scope="module")
def planted_fit(planted) -> andrey.StructureOutput:
    return _fit(planted, tau_max=2)


# ---- fixed cases ---------------------------------------------------------------------------------


def test_recovers_the_planted_links_and_their_weights(planted_fit):
    assert _same_step(planted_fit) == SAME_STEP
    assert _lagged(planted_fit) == LAGGED
    truth = np.stack([W, LAG_1, LAG_2])
    # The l1 penalty shrinks each weight toward 0; measured: 0.058 at most.
    np.testing.assert_allclose(planted_fit.structure.lag_weights, truth, atol=0.15)
    assert planted_fit.metadata["h_final"] <= 1e-8
    assert planted_fit.metadata["edges_removed"] == 0


def test_independent_series_give_no_links():
    out = _fit(np.random.default_rng(1).standard_normal((1000, 4)))
    assert _same_step(out) == set() and _lagged(out) == set()
    assert not out.structure.lag_weights.any()


def test_independent_autocorrelated_series_give_only_self_loops():
    innovations = np.random.default_rng(2).standard_normal((1200, 4))
    data = temporal.generate(np.eye(4), 0.7 * np.eye(4)[None], None, innovations, burn_in=200)
    out = _fit(data)
    assert _same_step(out) == set()
    assert _lagged(out) == {(i, i, 1) for i in range(4)}


# ---- generated truth -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 4])
def test_generated_svar_scores_well_above_the_empty_graph(seed):
    """Linear Gaussian VAR series in the data's units from ``andrey.data``: measured F1 of 1.0 on
    the same-step and the lagged links at each seed; the empty graph scores 0."""
    scm = SCM(
        graph=graphs.erdos_renyi(6, 1.0),
        functional=functional.var(n_lags=1, lag_density=0.2),
        noise=noise.gaussian(),
    )
    dataset = scm.sample(1000, seed, scale="raw")
    scores = temporal_scores(_fit(dataset.data), dataset.graph)
    assert scores["contemporaneous"]["f1"] >= 0.9
    assert scores["lagged"]["f1"] >= 0.85
    assert scores["summary"]["f1"] >= 0.85


# ---- what the method promises --------------------------------------------------------------------


def test_same_result_on_every_run(planted, planted_fit):
    again = _fit(planted, tau_max=2)
    assert again.structure == planted_fit.structure
    np.testing.assert_array_equal(again.structure.lag_weights, planted_fit.structure.lag_weights)
    assert again.metadata == planted_fit.metadata


def test_reordering_the_columns_only_reorders_the_result(planted, planted_fit):
    """The columns are fit in the order of their ranks, so the order they arrive in is lost."""
    order = [2, 0, 3, 1]
    moved = _fit(planted[:, order], tau_max=2)
    back = {(order[i], order[j]) for i, j in _same_step(moved)}
    assert back == _same_step(planted_fit)
    assert {(order[i], order[j], k) for i, j, k in _lagged(moved)} == _lagged(planted_fit)
    np.testing.assert_allclose(
        moved.structure.lag_weights,
        planted_fit.structure.lag_weights[:, order][:, :, order],
        atol=1e-6,
    )


def test_an_offset_on_every_column_leaves_the_fit_unchanged(planted, planted_fit):
    """Each column is centered first, so a constant added to a series changes no link and moves
    the weights only within the solver's precision."""
    moved = _fit(planted + [5.0, -1e3, 1e6, 0.0], tau_max=2)
    assert _same_step(moved) == _same_step(planted_fit)
    assert _lagged(moved) == _lagged(planted_fit)
    np.testing.assert_allclose(
        moved.structure.lag_weights, planted_fit.structure.lag_weights, rtol=0.0, atol=5e-3
    )


def test_matches_the_recorded_output():
    out = _fit(datagen.generate("svar_4v_gauss"))
    assert_recorded_output(out, "DYNOTEARS", case="svar_4v_gauss")
