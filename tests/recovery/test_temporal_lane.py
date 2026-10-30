from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from andrey.lingam.var import var_lingam
from andrey.temporal.granger import granger_f_test
from tests.recovery.helpers import datagen, harness
from tests.recovery.helpers.slice_fixture import assert_recorded_output


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


def _panel(seed: int, t_points: int = 4, n: int = 600, p: int = 4) -> list[np.ndarray]:
    """A stationary longitudinal non-Gaussian SEM: instantaneous DAG plus a lag-1 effect."""
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(p))
    b_inst = np.zeros((p, p))
    for a in range(p):
        for b in range(a):
            if rng.random() < 0.5:
                b_inst[order[a], order[b]] = rng.uniform(0.4, 0.9) * rng.choice([-1.0, 1.0])
    b_lag = np.zeros((p, p))
    mask = rng.random((p, p)) < 0.4
    b_lag[mask] = rng.uniform(0.3, 0.7, mask.sum()) * rng.choice([-1.0, 1.0], mask.sum())
    inv = np.linalg.inv(np.eye(p) - b_inst)

    def noise() -> np.ndarray:
        u = rng.uniform(size=(n, p))
        e = np.sign(u - 0.5) * np.abs(u - 0.5) ** 0.7
        return (e - e.mean(0)) / e.std(0)

    panel = []
    x_prev = inv @ noise().T
    panel.append(x_prev.T.copy())
    for _ in range(1, t_points):
        x_cur = inv @ (b_lag @ x_prev + noise().T)
        panel.append(x_cur.T.copy())
        x_prev = x_cur
    return panel


def test_varma_lingam_baseline():
    from andrey.temporal.varma import varma_lingam

    order, psis, omegas = varma_lingam(datagen.generate("varma_3v"), order=(1, 1), prune=False)
    harness.assert_baseline(
        "VARMALiNGAM",
        {"causal_order": list(order), "psis": np.asarray(psis), "omegas": np.asarray(omegas)},
        case="varma_3v",
    )


def test_varma_lingam_pruned_baseline():
    from andrey.temporal.varma import varma_lingam

    order, psis, omegas = varma_lingam(datagen.generate("varma_3v"), order=(1, 1), prune=True)
    harness.assert_baseline(
        "VARMALiNGAMPruned",
        {"causal_order": list(order), "psis": np.asarray(psis), "omegas": np.asarray(omegas)},
        case="varma_3v",
    )


def test_varma_lingam_output_contract():
    from andrey.api.temporal import varma_lingam

    out = varma_lingam(datagen.generate("varma_3v"), order=(1, 1))
    assert out.structure.type == "temporal"
    assert out.structure.lag_weights.shape == (2, 3, 3)
    assert out.structure.lag_weights_ma.shape == (1, 3, 3)


def test_varma_lingam_rejects_max_iter_below_one():
    from andrey.api.temporal import varma_lingam

    with pytest.raises(ValueError, match="max_iter"):
        varma_lingam(datagen.generate("varma_3v"), max_iter=0)


def test_varma_lingam_orders_innovations_by_least_squares():
    """The order comes from least-squares pairwise regressions of the fitted innovations.

    On these 40 steps a slope scaled by ``n / (n - 1)`` orders the innovations [1, 2, 0]; least
    squares gives the true order, [0, 1, 2].
    """
    from statsmodels.tsa.statespace.varmax import VARMAX

    from andrey.lingam import pwling
    from andrey.temporal.varma import varma_lingam

    rng = np.random.default_rng(80)
    n, steps = 3, 90
    inv = np.linalg.inv(np.eye(n) - np.tril(np.full((n, n), 0.4), -1))
    e = rng.uniform(-1.0, 1.0, (steps, n))
    x = np.zeros((steps, n))
    for t in range(1, steps):
        x[t] = inv @ (0.3 * x[t - 1] + e[t] + 0.2 * e[t - 1])
    X = x[50:]

    order, _, _ = varma_lingam(X, order=(1, 1), prune=True, max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = VARMAX(X, order=(1, 1), trend="c").fit(maxiter=2000, disp=False, cov_type="none")
    innovations = np.asarray(fit.resid)
    assert order == pwling.find_causal_order(innovations) == [0, 1, 2]


def test_longitudinal_lingam_output_contract():
    from andrey.api.temporal import longitudinal_lingam

    out = longitudinal_lingam(_panel(0), tau_max=1)
    assert out.structure.type == "temporal"
    assert out.structure.lag_weights.shape == (2, 4, 4)
    assert out.metadata["algorithm"] == "longitudinal_lingam"


def test_longitudinal_lingam_matches_the_recorded_output():
    from andrey.api.temporal import longitudinal_lingam

    out = longitudinal_lingam(datagen.generate_groups("panel_4v"))
    assert_recorded_output(out, "LongitudinalLiNGAM", case="panel_4v")


def test_var_lingam_recovery():
    from andrey.api._temporal import _adapt_var_lingam

    order, adj = var_lingam(datagen.generate("var_4v_stable"))
    model = SimpleNamespace(adjacency_matrices_=adj, causal_order_=order)
    out = _adapt_var_lingam(model)
    assert_recorded_output(out, "VARLiNGAM", case="var_4v_stable")


def test_granger_recovery():
    coeff, _ = granger_f_test(datagen.generate("var_4v_stable"), maxlag=2)
    harness.assert_baseline("Granger", {"coeff": coeff, "adj": coeff != 0}, case="var_4v_stable")


def test_var_lingam_output_contract():
    order, adj = var_lingam(datagen.generate("var_4v_stable"))
    # The planted instantaneous order is x0, x1, x2, x3.
    assert list(order) == datagen.column_of("var_4v_stable").tolist()
    assert np.asarray(adj).shape == (2, 4, 4)


def test_granger_output_contract():
    coeff, p_values = granger_f_test(datagen.generate("var_4v_stable"), maxlag=2)
    assert coeff.shape == p_values.shape == (4, 8)


@pytest.fixture(scope="module")
def var_lag2_fit():
    from andrey.api.temporal import granger

    return granger(datagen.generate("var_3v_lag2"), tau_max=2)


def _var_lag2_truth() -> np.ndarray:
    """The planted lag weights of ``var_3v_lag2`` as ``[lag - 1, cause, effect]``, by column."""
    order = datagen.column_order("var_3v_lag2")
    lags = np.asarray(datagen.CASES["var_3v_lag2"]["lags"]).transpose(0, 2, 1)
    return lags[:, order][:, :, order]


def test_granger_finds_the_planted_lagged_edges_as_the_largest_weights(var_lag2_fit):
    structure, truth = var_lag2_fit.structure, _var_lag2_truth()
    assert structure.type == "temporal" and structure.lags == (1, 2)
    assert var_lag2_fit.ordering is None and var_lag2_fit.metadata["algorithm"] == "granger"
    weights = structure.lag_weights
    for k in structure.lags:
        planted = {(int(i), int(j)) for i, j in zip(*np.nonzero(truth[k - 1]), strict=True)}
        found = {(i, j) for i, j, _ in structure.lag(k).oriented_edges()}
        assert planted <= found, f"lag {k} misses {planted - found}"
    # Measured: 0.076 off the planted weights at most, and 0.044 at most elsewhere.
    assert np.abs(weights - truth)[truth != 0].max() < 0.15
    assert np.abs(weights[truth == 0]).max() < 0.1


def test_granger_keeps_absent_edges_absent_when_units_are_far_apart():
    from andrey.api.temporal import granger

    # Columns in units 1e-200 and 1e200 with no lag-one dependence: the unit ratio overflows.
    x = np.column_stack(
        (
            1e-200 * np.resize([0.0, 1.0, 0.0, -1.0], 17),
            1e200 * np.resize([0, 0, 1, 0, 0, 0, -1, 0.0], 17),
        )
    )
    weights = granger(x, tau_max=1).structure.lag_weights
    assert not np.isnan(weights).any()


# 1e-200: the squared deviations of that column would underflow to zero.
@pytest.mark.parametrize("scale", [[100.0, 0.01, 7.0], [1e-200, 1.0, 7.0]])
def test_granger_does_not_depend_on_the_units(var_lag2_fit, scale):
    from andrey.api.temporal import granger

    scale = np.array(scale)
    out = granger(datagen.generate("var_3v_lag2") * scale, tau_max=2)
    weights = var_lag2_fit.structure.lag_weights
    # A weight of x_i on x_j scales with the units of x_j over those of x_i.
    np.testing.assert_allclose(
        out.structure.lag_weights, weights * scale[None, None, :] / scale[None, :, None], rtol=1e-7
    )
    assert np.array_equal(out.structure.lag_weights != 0, weights != 0)
    assert np.array_equal(
        granger(datagen.generate("var_3v_lag2"), tau_max=2).structure.lag_weights, weights
    )


def test_granger_keeps_only_small_weights_between_independent_series():
    from andrey.api.temporal import granger

    noise = np.random.default_rng(1).standard_normal((1000, 3))
    assert np.abs(granger(noise).structure.lag_weights).max() < 0.1  # measured 0.043
