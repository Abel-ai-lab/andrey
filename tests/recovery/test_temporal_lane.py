from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from andrey.lingam.var import var_lingam
from andrey.temporal.granger import granger_lasso
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


def test_varma_lingam_output_contract():
    from andrey.api.temporal import varma_lingam

    out = varma_lingam(datagen.generate("varma_3v"), order=(1, 1))
    assert out.structure.type == "temporal"
    assert out.structure.lag_weights.shape == (2, 3, 3)
    assert out.structure.lag_weights_ma.shape == (1, 3, 3)


def test_longitudinal_lingam_output_contract():
    from andrey.api.temporal import longitudinal_lingam

    out = longitudinal_lingam(_panel(0), n_lags=1)
    assert out.structure.type == "temporal"
    assert out.structure.lag_weights.shape == (2, 4, 4)
    assert out.metadata["algorithm"] == "longitudinallingam"


def test_var_lingam_recovery():
    from andrey.api._temporal import _adapt_var_lingam

    order, adj = var_lingam(datagen.generate("var_4v_stable"))
    model = SimpleNamespace(adjacency_matrices_=adj, causal_order_=order)
    out = _adapt_var_lingam(model)
    assert_recorded_output(out, "VARLiNGAM", case="var_4v_stable")


def test_granger_recovery():
    coeff = granger_lasso(datagen.generate("var_4v_stable"))
    harness.assert_baseline("Granger", {"coeff": coeff, "adj": coeff != 0}, case="var_4v_stable")


def test_var_lingam_output_contract():
    order, adj = var_lingam(datagen.generate("var_4v_stable"))
    assert list(order) == [0, 1, 2, 3] and np.asarray(adj).shape == (2, 4, 4)


def test_granger_output_contract():
    coeff = granger_lasso(datagen.generate("var_4v_stable"))
    assert coeff.shape == (4, 8)
