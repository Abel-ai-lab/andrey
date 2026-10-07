"""Check VAR/VARMA samples, temporal truth, reproducibility, size limits, and storage errors."""

from __future__ import annotations

import numpy as np
import pytest

import andrey.data as data
from andrey.core.structure import TemporalStructure
from andrey.data.temporal import MAX_TEMPORAL_D


def _var_scm(d=8, n_lags=1, noise="gaussian"):
    return data.SCM(
        graph=data.graphs.erdos_renyi(d, 2.0),
        functional=data.functional.var(n_lags=n_lags),
        noise=getattr(data.noise, noise)(),
    )


def test_var_valid_temporal_structure():
    ds = _var_scm(d=8, n_lags=1).sample(n=500, seed=0)
    assert ds.data.shape == (500, 8)
    assert np.all(np.isfinite(ds.data))
    assert isinstance(ds.graph, TemporalStructure)
    assert ds.graph.n_lags == 2  # lag 0 (instantaneous) + lag 1
    assert tuple(ds.graph.lags) == (0, 1)
    assert ds.graph.lag_weights.shape == (2, 8, 8)
    assert ds.graph.lag(0).kind == "dag"  # instantaneous is a DAG


def test_var_stationary():
    """A sampled VAR has finite values and bounded spread."""
    ds = _var_scm(d=10, n_lags=2).sample(n=2000, seed=1)
    assert np.all(np.isfinite(ds.data))
    assert ds.data.std() < 1e3  # a stable VAR does not blow up


def test_varma_has_ma_stack():
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(6, 2.0),
        functional=data.functional.varma(n_lags=1, ma_lags=1),
        noise=data.noise.gaussian(),
    )
    ds = scm.sample(n=500, seed=0)
    assert np.all(np.isfinite(ds.data))
    assert ds.graph.lag_weights_ma is not None
    assert ds.graph.lag_weights_ma.shape == (1, 6, 6)


def test_temporal_reproducible():
    scm = _var_scm(d=8, n_lags=1, noise="uniform")
    a, b = scm.sample(n=300, seed=5), scm.sample(n=300, seed=5)
    assert np.array_equal(a.data, b.data)
    assert a.graph == b.graph


def test_temporal_size_gate_raises():
    scm = _var_scm(d=MAX_TEMPORAL_D + 1)
    with pytest.raises(ValueError, match="MAX_TEMPORAL_D"):
        scm.sample(n=50, seed=0)


def test_temporal_rejects_latents_and_scale():
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(8, 2.0),
        functional=data.functional.var(),
        noise=data.noise.gaussian(),
        latents=2,
    )
    with pytest.raises(NotImplementedError, match="latent"):
        scm.sample(n=100, seed=0)
    with pytest.raises(ValueError, match="only scale='raw'"):
        _var_scm().sample(n=100, seed=0, scale="standardize")


def test_temporal_storage_raises_without_safetensors(monkeypatch, tmp_path):
    """Saving temporal truth raises before loading the optional storage dependency.

    The error identifies unsupported temporal storage even when the dependency is absent.
    """
    from andrey.data import storage

    def _no_safetensors():
        raise ImportError("safetensors not installed")

    monkeypatch.setattr(storage, "_require_safetensors", _no_safetensors)
    ds = _var_scm(d=6).sample(n=100, seed=0)
    with pytest.raises(NotImplementedError, match="TemporalStructure"):
        storage.save(ds, tmp_path / "t.safetensors")


def test_stabilize_stationary_at_high_lag_order():
    """Stabilization keeps the spectral radius below one at high lag order."""
    from andrey.data.temporal import _spectral_radius, stabilize

    d, p = 40, 12
    b0 = np.zeros((d, d))
    a_stack = np.zeros((p, d, d))
    a_stack[-1] = 2.0  # adversarial: all mass on the last lag
    a_stack, ib_inv = stabilize(b0, a_stack)
    assert _spectral_radius(ib_inv, a_stack) < 1.0


def test_lag_matrices_have_self_memory():
    """Every generated lag matrix has a nonzero diagonal."""
    from andrey.data.temporal import draw_lag_stack

    stack = draw_lag_stack(
        8, 2, density=0.05, weight_range=(0.5, 2.0), rng=np.random.default_rng(0)
    )
    for k in range(stack.shape[0]):
        assert np.all(np.diag(stack[k]) != 0)


@pytest.mark.parametrize("n_lags", [1, 2])
def test_truth_weights_sit_on_the_truth_edges(n_lags):
    """``lag_weights[k, i, j]`` is the weight of ``i -> j``, the edge the lag's graph draws."""
    truth = _var_scm(d=6, n_lags=n_lags).sample(n=50, seed=3).graph
    for position, lag in enumerate(truth.lags):
        weights = truth.lag_weights[position]
        support = {(int(i), int(j)) for i, j in zip(*np.nonzero(weights), strict=True)}
        assert support == {(i, j) for i, j, _ in truth.lag(lag).oriented_edges()}
        assert support != {(j, i) for i, j in support}  # asymmetric, so a transpose would fail


def test_truth_stores_the_process_matrices_transposed():
    """The process matrices are ``[effect, cause]``; the truth stores ``[cause, effect]``."""
    from andrey.data.graphs import DAGDraw
    from andrey.data.temporal import build_b0, build_truth

    draw = DAGDraw(d=2, parents=np.array([0]), children=np.array([1]), topo_order=np.array([0, 1]))
    b0 = build_b0(draw, np.array([0.5]))  # 0 -> 1
    a_stack = np.array([[[0.3, 0.0], [0.7, 0.2]]])  # a[1, 0]: 0 at t-1 -> 1 at t
    m_stack = np.array([[[0.0, 0.4], [0.0, 0.0]]])  # m[0, 1]: noise of 1 at t-1 -> 0 at t
    truth = build_truth(draw, b0, a_stack, m_stack)
    assert truth.lag_weights[0][0, 1] == 0.5 and truth.lag_weights[0][1, 0] == 0.0
    assert truth.lag_weights[1][0, 1] == 0.7 and truth.lag_weights[1][1, 0] == 0.0
    assert truth.lag_weights_ma[0][1, 0] == 0.4 and truth.lag_weights_ma[0][0, 1] == 0.0
