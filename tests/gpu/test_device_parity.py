"""GPU device-parity: the real CUDA/MPS kernels match the CPU float64 oracle and are deterministic.

The one check the free hosted runners cannot run. numpy float64 is the oracle; the
on-device torch path (``cov`` / ``corrcoef`` / ``entropy``) must agree with it. Tolerance is
per-device: ``cuda`` computes in float64 (tight, ~1e-9), ``mps`` in float32 (the GPU tier,
``atol 1e-3``). The device is pinned via ``andrey.config`` (higher precedence than any env), and a
dispatch + hardware canary guards against a silent numpy fallback passing trivially.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.core import backend, stats

pytest.importorskip("torch")

# Per-device numpy-agreement tolerance: cuda/float64 is tight; mps/float32 holds the GPU tier atol.
_TOL = {"cuda": {"rtol": 1e-9, "atol": 1e-9}, "mps": {"rtol": 1e-3, "atol": 1e-3}}


@pytest.fixture
def data():
    # Non-trivial size so the GPU reduction ordering is actually exercised (not a 6x200 toy).
    rng = np.random.default_rng(0)
    return rng.standard_normal((8, 4000))  # 8 variables, 4000 samples (rowvar=True)


@pytest.fixture
def standardized():
    rng = np.random.default_rng(1)
    U = rng.standard_normal((5000, 6))
    return (U - U.mean(0)) / U.std(0)


def test_pinned_device_actually_runs_on_hardware(device):
    """Guard against a silent CPU pass: prove the pin dispatches and executes on the device."""
    with andrey.config(backend=device):
        # A pin overrides the size threshold, so even a tiny call must resolve to the device.
        assert backend.resolve(1, stats._COV_GPU_THRESHOLDS) == device
    # describe() sees the hardware, and a real tensor lands on it at the expected precision.
    assert getattr(andrey.describe(), device) is True
    t = backend.as_tensor(np.ones(4), device)
    assert t.device.type == device
    expected_dtype = backend.torch().float32 if device == "mps" else backend.torch().float64
    assert t.dtype == expected_dtype


def test_cov_corrcoef_match_oracle(device, data):
    tol = _TOL[device]
    with andrey.config(backend=device):
        assert np.allclose(stats.cov(data), np.cov(data), **tol)
        assert np.allclose(stats.corrcoef(data), np.corrcoef(data), **tol)
        # Exercise the transpose + correction mapping too, not just the default combo.
        assert np.allclose(stats.cov(data.T, rowvar=False), np.cov(data.T, rowvar=False), **tol)
        assert np.allclose(stats.cov(data, ddof=0), np.cov(data, ddof=0), **tol)
        assert np.allclose(
            stats.corrcoef(data.T, rowvar=False), np.corrcoef(data.T, rowvar=False), **tol
        )


def test_entropy_matches_oracle(device, standardized):
    ref = stats._entropy_numpy(standardized)  # the numpy reference the torch path must match
    with andrey.config(backend=device):
        got = stats.entropy(standardized)
    assert got.shape == ref.shape
    assert np.allclose(got, ref, **_TOL[device])


def test_on_device_determinism(device, standardized, data):
    """Same input -> bit-identical output across repeats (these ops consume no RNG)."""
    with andrey.config(backend=device):
        e1, e2 = stats.entropy(standardized), stats.entropy(standardized)
        c1, c2 = stats.cov(data), stats.cov(data)
    if device == "cuda":
        # float64 + use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG -> bit-identical.
        assert np.array_equal(e1, e2) and np.array_equal(c1, c2)
    else:
        # mps float32: no bit-deterministic guarantee -> tolerance-stable (~1e-5 rel;
        # docs/docs/ci.md).
        assert np.allclose(e1, e2, rtol=1e-5, atol=1e-6)
        assert np.allclose(c1, c2, rtol=1e-5, atol=1e-6)
