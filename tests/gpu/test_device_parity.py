"""GPU device parity: the CUDA and MPS paths match the NumPy float64 result and run on the device.

The one check the free hosted runners cannot run. NumPy float64 is the reference. The kernels
(``cov``, ``corrcoef``, ``entropy``) and the methods that read them (PC with Fisher-Z, GES with
BIC, DirectLiNGAM) run on ``cuda`` or ``mps``; KCI, CMIknn, and PNL run on ``cuda`` only. Tolerance
is per device: ``cuda`` computes in float64 (tight, ~1e-9), ``mps`` in float32 (the GPU tier,
``atol 1e-3``). The device is pinned via ``andrey.config`` (higher precedence than any env), each
test reads the dispatch counts of its device calls, and a hardware canary checks the device, so a
silent numpy fallback fails.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.core import backend, stats
from andrey.core.ci_cmiknn import CMIknn
from andrey.core.ci_kernel import KCI
from andrey.irregular.pnl import pnl_pairs

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
        assert backend.resolve(1, backend.GPU_THRESHOLDS["cov"]) == device
    # describe() sees the hardware, and a real tensor lands on it at the expected precision.
    assert getattr(andrey.describe(), device) is True
    t = backend.as_tensor(np.ones(4), device)
    assert t.device.type == device
    expected_dtype = backend.torch().float32 if device == "mps" else backend.torch().float64
    assert t.dtype == expected_dtype


def test_cov_corrcoef_match_oracle(device, data):
    tol = _TOL[device]
    backend._dispatch_counts.clear()
    with andrey.config(backend=device):
        assert np.allclose(stats.cov(data), np.cov(data), **tol)
        assert np.allclose(stats.corrcoef(data), np.corrcoef(data), **tol)
        # Exercise the transpose + correction mapping too, not just the default combo.
        assert np.allclose(stats.cov(data.T, rowvar=False), np.cov(data.T, rowvar=False), **tol)
        assert np.allclose(stats.cov(data, ddof=0), np.cov(data, ddof=0), **tol)
        assert np.allclose(
            stats.corrcoef(data.T, rowvar=False), np.corrcoef(data.T, rowvar=False), **tol
        )
    assert backend.dispatch_stats() == {("cov", device): 3, ("corrcoef", device): 2}


def test_entropy_matches_oracle(device, standardized):
    ref = stats._entropy_numpy(standardized)  # the numpy reference the torch path must match
    backend._dispatch_counts.clear()
    with andrey.config(backend=device):
        got = stats.entropy(standardized)
    assert backend.dispatch_stats() == {("entropy", device): 1}
    assert got.shape == ref.shape
    assert np.allclose(got, ref, **_TOL[device])


def test_on_device_determinism(device, standardized, data):
    """Same input -> bit-identical output across repeats (these ops consume no RNG)."""
    backend._dispatch_counts.clear()
    with andrey.config(backend=device):
        e1, e2 = stats.entropy(standardized), stats.entropy(standardized)
        c1, c2 = stats.cov(data), stats.cov(data)
    assert backend.dispatch_stats() == {("entropy", device): 2, ("cov", device): 2}
    if device == "cuda":
        # float64 + use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG -> bit-identical.
        assert np.array_equal(e1, e2) and np.array_equal(c1, c2)
    else:
        # mps float32: no bit-deterministic guarantee -> tolerance-stable (~1e-5 rel;
        # docs/docs/ci.md).
        assert np.allclose(e1, e2, rtol=1e-5, atol=1e-6)
        assert np.allclose(c1, c2, rtol=1e-5, atol=1e-6)


@pytest.fixture
def linear():
    """A linear model with uniform noise whose columns hold ``x3, x0, x4, x1, x2``:
    ``x0 -> x1 -> x2 -> x3 -> x4`` and ``x1 -> x4``. Its tests, scores, and causal order decide
    by margins wider than float32 rounding, so a difference on ``mps`` is a fault of the path."""
    rng = np.random.default_rng(0)
    u = rng.uniform(-1, 1, (2000, 5))
    x = np.zeros((2000, 5))
    x[:, 0] = u[:, 0]
    x[:, 1] = 0.9 * x[:, 0] + u[:, 1]
    x[:, 2] = -0.8 * x[:, 1] + u[:, 2]
    x[:, 3] = 0.7 * x[:, 2] + u[:, 3]
    x[:, 4] = 0.8 * x[:, 3] - 0.6 * x[:, 1] + u[:, 4]
    return x[:, [3, 0, 4, 1, 2]]


@pytest.mark.parametrize(
    ("method", "op"), [("pc", "corrcoef"), ("ges", "cov"), ("direct_lingam", "entropy")]
)
def test_a_method_finds_the_same_graph_and_order_on_the_device(device, linear, method, op):
    """PC's Fisher-Z correlations, GES's BIC covariance, and DirectLiNGAM's entropies run on the
    device, and the method finds the graph and causal order it finds on NumPy."""
    fit = getattr(andrey, method)
    backend._dispatch_counts.clear()
    with andrey.config(backend="numpy"):
        cpu = fit(linear)
    assert {key for key in backend.dispatch_stats() if key[0] == op} == {(op, "numpy")}
    backend._dispatch_counts.clear()
    with andrey.config(backend=device):
        got = fit(linear)
    assert {key for key in backend.dispatch_stats() if key[0] == op} == {(op, device)}
    assert got.structure.oriented_edges() == cpu.structure.oriented_edges()
    assert got.ordering == cpu.ordering


@pytest.fixture
def nonlinear():
    """Columns ``x, y, z1, z2, w``: ``x`` and ``y`` nonlinear in ``z``, ``w`` driven by ``y``."""
    rng = np.random.default_rng(2)
    z = rng.standard_normal((600, 2))
    s = z.sum(axis=1)
    x = np.tanh(s) + 0.4 * rng.standard_normal(600)
    y = s**2 + 0.4 * rng.standard_normal(600)
    return np.column_stack([x, y, z, np.cos(y) + 0.3 * rng.standard_normal(600)])


def test_kci_and_cmiknn_match_the_cpu(device, nonlinear):
    """KCI on cuda matches the CPU's p-values within 1e-9 and CMIknn's equal them, and PC with each
    finds the same graph."""
    if device != "cuda":
        pytest.skip("KCI and CMIknn have no mps backend")
    queries = [(0, 1, (2,)), (0, 1, (2, 3)), (1, 4, (0,)), (0, 4, (1, 2, 3))]
    kci, cmiknn = KCI(nonlinear), CMIknn(nonlinear, n_permutations=100)
    backend._dispatch_counts.clear()
    with andrey.config(backend="numpy"):
        cpu_kci = [kci(*q) for q in queries]
        cpu_cmiknn = [cmiknn(*q) for q in queries]
        cpu_graphs = [andrey.pc(nonlinear, indep_test=t).structure for t in ("kci", "cmiknn")]
    assert set(backend.dispatch_stats()) == {("kci", "numpy"), ("cmiknn", "numpy")}
    backend._dispatch_counts.clear()
    with andrey.config(backend=device):
        got_kci = [kci(*q) for q in queries]
        got_cmiknn = [cmiknn(*q) for q in queries]
        graphs = [andrey.pc(nonlinear, indep_test=t).structure for t in ("kci", "cmiknn")]
    assert set(backend.dispatch_stats()) == {("kci", "cuda"), ("cmiknn", "cuda")}
    assert np.allclose(got_kci, cpu_kci, rtol=0, atol=1e-9)
    assert got_cmiknn == cpu_cmiknn
    assert [g.oriented_edges() for g in graphs] == [g.oriented_edges() for g in cpu_graphs]


def test_pnl_matches_the_cpu(device):
    """PNL on cuda matches the CPU's p-values and statistics to rounding and makes the same
    decisions, on 2,000 rows where dependent pairs' p-values are zero and the statistics decide,
    with one worker or three."""
    if device != "cuda":
        pytest.skip("PNL has no mps backend")
    rng = np.random.default_rng(3)
    x = rng.standard_normal(2000)
    y = np.tanh(x + 0.4 * rng.laplace(size=2000)) ** 3
    u = rng.standard_normal(2000)  # a hidden cause of the last two columns
    a = u**3 + 0.3 * rng.standard_normal(2000)
    b = np.sin(2 * u) + 0.3 * rng.standard_normal(2000)
    table = np.column_stack([x, y, a, b])
    with andrey.config(backend="numpy"):
        cpu = pnl_pairs(table)
    backend._dispatch_counts.clear()
    with andrey.config(backend=device):
        got = pnl_pairs(table)
        threaded = pnl_pairs(table, n_jobs=3)
    assert backend.dispatch_stats() == {("pnl", "cuda"): 2}
    assert (cpu[:, :2] == 0).all(axis=1).any()  # the statistics decide a pair
    # The largest difference over 4,490 pairs of up to 5,000 rows was 5e-15 on an A100.
    for rows in (got, threaded):
        assert np.allclose(rows[:, :2], cpu[:, :2], rtol=0, atol=1e-12)
        assert np.allclose(rows[:, 2:], cpu[:, 2:], rtol=1e-12, atol=0)

    def decisions(rows):
        by_pvalue = np.sign(rows[:, 0] - rows[:, 1])
        return np.where(by_pvalue != 0, by_pvalue, np.sign(rows[:, 3] - rows[:, 2]))

    assert np.array_equal(decisions(got), decisions(cpu))
    assert np.array_equal(decisions(threaded), decisions(cpu))
