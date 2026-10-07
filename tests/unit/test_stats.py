"""Statistical-primitive tests: numpy is the exact oracle; the torch path matches it, deterministic.

The numpy-path tests always run (they pin ``ANDREY_DEVICE=numpy``). The torch-path tests run only
when the ``[torch]`` extra is installed -- the CI torch lane pins ``ANDREY_DEVICE=cpu`` to exercise
them with no GPU. Real on-device parity lives in ``tests/gpu`` (the non-blocking self-hosted lane).
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from andrey.core import backend, env, stats
from andrey.core.stats import corrcoef, cov, entropy


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((6, 200))  # 6 variables, 200 samples (rowvar=True)


@pytest.fixture
def standardized():
    rng = np.random.default_rng(1)
    U = rng.standard_normal((500, 4))
    return (U - U.mean(0)) / U.std(0)


# ---- numpy path: numpy is the exact oracle ------------------------------------------------


@pytest.fixture
def force_numpy(monkeypatch):
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


def test_cov_equals_numpy_exactly(force_numpy, data):
    assert np.array_equal(cov(data), np.cov(data))
    assert np.array_equal(cov(data.T, rowvar=False), np.cov(data.T, rowvar=False))
    assert np.array_equal(cov(data, ddof=0), np.cov(data, ddof=0))


def test_corrcoef_equals_numpy_exactly(force_numpy, data):
    assert np.array_equal(corrcoef(data), np.corrcoef(data))
    assert np.array_equal(corrcoef(data.T, rowvar=False), np.corrcoef(data.T, rowvar=False))


def test_entropy_numpy_reference(force_numpy, standardized):
    e = entropy(standardized)
    assert e.shape == (4,)
    assert np.isfinite(e).all()
    # Differential entropy of a standard normal is 0.5 * (1 + log(2*pi)) ~= 1.4189.
    assert np.allclose(e, 1.4189, atol=0.05)


def test_entropy_1d_matches_single_column(force_numpy, standardized):
    col = standardized[:, 0]
    one_d = entropy(col)  # 1-D input is promoted to a single column
    assert one_d.shape == (1,)
    assert np.allclose(one_d, entropy(col[:, None]))


def test_numpy_path_is_deterministic(force_numpy, standardized, data):
    assert np.array_equal(entropy(standardized), entropy(standardized))
    assert np.array_equal(cov(data), cov(data))


# ---- torch path: matches the numpy oracle within device tolerance -------------------------


@pytest.fixture
def torch_cpu(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("ANDREY_DEVICE", "cpu")
    # A pinned device overrides the size threshold, so even a tiny call runs on torch-cpu.
    assert backend.resolve(1, stats._COV_GPU_THRESHOLDS) == "cpu"


def test_torch_cov_matches_numpy(torch_cpu, data):
    assert np.allclose(cov(data), np.cov(data), atol=1e-3)
    assert np.allclose(corrcoef(data), np.corrcoef(data), atol=1e-3)
    # Exercise the torch branch's transpose + correction mapping, not just the default combo.
    assert np.allclose(cov(data.T, rowvar=False), np.cov(data.T, rowvar=False), atol=1e-3)
    assert np.allclose(cov(data, ddof=0), np.cov(data, ddof=0), atol=1e-3)
    assert np.allclose(corrcoef(data.T, rowvar=False), np.corrcoef(data.T, rowvar=False), atol=1e-3)


def test_torch_entropy_matches_reference_and_is_deterministic(torch_cpu, standardized):
    ref = stats._entropy_numpy(standardized)
    got = entropy(standardized)
    assert np.allclose(got, ref, atol=1e-3)
    assert np.array_equal(entropy(standardized), got)


# ---- the dispatch gate wiring: element metrics, cadence, the counts -----------------------


def test_entropy_gates_on_elements_and_is_amortized(monkeypatch):
    # Entropy gates on `n*B` elements and recurs often enough to amortize device startup.
    monkeypatch.delenv("ANDREY_ENTROPY_GPU_THRESHOLD", raising=False)
    monkeypatch.delenv("ANDREY_GPU_CALIBRATE", raising=False)
    seen = {}

    def spy(metric, threshold, *, amortized=False, op=None):
        seen.update(metric=metric, amortized=amortized, op=op)
        return "numpy"

    monkeypatch.setattr(stats.backend, "resolve", spy)
    U = np.random.default_rng(0).standard_normal((300, 5))
    stats.entropy(U)
    assert seen == {"metric": U.size, "amortized": True, "op": "entropy"}


def test_cov_and_corrcoef_are_one_shot(monkeypatch):
    # Once-per-fit ops must resolve with amortized=False: never the call that boots the GPU.
    monkeypatch.delenv("ANDREY_COV_GPU_THRESHOLD", raising=False)
    monkeypatch.delenv("ANDREY_GPU_CALIBRATE", raising=False)
    calls = []

    def spy(metric, threshold, *, amortized=False, op=None):
        calls.append((metric, amortized, op))
        return "numpy"

    monkeypatch.setattr(stats.backend, "resolve", spy)
    X = np.random.default_rng(0).standard_normal((40, 3))
    stats.cov(X)
    stats.corrcoef(X)
    assert calls == [(X.size, False, "cov"), (X.size, False, "corrcoef")]


def test_mps_cutoffs_are_unmeasured_never():
    # mps has no measured crossover: None = auto never offloads there (a pin still can).
    assert stats._ENTROPY_GPU_THRESHOLDS["mps"] is None
    assert stats._COV_GPU_THRESHOLDS["mps"] is None


def test_probe_builders_measure_the_gate_metric():
    # Calibration sweeps element count with a tall, workload-realistic aspect ratio.
    for builder in (stats._entropy_probe, stats._cov_probe):
        for metric in (10_000, 25_000, 100_000):
            x = builder(metric)
            assert 0.9 <= x.size / metric <= 1.1
            n, b = x.shape
            assert n > b  # rows >> columns, like the real batches


def test_direct_lingam_batches_reach_the_gpu_under_auto(monkeypatch):
    # Tall DirectLiNGAM batches above the element cutoff offload; small batches stay on numpy.
    # CUDA availability and the torch kernel are stubbed to test dispatch without a GPU.
    for var in (
        "ANDREY_ENTROPY_GPU_THRESHOLD",
        "ANDREY_GPU_CALIBRATE",
        "ANDREY_BACKEND",
        "ANDREY_DEVICE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(backend, "_is_available", lambda b: b in ("cuda", "numpy"))
    monkeypatch.setattr(stats, "_entropy_torch", lambda U, dev: stats._entropy_numpy(U))
    from andrey.lingam.pwling import score_candidates

    # Derive both workloads from the configured cutoff.
    cut = stats._ENTROPY_GPU_THRESHOLDS["cuda"]
    X = np.random.default_rng(0).standard_normal((1000, 60))
    assert 1000 * 59 > cut  # the tall-batch case is above the gate
    backend.reset_dispatch_stats()
    score_candidates(X, list(range(60)))
    assert backend.dispatch_stats().get(("entropy", "cuda"), 0) > 0

    small_n = max(20, cut // (12 * 4))  # batches near a quarter of the cutoff: clearly below it
    backend.reset_dispatch_stats()
    score_candidates(X[:small_n, :12], list(range(12)))
    counts = backend.dispatch_stats()
    assert counts.get(("entropy", "cuda"), 0) == 0 and counts[("entropy", "numpy")] > 0


def test_stats_ops_record_dispatch(force_numpy, data, standardized):
    backend.reset_dispatch_stats()
    cov(data)
    corrcoef(data)
    entropy(standardized)
    counts = backend.dispatch_stats()
    assert counts[("cov", "numpy")] == 1
    assert counts[("corrcoef", "numpy")] == 1
    assert counts[("entropy", "numpy")] == 1


# ---- opt-in calibration (ANDREY_GPU_CALIBRATE): integration logic, no GPU needed ----


def _cov_thr(*, amortized=True):
    return stats._resolve_thresholds(
        "cov",
        env.COV_GPU_THRESHOLD,
        stats._COV_GPU_THRESHOLDS,
        stats._cov_probe,
        stats._cov_probe_call,
        stats._COV_PROBE_START,
        amortized=amortized,
    )


def _fake_clock(crossover):
    """Time ``_timed`` so the device beats numpy exactly at and above ``crossover`` elements."""

    def timed(call, x):
        if backend.config.backend == "numpy":
            return 1.0
        return 0.001 if x.size >= crossover else 10.0

    return timed


def test_calibrate_is_a_noop_without_an_accelerator(monkeypatch):
    # Flag on but no GPU (the CI case): fall back to the static per-device defaults, no benchmark.
    monkeypatch.setenv("ANDREY_GPU_CALIBRATE", "1")
    monkeypatch.delenv("ANDREY_COV_GPU_THRESHOLD", raising=False)
    monkeypatch.delenv("ANDREY_BACKEND", raising=False)
    monkeypatch.delenv("ANDREY_DEVICE", raising=False)
    monkeypatch.setattr(stats.backend, "_is_available", lambda b: b == "numpy")
    assert _cov_thr() == stats._COV_GPU_THRESHOLDS


def test_calibrate_tunes_only_the_available_device(monkeypatch):
    monkeypatch.setenv("ANDREY_GPU_CALIBRATE", "1")
    monkeypatch.delenv("ANDREY_COV_GPU_THRESHOLD", raising=False)
    monkeypatch.delenv("ANDREY_BACKEND", raising=False)
    monkeypatch.delenv("ANDREY_DEVICE", raising=False)
    monkeypatch.setattr(stats.backend, "_is_available", lambda b: b in ("cuda", "numpy"))
    monkeypatch.setattr(stats, "_calibrate", lambda *a, **k: 12345)
    thr = _cov_thr()
    assert thr["cuda"] == 12345  # calibrated for the present device
    assert thr["mps"] == stats._COV_GPU_THRESHOLDS["mps"]  # others left at the default


def test_manual_env_override_beats_calibration(monkeypatch):
    monkeypatch.setenv("ANDREY_GPU_CALIBRATE", "1")
    monkeypatch.setenv("ANDREY_COV_GPU_THRESHOLD", "42")
    assert _cov_thr() == 42  # the explicit override wins; calibration is not consulted


def test_pinned_call_skips_calibration(monkeypatch):
    # Calibration runs only under `auto`, so its pinned probe calls cannot recurse.
    monkeypatch.setenv("ANDREY_GPU_CALIBRATE", "1")
    monkeypatch.delenv("ANDREY_COV_GPU_THRESHOLD", raising=False)
    monkeypatch.setattr(stats.backend, "_is_available", lambda b: b in ("cuda", "numpy"))

    def _boom(*a, **k):
        raise AssertionError("calibration ran under a pin")

    monkeypatch.setattr(stats, "_calibrate", _boom)
    with backend.config(backend="numpy"):
        assert _cov_thr() == stats._COV_GPU_THRESHOLDS


def test_calibration_cache_is_reused(monkeypatch):
    # A cached fingerprint-specific cutoff is returned without benchmarking.
    monkeypatch.setenv("ANDREY_GPU_CALIBRATE", "1")
    monkeypatch.delenv("ANDREY_COV_GPU_THRESHOLD", raising=False)
    monkeypatch.delenv("ANDREY_BACKEND", raising=False)
    monkeypatch.delenv("ANDREY_DEVICE", raising=False)
    monkeypatch.setattr(stats.backend, "_is_available", lambda b: b in ("cuda", "numpy"))
    # `conftest` clears this process-level entry after the test.
    stats._CALIBRATED[f"cov|{stats._device_fingerprint('cuda')}"] = 777
    assert _cov_thr()["cuda"] == 777


def test_calibration_is_cached_per_host_and_device_on_disk(monkeypatch, tmp_path):
    # One process measures; a later process (fresh in-process cache) reuses the on-disk cutoff.
    import json

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("ANDREY_BACKEND", raising=False)
    monkeypatch.delenv("ANDREY_DEVICE", raising=False)
    monkeypatch.setattr(stats, "_timed", _fake_clock(4_000))
    make, call = (lambda m: np.empty(m)), (lambda x: None)
    got = stats._calibrate("entropy", "cuda", make, call, stats._ENTROPY_PROBE_START, 999)
    payload = json.loads((tmp_path / "andrey" / "gpu_calibration.json").read_text())
    assert got in payload.values() and all(k.startswith("entropy|") for k in payload)

    # Clear the in-process cache and require the next lookup to use disk.
    stats._CALIBRATED.clear()

    def _boom(call, x):
        raise AssertionError("re-measured despite a disk-cached calibration")

    monkeypatch.setattr(stats, "_timed", _boom)
    assert stats._calibrate("entropy", "cuda", make, call, stats._ENTROPY_PROBE_START, 999) == got


def test_calibration_failure_is_not_persisted(monkeypatch, tmp_path):
    # A device error uses the process-local default without writing a disk entry.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def _boom(call, x):
        raise RuntimeError("device fell over")

    monkeypatch.setattr(stats, "_timed", _boom)
    make, call = (lambda m: np.empty(2)), (lambda x: None)
    assert stats._calibrate("entropy", "cuda", make, call, 10, 999) == 999
    assert not (tmp_path / "andrey" / "gpu_calibration.json").exists()


def test_crossover_below_the_first_size_tried_is_found(monkeypatch):
    # Downward bracketing can find a crossover below the initial probe size.
    monkeypatch.setattr(stats, "_timed", _fake_clock(3_000))
    make, call = (lambda m: np.empty(m)), (lambda x: None)
    got = stats._crossover("cuda", make, call, stats._ENTROPY_PROBE_START)
    assert got < stats._ENTROPY_PROBE_START  # not floored at where the search started
    assert 3_000 <= got <= 3_000 * stats._BRACKET_TOL  # brackets the truth from above


def test_a_single_noisy_reading_does_not_set_the_cutoff(monkeypatch):
    # Inject one spurious device win at the first descending probe.
    truth = 20_000
    spurious_at = stats._ENTROPY_PROBE_START // 4  # the step-down search probes this first
    state = {"fired": False}

    def timed(call, x):
        if backend.config.backend == "numpy":
            return 1.0
        if x.size == spurious_at and not state["fired"]:
            state["fired"] = True
            return 0.001  # one impossibly fast reading, outvoted by the other rounds
        return 0.001 if x.size >= truth else 10.0

    monkeypatch.setattr(stats, "_timed", timed)
    make, call = (lambda m: np.empty(m)), (lambda x: None)
    got = stats._crossover("cuda", make, call, stats._ENTROPY_PROBE_START)
    assert got is not None
    assert truth <= got <= truth * stats._BRACKET_TOL  # the outlier did not move the answer


def test_crossover_never_returns_a_size_it_did_not_probe(monkeypatch):
    # Exhausting the budget without a confirmed win reports no crossover.
    monkeypatch.setattr(stats, "_MAX_PROBES", 3)
    monkeypatch.setattr(stats, "_timed", _fake_clock(1 << 40))
    make, call = (lambda m: np.empty(min(m, 4))), (lambda x: None)
    assert stats._crossover("cuda", make, call, stats._ENTROPY_PROBE_START) is None


def test_crossover_floors_when_the_device_wins_everywhere(monkeypatch):
    monkeypatch.setattr(stats, "_timed", _fake_clock(0))  # device wins at every size
    make, call = (lambda m: np.empty(m)), (lambda x: None)
    assert stats._crossover("cuda", make, call, stats._ENTROPY_PROBE_START) == stats._PROBE_FLOOR


def test_calibration_probes_stay_out_of_the_dispatch_counts(monkeypatch, tmp_path):
    # Calibration timing calls are not user dispatch decisions.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(stats, "_timed", _fake_clock(4_000))
    monkeypatch.setattr(backend, "_is_available", lambda b: b in ("cuda", "numpy"))
    monkeypatch.setattr(stats, "_entropy_torch", lambda U, dev: stats._entropy_numpy(U))
    backend.reset_dispatch_stats()
    stats._crossover("cuda", stats._entropy_probe, stats.entropy, stats._ENTROPY_PROBE_START)
    assert backend.dispatch_stats() == {}


def test_crossover_probes_the_ceiling_before_giving_up(monkeypatch):
    # The ceiling remains a valid probe point.
    monkeypatch.setattr(stats, "_timed", _fake_clock(stats._PROBE_CEILING))
    make, call = (lambda m: np.empty(m)), (lambda x: None)
    assert stats._crossover("cuda", make, call, stats._COV_PROBE_START) == stats._PROBE_CEILING


def test_crossover_returns_only_sizes_it_probed(monkeypatch):
    # Every result, including the floor, is a timed size where the device won.
    for truth in (900, 3_000, 40_000):
        probed = []

        def timed(call, x, clock=_fake_clock(truth), probed=probed):
            probed.append(x.size)
            return clock(call, x)

        monkeypatch.setattr(stats, "_timed", timed)
        got = stats._crossover(
            "cuda", (lambda m: np.empty(m)), (lambda x: None), stats._ENTROPY_PROBE_START
        )
        assert got in probed
        assert got >= max(truth, stats._PROBE_FLOOR)


def test_crossover_reports_a_device_that_never_wins(monkeypatch):
    monkeypatch.setattr(stats, "_timed", _fake_clock(1 << 40))  # beyond the ceiling
    make, call = (lambda m: np.empty(min(m, 4))), (lambda x: None)
    assert stats._crossover("cuda", make, call, stats._ENTROPY_PROBE_START) is None


def test_never_wins_is_cached_but_expires(monkeypatch, tmp_path):
    # A time-limited cache avoids repeated probes without preserving a transient result forever.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(stats, "_crossover", lambda *a, **k: None)
    make, call = (lambda m: np.empty(2)), (lambda x: None)
    assert stats._calibrate("entropy", "cuda", make, call, 10, 999) == stats._NEVER

    fingerprint = f"entropy|{stats._device_fingerprint('cuda')}"
    assert stats._load_calibration(fingerprint) == stats._NEVER  # a fresh verdict is honored
    monkeypatch.setattr(stats, "_NEVER_TTL_S", -1)
    assert stats._load_calibration(fingerprint) is None  # a stale one is re-measured


def test_one_shot_op_does_not_calibrate_on_a_cold_device(monkeypatch):
    # One-shot calibration cannot initialize the runtime it is meant to treat as cold.
    monkeypatch.setenv("ANDREY_GPU_CALIBRATE", "1")
    monkeypatch.delenv("ANDREY_COV_GPU_THRESHOLD", raising=False)
    monkeypatch.delenv("ANDREY_BACKEND", raising=False)
    monkeypatch.delenv("ANDREY_DEVICE", raising=False)
    monkeypatch.setattr(stats.backend, "_is_available", lambda b: b in ("cuda", "numpy"))
    monkeypatch.setattr(stats.backend, "_is_live", lambda b: False)

    def _boom(*a, **k):
        raise AssertionError("a one-shot op calibrated on a cold device")

    monkeypatch.setattr(stats, "_calibrate", _boom)
    assert _cov_thr(amortized=False) == stats._COV_GPU_THRESHOLDS
    # An already-live device may calibrate.
    monkeypatch.setattr(stats.backend, "_is_live", lambda b: b == "cuda")
    monkeypatch.setattr(stats, "_calibrate", lambda *a, **k: 4242)
    assert _cov_thr(amortized=False)["cuda"] == 4242


def test_cov_calibrates_through_its_own_probe_and_as_a_one_shot(monkeypatch):
    # Capture the probe wrapper and cadence passed by the public operation.
    seen = {}
    monkeypatch.setenv("ANDREY_GPU_CALIBRATE", "1")
    monkeypatch.delenv("ANDREY_COV_GPU_THRESHOLD", raising=False)
    monkeypatch.setattr(
        stats,
        "_resolve_thresholds",
        lambda op, env, defaults, make, call, start, *, amortized: (
            seen.update(op=op, call=call, start=start, amortized=amortized) or defaults
        ),
    )
    stats.cov(np.random.default_rng(0).standard_normal((50, 4)), rowvar=False)
    assert seen["op"] == "cov"
    assert seen["call"] is stats._cov_probe_call  # not bare `cov`, which defaults to rowvar=True
    assert seen["amortized"] is False  # cov runs once per fit; it may not boot a cold runtime

    seen.clear()
    stats.entropy(np.random.default_rng(0).standard_normal((50, 4)))
    assert seen["op"] == "entropy" and seen["amortized"] is True  # recurs; may bootstrap


def test_cov_probe_times_the_orientation_its_callers_use(monkeypatch):
    # Calibration uses the `rowvar=False` orientation passed by BICScore and Fisher-Z.
    seen = {}
    monkeypatch.setattr(stats, "cov", lambda x, **kw: seen.update(kw) or np.eye(2))
    monkeypatch.setattr(stats, "corrcoef", lambda x, **kw: seen.update(kw) or np.eye(2))
    stats._cov_probe_call(stats._cov_probe(10_000))
    assert seen == {"rowvar": False}
    seen.clear()
    stats._corrcoef_probe_call(stats._cov_probe(10_000))
    assert seen == {"rowvar": False}


def test_fingerprint_separates_machines_that_would_measure_differently(monkeypatch):
    # CPU thread settings affect the numpy side of the crossover.
    monkeypatch.setattr(stats, "_cpu_threads", lambda: "8")
    eight = stats._device_fingerprint("cuda")
    monkeypatch.setattr(stats, "_cpu_threads", lambda: "64")
    assert stats._device_fingerprint("cuda") != eight  # BLAS threads move the numpy half
    assert eight.startswith(f"v{stats._CALIBRATION_VERSION}|")  # a probe change retires the cache


def test_fingerprint_changes_when_the_timed_libraries_change(monkeypatch):
    # An upgrade can change kernel speed, so it must not replay a crossover measured before it.
    fake = types.SimpleNamespace(
        __version__="2.5.0",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        get_num_threads=lambda: 8,
    )
    monkeypatch.setattr(stats.backend, "torch", lambda: fake)
    before = stats._device_fingerprint("cuda")
    fake.__version__ = "2.6.0"
    after_torch = stats._device_fingerprint("cuda")
    assert after_torch != before
    monkeypatch.setattr(stats.np, "__version__", "0.0.0")
    assert stats._device_fingerprint("cuda") != after_torch
