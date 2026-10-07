"""Backend dispatch: device resolution, env overrides, graceful fallback, torch-free import."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import andrey
from andrey.core import backend, stats


def test_auto_dispatch_stays_on_numpy_below_threshold(monkeypatch):
    # Default (auto): a call below the threshold resolves to numpy regardless of torch's presence.
    monkeypatch.delenv("ANDREY_DEVICE", raising=False)
    assert backend.resolve(100, 10_000_000) == "numpy"


def test_forced_numpy_ignores_size(monkeypatch):
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")
    assert backend.resolve(10**12, 1) == "numpy"


def test_bad_device_env_raises(monkeypatch):
    monkeypatch.setenv("ANDREY_DEVICE", "quantum")
    if backend.torch() is None:
        pytest.skip("dispatch short-circuits to numpy without torch")
    with pytest.raises(ValueError, match="ANDREY_DEVICE"):
        backend.resolve(1, 1)


def test_unavailable_device_warns_and_falls_back(monkeypatch):
    # A pinned but unavailable accelerator warns and falls back to auto-selection.
    pytest.importorskip("torch")
    monkeypatch.setenv("ANDREY_DEVICE", "mps")
    if backend._is_available("mps"):
        pytest.skip("mps is actually available here")
    with pytest.warns(andrey.BackendFallbackWarning, match="unavailable"):
        dev = backend.resolve(1, 1)
    assert dev != "mps"


def test_pinned_cpu_overrides_threshold(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("ANDREY_DEVICE", "cpu")
    assert backend.resolve(1, 10**9) == "cpu"  # tiny call, still torch-cpu


def test_importing_core_does_not_import_torch():
    # Holds even when the [torch] extra is installed: torch loads lazily inside backend.
    import os
    import pathlib

    import andrey

    src_dir = str(pathlib.Path(andrey.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": src_dir + os.pathsep + os.environ.get("PYTHONPATH", "")}
    code = "import andrey.core, sys; assert 'torch' not in sys.modules; print('ok')"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


# ---- the two-layer grammar: backend + num_workers, layered selection ----


def test_backend_precedence_context_over_config_over_env(monkeypatch):
    monkeypatch.delenv("ANDREY_BACKEND", raising=False)
    monkeypatch.delenv("ANDREY_DEVICE", raising=False)
    assert andrey.config.backend == "auto"  # default
    monkeypatch.setenv("ANDREY_BACKEND", "numpy")
    assert andrey.config.backend == "numpy"  # env
    andrey.config.backend = "numba"
    assert andrey.config.backend == "numba"  # config object over env
    with andrey.config(backend="numpy"):
        assert andrey.config.backend == "numpy"  # context over config object
    assert andrey.config.backend == "numba"  # context restored


def test_andrey_device_is_alias_of_andrey_backend(monkeypatch):
    monkeypatch.delenv("ANDREY_BACKEND", raising=False)
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")
    assert andrey.config.backend == "numpy"


def test_num_workers_resolution(monkeypatch):
    monkeypatch.delenv("ANDREY_NUM_WORKERS", raising=False)
    assert andrey.config.num_workers == 1  # serial default
    monkeypatch.setenv("ANDREY_NUM_WORKERS", "4")
    assert andrey.config.num_workers == 4
    with andrey.config(num_workers=8):
        assert andrey.config.num_workers == 8
    with pytest.raises(ValueError, match="num_workers"):
        andrey.config.num_workers = -2


def test_bad_backend_value_raises():
    with pytest.raises(ValueError, match="backend"):
        andrey.config.backend = "quantum"


def test_describe_reports_state(monkeypatch):
    monkeypatch.delenv("ANDREY_BACKEND", raising=False)
    monkeypatch.delenv("ANDREY_DEVICE", raising=False)
    d = andrey.describe()
    assert d.backend == "auto" and d.num_workers == 1
    assert set(d.as_dict()) == {"backend", "num_workers", "available_backends"}
    assert set(d.as_dict()["available_backends"]) == {"numpy", "numba", "cpu", "cuda", "mps"}


def test_describe_detects_backends_without_importing_them():
    # A fresh process: numba stays unimported, and torch does too unless a device check needs it.
    import json
    import os
    import pathlib

    src_dir = str(pathlib.Path(andrey.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": src_dir + os.pathsep + os.environ.get("PYTHONPATH", "")}
    code = (
        "import json, sys, andrey\n"
        "from andrey.core import backend as b\n"
        "d = andrey.describe()\n"
        "device_check = d.torch and (b._torch_gpu_build() or sys.platform == 'darwin')\n"
        "loaded = sorted({'numba', 'torch'} & set(sys.modules))\n"
        "found = [d.numba, d.torch, d.cuda, d.mps]\n"
        "print(json.dumps({'found': found, 'device_check': device_check, 'loaded': loaded}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    child = json.loads(proc.stdout)
    assert "numba" not in child["loaded"]
    assert child["device_check"] or "torch" not in child["loaded"]
    assert child["found"] == [backend._is_available(b) for b in ("numba", "cpu", "cuda", "mps")]


@pytest.mark.parametrize(
    ("version_py", "gpu"),
    [
        ("cuda: Optional[str] = None\nhip: Optional[str] = None\n", False),
        ("cuda: Optional[str] = '12.8'\nhip: Optional[str] = None\n", True),
        ("cuda: Optional[str] = None\nhip: Optional[str] = '6.4'\n", True),
        (None, True),  # no version file: import torch and ask it
    ],
)
def test_torch_gpu_build_reads_the_version_file(version_py, gpu, tmp_path, monkeypatch):
    import importlib.machinery

    if version_py is not None:
        (tmp_path / "version.py").write_text("from typing import Optional\n" + version_py)
    spec = importlib.machinery.ModuleSpec("torch", None, is_package=True)
    spec.submodule_search_locations = [str(tmp_path)]
    monkeypatch.setattr(backend.importlib.util, "find_spec", lambda name: spec)
    assert backend._torch_gpu_build() is gpu


def test_resolve_bitset_pin_numpy_ignores_numba():
    assert backend.resolve_bitset(backend="numpy") == "numpy"


def test_per_device_threshold_gates_by_chosen_device(monkeypatch):
    # Each accelerator uses its own cutoff. `amortized=True` isolates the size gate.
    monkeypatch.delenv("ANDREY_BACKEND", raising=False)
    monkeypatch.delenv("ANDREY_DEVICE", raising=False)
    monkeypatch.setattr(backend, "_is_available", lambda b: b in ("cuda", "numpy"))
    thr = {"cuda": 1000, "mps": 100_000}
    assert backend.resolve(100, thr, amortized=True) == "numpy"  # below cuda's cutoff
    assert backend.resolve(5000, thr, amortized=True) == "cuda"  # above cuda's cutoff
    assert backend.resolve(5000, 10_000, amortized=True) == "numpy"  # scalar: below -> numpy
    assert backend.resolve(50_000, 10_000, amortized=True) == "cuda"  # scalar: above -> device


# ---- the dispatch gate: element cutoffs, one-shot vs amortized, the counts ----


@pytest.fixture
def fake_cuda(monkeypatch):
    """Expose a synthetic CUDA device without requiring GPU hardware."""
    monkeypatch.delenv("ANDREY_BACKEND", raising=False)
    monkeypatch.delenv("ANDREY_DEVICE", raising=False)
    monkeypatch.setattr(backend, "_is_available", lambda b: b in ("cuda", "numpy"))


def test_amortized_auto_crosses_at_the_element_cutoff(fake_cuda):
    # Numpy handles inputs below the cutoff; CUDA handles inputs at or above it.
    thr = {"cuda": 25_000, "mps": None}
    assert backend.resolve(24_999, thr, amortized=True) == "numpy"
    assert backend.resolve(25_000, thr, amortized=True) == "cuda"


def test_none_cutoff_means_never(fake_cuda):
    # A device without a cutoff never auto-offloads, however large the input.
    assert backend.resolve(10**12, {"cuda": None}, amortized=True) == "numpy"


def test_one_shot_never_selects_a_gpu_that_is_not_live(fake_cuda, monkeypatch):
    # A one-shot op (cov/corrcoef) must never be the call that boots torch/CUDA: above the cutoff,
    # with cuda nominally available but no live torch runtime, auto stays on numpy.
    monkeypatch.setattr(backend, "_is_live", lambda b: False)
    assert backend.resolve(10**9, {"cuda": 25_000, "mps": None}) == "numpy"


def test_one_shot_uses_a_gpu_that_is_already_live(fake_cuda, monkeypatch):
    monkeypatch.setattr(backend, "_is_live", lambda b: b == "cuda")
    thr = {"cuda": 25_000, "mps": None}
    assert backend.resolve(10**9, thr) == "cuda"  # live runtime + above the cutoff -> offload
    assert backend.resolve(100, thr) == "numpy"  # live but below the cutoff -> still numpy


def test_pin_bypasses_cutoff_and_liveness(fake_cuda, monkeypatch):
    # An explicit cuda pin keeps its bypass semantics: no size gate, no liveness gate.
    monkeypatch.setattr(backend, "_is_live", lambda b: False)
    with andrey.config(backend="cuda"):
        assert backend.resolve(1, {"cuda": 25_000, "mps": None}) == "cuda"


def test_one_shot_decision_never_imports_torch():
    # A one-shot decision in a cold interpreter stays on numpy without importing torch.
    import os
    import pathlib

    src_dir = str(pathlib.Path(andrey.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": src_dir + os.pathsep + os.environ.get("PYTHONPATH", "")}
    env.pop("ANDREY_BACKEND", None)
    env.pop("ANDREY_DEVICE", None)
    code = (
        "import sys; from andrey.core import backend; "
        "dev = backend.resolve(10**9, {'cuda': 1, 'mps': None}); "
        "assert dev == 'numpy', dev; "
        "assert 'torch' not in sys.modules, 'deciding imported torch'; "
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_ges_fit_under_auto_never_imports_torch():
    # GES calls covariance once during BICScore initialization, so `auto` must not import torch.
    import os
    import pathlib

    src_dir = str(pathlib.Path(andrey.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": src_dir + os.pathsep + os.environ.get("PYTHONPATH", "")}
    env.pop("ANDREY_BACKEND", None)
    env.pop("ANDREY_DEVICE", None)
    code = (
        "import sys; import numpy as np; import andrey; "
        "from andrey.core.stats import _COV_GPU_THRESHOLDS; "
        # Derive the smallest matrix that clears the current covariance cutoff.
        "d = 8; n = -(-_COV_GPU_THRESHOLDS['cuda'] // d) + 1; "
        "assert n * d >= _COV_GPU_THRESHOLDS['cuda'], 'probe no longer clears the cov cutoff'; "
        "X = np.random.default_rng(0).standard_normal((n, d)); "
        "out = andrey.ges(X); "
        "assert out.structure.kind == 'cpdag'; "
        "assert 'torch' not in sys.modules, 'a backend=auto GES fit imported torch'; "
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_dispatch_counts_record_choices(monkeypatch):
    monkeypatch.delenv("ANDREY_BACKEND", raising=False)
    monkeypatch.delenv("ANDREY_DEVICE", raising=False)
    monkeypatch.setattr(backend, "_is_available", lambda b: b in ("cuda", "numpy"))
    backend.reset_dispatch_stats()
    backend.resolve(10, 100, op="cov")
    backend.resolve(10, 100, op="cov")
    backend.resolve(10**9, {"cuda": 25_000, "mps": None}, amortized=True, op="entropy")
    stats = backend.dispatch_stats()
    assert stats[("cov", "numpy")] == 2  # below the cutoff, twice
    assert stats[("entropy", "cuda")] == 1  # above the cutoff, offloaded (fake cuda)
    backend.reset_dispatch_stats()
    assert backend.dispatch_stats() == {}


def test_dispatch_counts_record_pins(monkeypatch):
    monkeypatch.setattr(backend, "_is_available", lambda b: b in ("cuda", "numpy"))
    backend.reset_dispatch_stats()
    with andrey.config(backend="cuda"):
        backend.resolve(1, {"cuda": 25_000, "mps": None}, op="cov")
    assert backend.dispatch_stats() == {("cov", "cuda"): 1}


# ---- mps: pin-only ----


@pytest.fixture
def fake_mps(monkeypatch):
    """Report a live MPS device without Apple hardware."""
    for var in ("ANDREY_BACKEND", "ANDREY_DEVICE", "ANDREY_GPU_CALIBRATE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(backend, "_is_available", lambda b: b in ("mps", "numpy"))
    monkeypatch.setattr(backend, "_is_live", lambda b: b == "mps")


@pytest.mark.parametrize(
    ("op", "env"),
    [
        ("cov", "ANDREY_COV_GPU_THRESHOLD"),
        ("corrcoef", "ANDREY_COV_GPU_THRESHOLD"),
        ("entropy", "ANDREY_ENTROPY_GPU_THRESHOLD"),
    ],
)
@pytest.mark.parametrize("cutoff", ["threshold", "calibration"])
def test_auto_never_selects_mps(fake_mps, monkeypatch, op, env, cutoff):
    # Neither a threshold of 1 nor calibration takes a torch op to mps under `auto`.
    if cutoff == "threshold":
        monkeypatch.setenv(env, "1")  # one value, applied to every auto device
    else:
        monkeypatch.delenv(env, raising=False)
        monkeypatch.setenv("ANDREY_GPU_CALIBRATE", "1")
        monkeypatch.setattr(stats, "_calibrate", lambda *a, **k: pytest.fail("auto calibrated mps"))
    backend.reset_dispatch_stats()
    getattr(stats, op)(np.random.default_rng(0).standard_normal((8, 2)))
    assert backend.dispatch_stats() == {(op, "numpy"): 1}


@pytest.mark.parametrize(("var", "value"), [("ANDREY_BACKEND", "mps"), ("ANDREY_DEVICE", "metal")])
def test_pinned_mps_is_selected(fake_mps, monkeypatch, var, value):
    # A pin still reaches mps, on a cold runtime and below every cutoff.
    monkeypatch.setattr(backend, "_is_live", lambda b: False)
    monkeypatch.setenv(var, value)
    assert backend.resolve(1, 10**9) == "mps"


def test_fork_reset_frees_a_held_dispatch_lock():
    # A lock held by a parent thread at fork time has no owner in the child.
    held = backend._dispatch_lock
    held.acquire()
    token = backend._recording.set(False)
    try:
        backend._reset_after_fork()
        assert backend._dispatch_lock.acquire(timeout=1)
        backend._dispatch_lock.release()
        assert backend._recording.get() is True
    finally:
        backend._recording.reset(token)
        held.release()
