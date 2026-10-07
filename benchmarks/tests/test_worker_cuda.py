"""Test worker CUDA metrics with a stubbed ``torch.cuda``.

The tests cover backend selection and the order of warmup synchronization, counter reset, and timed
fit. ``test_worker_cuda_device.py`` checks the measurements on a card.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

_argv = sys.argv
# Prevent the worker from reading a pytest argument as its start-marker path.
sys.argv = sys.argv[:2]
try:
    from andrey_bench import _worker
finally:
    sys.argv = _argv

_ADJ = np.array([[0, 1], [0, 0]], dtype=np.int8)


class _FakeCuda:
    """Log worker calls to a fake ``torch.cuda`` API."""

    def __init__(self, log, peak_bytes=0, available=True, broken=False):
        self.log = log
        self._peak = peak_bytes
        self._available = available
        self._broken = broken

    def is_available(self):
        if self._broken:
            raise RuntimeError("driver gone")
        return self._available

    def reset_peak_memory_stats(self):
        self.log.append("reset")

    def max_memory_allocated(self):
        return self._peak

    def synchronize(self):
        self.log.append("sync")


class _Adapter:
    """Provide the backend and fit method used by the worker."""

    def __init__(self, backend, log):
        self.backend = backend
        self._log = log

    def fit(self, data, params):
        self._log.append("fit")
        return _ADJ


@pytest.fixture
def cuda(monkeypatch):
    """Install a fake ``torch`` module and return its call log."""

    def install(**kw):
        log = []
        monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=_FakeCuda(log, **kw)))
        return log

    return install


def _run(backend, log):
    return _worker._run(_Adapter(backend, log), np.zeros((4, 2)), {}, warmup=1)


def test_off_cuda_reports_no_gpu_memory(cuda):
    log = cuda(peak_bytes=8 * 1024 * 1024)
    out = _run("numpy", log)
    assert out["gpu_peak_alloc_mb"] is None
    assert log == ["fit", "fit"]


def test_peak_is_reported_in_mib(cuda):
    log = cuda(peak_bytes=3 * 1024 * 1024)
    assert _run("torch-cuda", log)["gpu_peak_alloc_mb"] == pytest.approx(3.0)


def test_warmup_is_drained_then_the_counter_is_reset_before_the_timed_fit(cuda):
    log = cuda(peak_bytes=1024)
    _run("torch-cuda", log)
    assert log == ["fit", "sync", "reset", "fit", "sync"]


@pytest.mark.parametrize("backend", ["native", "numba", "numpy", "torch-cpu"])
def test_a_cpu_backend_ignores_an_exported_cuda_device(cuda, monkeypatch, backend):
    monkeypatch.setenv("ANDREY_DEVICE", "cuda")
    log = cuda(peak_bytes=2 * 1024 * 1024)
    out = _run(backend, log)
    assert out["gpu_peak_alloc_mb"] is None
    assert out["device_id"] == "cpu"
    assert "reset" not in log


def test_an_unknown_backend_falls_back_to_the_environment(cuda, monkeypatch):
    monkeypatch.setenv("ANDREY_DEVICE", "cuda")
    log = cuda(peak_bytes=2 * 1024 * 1024)
    out = _run("some-future-backend", log)
    assert out["gpu_peak_alloc_mb"] == pytest.approx(2.0)


def test_unavailable_cuda_reports_no_memory_and_does_not_reset(cuda):
    log = cuda(peak_bytes=99, available=False)
    out = _run("torch-cuda", log)
    assert out["gpu_peak_alloc_mb"] is None
    assert "reset" not in log


def test_a_broken_cuda_api_leaves_the_fit_intact(cuda):
    log = cuda(broken=True)
    out = _run("torch-cuda", log)
    assert out["gpu_peak_alloc_mb"] is None
    assert out["adj"] is not None
