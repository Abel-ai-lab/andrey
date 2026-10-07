"""Test worker CUDA telemetry on a real card: the reported device and the allocation size.

Backend selection and the timed-fit scope of the peak are in :mod:`test_worker_cuda`, on a stub.

The module skips without PyTorch or CUDA unless ``ANDREY_REQUIRE_GPU=1`` requires the GPU lane.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_argv = sys.argv
# Prevent the worker from reading a pytest argument as its start-marker path.
sys.argv = sys.argv[:2]
try:
    from andrey_bench import _worker
finally:
    sys.argv = _argv

_TRUTHY = {"1", "true", "yes", "on"}
_REQUIRED = os.environ.get("ANDREY_REQUIRE_GPU", "").strip().lower() in _TRUTHY

try:
    import torch
except ImportError:
    if _REQUIRED:
        pytest.fail("ANDREY_REQUIRE_GPU is set but torch is not installed", pytrace=False)
    pytest.skip("torch is not installed", allow_module_level=True)

if not torch.cuda.is_available():
    if _REQUIRED:
        raise RuntimeError("ANDREY_REQUIRE_GPU is set but torch.cuda.is_available() is False")
    pytest.skip("no CUDA device", allow_module_level=True)

_ADJ = np.array([[0, 1], [0, 0]], dtype=np.int8)
_MIB = 1024.0 * 1024.0


class _Allocating:
    """Allocate a CUDA tensor of the configured size in every fit."""

    def __init__(self, backend: str, mib: float):
        self.backend = backend
        self._mib = mib

    def fit(self, data, params):
        block = torch.empty(int(self._mib * _MIB // 4), dtype=torch.float32, device="cuda")
        del block
        return _ADJ


def _run(adapter):
    return _worker._run(adapter, np.zeros((4, 2)), {}, warmup=1)


def test_a_pinned_cuda_fit_reports_the_card_and_its_allocation():
    out = _run(_Allocating("torch-cuda", mib=16))
    device = torch.cuda.current_device()
    assert out["device_id"].startswith("cuda:")
    assert torch.cuda.get_device_name(device) in out["device_id"]
    assert out["gpu_peak_alloc_mb"] == pytest.approx(16, rel=0.1)
