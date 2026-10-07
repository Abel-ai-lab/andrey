"""Fixtures for the GPU device-parity suite.

The suite is meaningful only on real GPU hardware, and a subtle failure mode is a *silent* green:
``backend.select`` warns-and-falls-back to numpy when a pinned device is absent, so a
parity-vs-numpy check passes trivially with no GPU -- pytest exits 0 when every test *skips*. So:

- ``ANDREY_REQUIRE_GPU=1`` (set on GPU runs) turns "no accelerator" from a skip into a hard
  error, so a mis-provisioned runner fails loudly instead of passing green. Unset (local default),
  the suite skips cleanly -- ``testpaths = ["tests"]`` means a plain ``pytest`` collects it too.
- ``CUBLAS_WORKSPACE_CONFIG`` must be set before the first cuBLAS call for
  ``torch.use_deterministic_algorithms(True)`` to hold; conftest is imported before the test modules
  (and thus before torch), so setting it here is early enough.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest  # noqa: E402

from andrey.core import backend, env  # noqa: E402


def accelerator() -> str | None:
    """The device to validate: an explicit ``ANDREY_DEVICE`` (cuda/mps) if present, else whichever
    accelerator is available, else ``None`` (also ``None`` if a requested device is unavailable).
    """
    requested = os.environ.get("ANDREY_DEVICE", "").strip().lower()
    requested = {"metal": "mps"}.get(requested, requested)
    if requested in ("cpu", "numpy", "none"):
        return None  # an explicit non-accelerator pin: skip device-parity rather than override it
    if requested in ("cuda", "mps"):
        return requested if backend._is_available(requested) else None
    for dev in ("cuda", "mps"):
        if backend._is_available(dev):
            return dev
    return None


def pytest_configure(config) -> None:
    """Fail the whole session (not skip) when a GPU is required but none is present."""
    try:
        require = env.REQUIRE_GPU.read()
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    if require and accelerator() is None:
        raise pytest.UsageError(
            "ANDREY_REQUIRE_GPU is set but no CUDA/MPS accelerator is available "
            "(torch missing, or torch.cuda.is_available() is False)"
        )


@pytest.fixture
def device() -> str:
    dev = accelerator()
    if dev is None:
        pytest.skip("no CUDA/MPS accelerator on this machine")
    return dev


@pytest.fixture(autouse=True)
def _deterministic_and_isolated():
    """Deterministic torch kernels for the on-device checks; isolate process-level backend state.

    ``use_deterministic_algorithms`` is process-global, so its prior value is saved and restored
    (a full local ``pytest`` run must not leak it into the ``tests/unit`` torch tests). The backend
    reset mirrors ``tests/unit/conftest.py`` -- that autouse reset is dir-scoped, and ``tests/gpu``
    sorts first, so without this a pinned backend or warn-once entry would leak into later lanes.
    """
    torch = backend.torch()
    prev = torch.are_deterministic_algorithms_enabled() if torch is not None else None
    if torch is not None:
        torch.use_deterministic_algorithms(True)
    yield
    if torch is not None:
        torch.use_deterministic_algorithms(prev)
    backend.config._backend = None
    backend.config._num_workers = None
    backend._reset_warnings()
