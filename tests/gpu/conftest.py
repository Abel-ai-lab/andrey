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
    """The device to validate, or ``None`` to skip device parity.

    The pin is ``ANDREY_BACKEND``, else ``ANDREY_DEVICE``. Both are read as the package reads
    them, so an unknown value in either raises ``ValueError``. A ``cuda`` or ``mps`` pin gives that
    device, or ``None`` when it is unavailable; a ``cpu`` or ``numpy`` pin gives ``None`` rather
    than override the pin. With no pin, the result is whichever accelerator is available.
    """
    backend_pin, device_pin = env.BACKEND.read(), env.DEVICE.read()
    requested = backend_pin or device_pin
    if requested in ("cuda", "mps"):
        return requested if backend._is_available(requested) else None
    if requested in ("cpu", "numpy"):
        return None
    for dev in ("cuda", "mps"):
        if backend._is_available(dev):
            return dev
    return None


def pytest_configure(config) -> None:
    """Fail the whole session (not skip) when a GPU is required but none is present."""
    try:
        missing = env.REQUIRE_GPU.read() and accelerator() is None
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    if missing:
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
def _deterministic():
    """Deterministic torch kernels for the on-device checks.

    ``use_deterministic_algorithms`` is process-global, so its prior value is saved and restored
    (a full local ``pytest`` run must not leak it into the ``tests/unit`` torch tests).
    """
    torch = backend.torch()
    prev = torch.are_deterministic_algorithms_enabled() if torch is not None else None
    if torch is not None:
        torch.use_deterministic_algorithms(True)
    yield
    if torch is not None:
        torch.use_deterministic_algorithms(prev)
