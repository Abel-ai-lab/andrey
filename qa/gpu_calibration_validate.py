"""Validate auto-dispatch calibration on an accelerator.

Run ``uv run --no-sync python qa/gpu_calibration_validate.py`` in an environment with a
driver-matched torch build (see "PyTorch for a GPU" in the getting-started guide); plain
``uv run`` re-syncs the locked CPU torch first. The script exits with an error when no accelerator
is available.

For each operation, the report shows the measured crossover next to this device's shipped default,
so the user can decide whether to set a cutoff. It also validates the disk-cache round trip and
confirms that the device runtime is live. KCI and CMIknn have no runtime calibration: their
crossovers set `ANDREY_KCI_GPU_THRESHOLD` and `ANDREY_CMIKNN_GPU_THRESHOLD`, both in rows squared.
"""

from __future__ import annotations

import math
import os
import tempfile
import time

import numpy as np

from andrey.core import backend, ci_cmiknn, ci_kernel, stats

# Past this factor from the shipped default, the report suggests setting a cutoff for this machine.
_TOLERANCE = 4.0


def _ci_probe(metric: int) -> np.ndarray:
    """Columns ``x, y, z1, z2`` on ``isqrt(metric)`` rows: ``x`` and ``y`` nonlinear in ``z``."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((math.isqrt(metric), 2))
    s = z.sum(axis=1)
    noise = rng.standard_normal((z.shape[0], 2))
    return np.column_stack([np.tanh(s) + 0.4 * noise[:, 0], s**2 + 0.4 * noise[:, 1], z])


def _kci_probe(metric: int) -> ci_kernel.KCI:
    return ci_kernel.KCI(_ci_probe(metric))


def _cmiknn_probe(metric: int) -> ci_cmiknn.CMIknn:
    # 100 shuffles keep a probe short; the per-shuffle work, which sets the crossover, is the same.
    return ci_cmiknn.CMIknn(_ci_probe(metric), n_permutations=100)


def _query(test) -> float:
    return test(0, 1, [2, 3])


def _device() -> str | None:
    for dev in ("cuda", "mps"):
        if backend._is_available(dev):
            return dev
    return None


def _report(op: str, device: str, make, call, start: int, default: int | None) -> bool:
    """Report one crossover beside the shipped default; return whether one was found."""
    began = time.perf_counter()
    got = stats._crossover(device, make, call, start)
    took = time.perf_counter() - began
    if got is None:
        print(f"{op:>9}: the device never won up to the ceiling ({took:.1f}s)")
        return False
    if default is None:
        shipped, advice = "none", "no shipped default; set a cutoff to offload"
    else:
        ratio = got / default
        shipped = f"{default:,}"
        advice = f"{ratio:.2f}x the default"
        if not 1 / _TOLERANCE <= ratio <= _TOLERANCE:
            advice += "; consider setting a cutoff"
    print(
        f"{op:>9}: found {got:>12,} elements | shipped default {shipped} | {advice} | "
        f"below start: {'yes' if got < start else 'no'} | {took:.1f}s"
    )
    # Crossovers differ by machine, so only finding one is asserted, never its value.
    return True


def main() -> None:
    device = _device()
    if device is None:
        # A CPU-only torch build validates no accelerator path, so the run must fail.
        t = backend.torch()
        raise SystemExit(
            "no accelerator available; nothing was validated. "
            f"torch={'absent' if t is None else t.__version__}"
        )
    name = ""
    t = backend.torch()
    if device == "cuda" and t is not None:
        name = t.cuda.get_device_name(0)
    print(f"device: {device} ({name or 'unnamed'})")
    print(f"fingerprint: {stats._device_fingerprint(device)}")
    print(
        f"shipped defaults: entropy={stats._ENTROPY_GPU_THRESHOLDS[device]!r} "
        f"cov={stats._COV_GPU_THRESHOLDS[device]!r}\n"
    )

    ok = _report(
        "entropy",
        device,
        stats._entropy_probe,
        stats.entropy,
        stats._ENTROPY_PROBE_START,
        stats._ENTROPY_GPU_THRESHOLDS[device],
    )
    ok &= _report(
        "cov",
        device,
        stats._cov_probe,
        stats._cov_probe_call,
        stats._COV_PROBE_START,
        stats._COV_GPU_THRESHOLDS[device],
    )

    if device == "cuda":  # KCI and CMIknn have no mps backend
        for op, make, default in (
            ("kci", _kci_probe, ci_kernel._KCI_GPU_THRESHOLDS["cuda"]),
            ("cmiknn", _cmiknn_probe, ci_cmiknn._CMIKNN_GPU_THRESHOLDS["cuda"]),
        ):
            ok &= _report(op, device, make, _query, 250_000, default)

    # Use an isolated cache to exercise measure, persist, clear, and reload without affecting the
    # user's calibration data.
    cache = tempfile.mkdtemp(prefix="andrey-calib-")
    os.environ["XDG_CACHE_HOME"] = cache
    print(f"\ncache round-trip under {cache}")
    stats._CALIBRATED.clear()
    os.environ["ANDREY_GPU_CALIBRATE"] = "1"
    first = stats._calibrate(
        "entropy",
        device,
        stats._entropy_probe,
        stats.entropy,
        stats._ENTROPY_PROBE_START,
        stats._ENTROPY_GPU_THRESHOLDS[device],
    )
    stats._CALIBRATED.clear()
    began = time.perf_counter()
    second = stats._calibrate(
        "entropy",
        device,
        stats._entropy_probe,
        stats.entropy,
        stats._ENTROPY_PROBE_START,
        stats._ENTROPY_GPU_THRESHOLDS[device],
    )
    reuse_s = time.perf_counter() - began
    reused = reuse_s < 1.0 and second == first
    print(
        f"\ndisk cache: first={first!r} second={second!r} "
        f"({'reused' if reused else 'RE-MEASURED'} in {reuse_s:.2f}s)"
    )
    ok &= reused

    # Calibration leaves the device runtime live for one-shot dispatch.
    live = backend._is_live(device)
    print(f"{device} live in-process: {live}")
    print(f"dispatch counts: {backend.dispatch_stats()}")
    ok &= live

    # Return a failing process status when any required property is false.
    if not ok:
        raise SystemExit("validation FAILED: see the lines above")
    print("\nvalidation passed: crossovers found, cache reused, device live")


if __name__ == "__main__":
    main()
