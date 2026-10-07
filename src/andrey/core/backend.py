"""Backend and worker selection for the core primitives.

``backend`` chooses the numeric kernels and ``num_workers`` the process pool;
``docs/docs/configuration.md`` describes every value. Each op implements some
backends over an always-present numpy reference. A backend the op lacks falls back silently
(:func:`dispatch_stats` records which one ran); a backend unavailable on the machine warns once
and falls back. The ``torch`` and ``numba`` extras are imported lazily.

A setting resolves from the call's argument, then ``with andrey.config(...)``, then the
:data:`config` object, then ``ANDREY_BACKEND`` / ``ANDREY_NUM_WORKERS`` (read live), then the
default. ``with`` uses context variables, so its overrides do not reach threads started inside the
block.
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib.util
import logging
import os
import sys
import threading
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from andrey.core import warning_policy

from . import env

logger = logging.getLogger(__name__)

# Per-op auto-preference orders (best first) and the CPU fallback.
_TORCH_PINNABLE = ("numpy", "cpu", "cuda", "mps")
_TORCH_AUTO = ("cuda", "numpy")  # cpu-torch and mps are pin-only, never auto-selected
_BITSET_PINNABLE = ("numpy", "numba")
_BITSET_AUTO = ("numba", "numpy")

# What to install when a pinned-but-unavailable backend falls back (surfaced in the warning).
_UNAVAILABLE_HINT = {
    "numba": " (install the [numba] extra)",
    "cpu": " (torch-on-CPU; install the [torch] extra)",
    "cuda": " (needs a CUDA GPU and the [torch] extra)",
    "mps": " (needs Apple MPS and the [torch] extra)",
}

# Scoped overrides set by the ``andrey.config(...)`` context manager (thread- / task-safe).
_backend_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "andrey_backend", default=None
)
_num_workers_var: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "andrey_num_workers", default=None
)


def _reset_warnings() -> None:
    """Clear the package's warn-once memory (test isolation)."""
    warning_policy.reset()


def _warn_once(message: str) -> None:
    warning_policy.warn_once(message, warning_policy.BackendFallbackWarning)


# ---- optional accelerators ----------------------------------------------------------------------


@lru_cache(maxsize=1)
def torch():
    """The ``torch`` module, or ``None`` if the ``[torch]`` extra is not installed."""
    try:
        import torch as _torch
    except ImportError:
        return None
    return _torch


@lru_cache(maxsize=1)
def numba():
    """The ``numba`` module, or ``None`` if the ``[numba]`` extra is not installed."""
    try:
        import numba as _numba
    except ImportError:
        return None
    return _numba


def _is_available(b: str) -> bool:
    """True iff backend ``b`` can run on this machine right now."""
    if b == "numpy":
        return True
    if b == "numba":
        return numba() is not None
    t = torch()
    if t is None:
        return False
    if b == "cpu":
        return True
    if b == "cuda":
        return bool(t.cuda.is_available())
    if b == "mps":
        mps = getattr(t.backends, "mps", None)
        return mps is not None and bool(mps.is_available())
    return False


def _is_live(b: str) -> bool:
    """Return whether backend ``b`` already has a live runtime in this process.

    Torch must be present in ``sys.modules``, and the device runtime must have an initialized CUDA
    context or an active Metal allocation. The check does not import torch or touch the driver.
    """
    t = sys.modules.get("torch")
    if t is None:
        return False
    try:
        if b == "cuda":
            return bool(t.cuda.is_initialized())
        if b == "mps":
            return bool(t.mps.is_available() and t.mps.current_allocated_memory() > 0)
    except Exception:  # noqa: BLE001 - a probe of optional state must not break dispatch
        return False
    return True


# ---- the config object + context manager --------------------------------------------------------


class _Config:
    """Process-wide backend and worker settings, also usable as a ``with`` block.

    Set ``andrey.config.backend`` or ``andrey.config.num_workers`` to change the default for every
    later call, or assign ``None`` to return to ``ANDREY_BACKEND`` and ``ANDREY_NUM_WORKERS``.
    ``with andrey.config(backend=..., num_workers=...):`` changes them inside the block only, and
    does not reach threads started inside it. The order, most specific first: ``with`` block,
    these attributes, the environment variables, the defaults.

    ``backend`` is a preference: ``"auto"`` (the default), ``"numpy"``, ``"numba"``, ``"cuda"``,
    ``"mps"``, or ``"cpu"`` (torch on the CPU); ``"none"`` means ``"cpu"`` and ``"metal"`` means
    ``"mps"``. An operation without the requested backend uses its default, and an unavailable
    backend warns once with ``BackendFallbackWarning``. ``num_workers`` counts worker processes:
    ``-1`` uses every usable core, and ``0`` or ``1`` runs serially (the default).
    """

    _backend: str | None = None
    _num_workers: int | None = None

    @property
    def backend(self) -> str:
        """The backend preference in effect; assigning an unknown name raises ``ValueError``."""
        return _resolve_backend(None)

    @backend.setter
    def backend(self, value: str | None) -> None:
        self._backend = None if value is None else env.parse_backend(value, "config.backend")

    @property
    def num_workers(self) -> int:
        """The worker count in effect; assigning a value below ``-1`` raises ``ValueError``."""
        return _resolve_num_workers(None)

    @num_workers.setter
    def num_workers(self, value: int | None) -> None:
        self._num_workers = (
            None if value is None else env.parse_num_workers(value, "config.num_workers")
        )

    def __call__(self, *, backend: str | None = None, num_workers: int | None = None):
        """Return a context manager that sets ``backend`` and ``num_workers`` for a ``with`` block.

        Parameters
        ----------
        backend : str or None, default=None
            Backend preference for the block; ``None`` keeps the current one.
        num_workers : int or None, default=None
            Worker count for the block, ``-1`` or more; ``None`` keeps the current one.

        Returns
        -------
        contextlib.AbstractContextManager
            The context manager for the ``with`` statement.

        Raises
        ------
        ValueError
            On entering the block, if ``backend`` is not a known name or ``num_workers`` is not an
            integer of at least ``-1``.
        """
        return _scope(backend, num_workers)


config = _Config()


@contextlib.contextmanager
def _scope(backend: str | None, num_workers: int | None):
    b = None if backend is None else env.parse_backend(backend, "backend")
    w = None if num_workers is None else env.parse_num_workers(num_workers, "num_workers")
    bt = _backend_var.set(b) if b is not None else None
    wt = _num_workers_var.set(w) if w is not None else None
    try:
        yield
    finally:
        if bt is not None:
            _backend_var.reset(bt)
        if wt is not None:
            _num_workers_var.reset(wt)


# ---- resolution ---------------------------------------------------------------------------------


def _resolve_backend(kwarg: str | None) -> str:
    """Resolve the backend preference by precedence: kwarg > context > config > env > ``auto``."""
    if kwarg is not None:
        return env.parse_backend(kwarg, "backend")
    scoped = _backend_var.get()
    if scoped is not None:
        return scoped
    if config._backend is not None:
        return config._backend
    # Parse both, so a bad alias raises even when ANDREY_BACKEND takes precedence.
    backend, device = env.BACKEND.read(), env.DEVICE.read()
    return backend or device or "auto"


def _resolve_num_workers(kwarg: int | None) -> int:
    """Resolve num_workers by precedence: kwarg > context > config > env > ``1`` (serial)."""
    if kwarg is not None:
        return env.parse_num_workers(kwarg, "num_workers")
    scoped = _num_workers_var.get()
    if scoped is not None:
        return scoped
    if config._num_workers is not None:
        return config._num_workers
    n = env.NUM_WORKERS.read()
    return 1 if n is None else n


# ---- the dispatch counts: every select() decision, recorded per process -------------------------

_dispatch_lock = threading.Lock()
_dispatch_counts: Counter[tuple[str, str]] = Counter()


_recording = contextvars.ContextVar("andrey_dispatch_recording", default=True)


def _reset_after_fork() -> None:
    """Give a forked child an unlocked dispatch-count lock and enable recording.

    A child forked while another thread holds the lock inherits it locked, with no thread left to
    release it. Without this reset, a child forked inside :func:`unrecorded_dispatch` would keep
    recording disabled.
    """
    global _dispatch_lock
    _dispatch_lock = threading.Lock()
    _recording.set(True)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


@contextlib.contextmanager
def unrecorded_dispatch():
    """Keep decisions made inside this block out of :func:`dispatch_stats`.

    Calibration times real operations, but those probes are not user dispatches and would distort
    the counts.
    """
    token = _recording.set(False)
    try:
        yield
    finally:
        _recording.reset(token)


def _record_dispatch(op: str | None, chosen: str) -> str:
    if op is not None and _recording.get():
        with _dispatch_lock:
            _dispatch_counts[(op, chosen)] += 1
    return chosen


def dispatch_stats() -> dict[tuple[str, str], int]:
    """A snapshot of this process's dispatch decisions: ``{(op, backend): count}``.

    Every labeled :func:`select` call is counted. For example,
    ``dispatch_stats()[("entropy", "cuda")]`` is the number of offloaded entropy calls. Each
    process keeps its own counts.
    """
    with _dispatch_lock:
        return dict(_dispatch_counts)


def reset_dispatch_stats() -> None:
    """Clear the dispatch counts (test isolation / scoping a measurement)."""
    with _dispatch_lock:
        _dispatch_counts.clear()


def select(
    pinnable: tuple[str, ...],
    auto_order: tuple[str, ...],
    *,
    metric: int | None = None,
    thresholds: dict[str, int | None] | None = None,
    backend: str | None = None,
    amortized: bool = True,
    op: str | None = None,
) -> str:
    """Pick the backend an op should run on, from the ``pinnable`` set it supports.

    ``backend`` overrides the active preference for this call. A pinned preference the op supports
    and is available wins (a device pin overrides the size threshold *and* the bootstrap rule
    below). A preference the op does not support falls back to ``auto`` silently; a
    supported-but-unavailable one warns once and falls back.

    ``auto`` picks the first device in ``auto_order`` that clears *its own* ``thresholds`` entry
    (when ``metric`` is given; a ``None`` cutoff means the device never auto-offloads), else the
    CPU ``numpy`` path. The cutoff is per-device because the numpy->device crossover depends on the
    device. ``amortized`` states whether the op recurs enough per fit to amortize a runtime
    bootstrap: an amortized op may be the call that imports torch and creates a CUDA context; a
    one-shot op (``amortized=False``) only takes a GPU that is already live in-process
    (:func:`_is_live`), so a single millisecond-scale call can never pay the multi-second
    torch/CUDA start-up inside a fit. ``op`` labels the decision in :func:`dispatch_stats`.
    """
    pref = _resolve_backend(backend)
    if pref != "auto":
        if pref in pinnable:
            if _is_available(pref):
                return _record_dispatch(op, pref)
            _warn_once(
                f"backend {pref!r} is unavailable{_UNAVAILABLE_HINT.get(pref, '')}; "
                "falling back to auto-selection"
            )
        # unsupported by this op -> fall through to auto (silent, by design)
    for b in auto_order:
        if b == "numpy":
            return _record_dispatch(op, "numpy")
        if metric is not None and thresholds is not None:
            cutoff = thresholds.get(b)  # None -> no measured cutoff: never auto-offload to b
            if cutoff is None or metric < cutoff:
                continue  # too small for this device (or unmeasured) -> try the next tier
        # Order matters: the liveness check is free; _is_available may import torch. A one-shot op
        # checks liveness first so the *decision* to stay on numpy never imports anything.
        if b in ("cuda", "mps") and not amortized and not _is_live(b):
            continue
        if not _is_available(b):
            continue
        return _record_dispatch(op, b)
    return _record_dispatch(op, "numpy")


def resolve(
    metric: int,
    threshold: int | dict[str, int | None],
    *,
    amortized: bool = False,
    op: str | None = None,
) -> str:
    """Pick ``numpy`` / ``cpu`` / ``cuda`` / ``mps`` for a float torch op of size ``metric``.

    ``auto`` considers only ``cuda``; ``cpu`` and ``mps`` need a pin. ``threshold`` is either one
    cutoff or a ``{device: cutoff}`` mapping. A ``None`` cutoff disables automatic offload for that
    device. A pinned backend bypasses the cutoff. Set ``amortized=True`` for repeated operations
    that may bootstrap the GPU runtime; the default protects one-shot operations from startup cost.
    """
    thresholds: dict[str, int | None] = (
        threshold
        if isinstance(threshold, dict)
        else {b: threshold for b in _TORCH_AUTO if b != "numpy"}
    )
    dev = select(
        _TORCH_PINNABLE,
        _TORCH_AUTO,
        metric=metric,
        thresholds=thresholds,
        amortized=amortized,
        op=op,
    )
    logger.debug(
        "backend: %s (op=%s, metric=%d, amortized=%s, thresholds=%s)",
        dev,
        op,
        metric,
        amortized,
        thresholds,
    )
    return dev


def resolve_bitset(*, backend: str | None = None) -> str:
    """Pick ``numba`` or ``numpy`` for the bitset kernels (numba when available, else numpy)."""
    return select(_BITSET_PINNABLE, _BITSET_AUTO, backend=backend, op="bitset")


# ---- the executor axis: num_workers over a process pool -----------------------------------------


# CPU limits on Linux can come from both scheduler affinity and cgroups.
#
# cgroups have two layouts:
#
#   v2 (modern Linux / Kubernetes)
#       One unified hierarchy. CPU quota is stored in:
#           cpu.max -> "<quota> <period>" or "max <period>"
#
#   v1 (older Linux / many SLURM clusters)
#       Separate controller hierarchies. CPU quota is stored in:
#           cpu.cfs_quota_us
#           cpu.cfs_period_us
#       A quota of -1 means unlimited.
#
# A process may live below the cgroup mount root, and limits inherited from any
# parent apply. So the lookup starts at the process's own cgroup path and walks
# upward, taking the tightest quota found.


def _cgroup_rel_paths() -> tuple[str, str]:
    """Return this process's cgroup paths as ``(v2, v1_cpu)``, empty for the unused generation."""
    v2 = v1 = ""
    try:
        with open("/proc/self/cgroup") as fh:
            for line in fh:
                # v2 writes a single ``0::/path`` line; v1 writes one line per controller group,
                # for example, ``4:cpu,cpuacct:/path``.
                hid, controllers, path = line.strip().split(":", 2)
                if hid == "0":
                    v2 = path
                elif "cpu" in controllers.split(","):
                    v1 = path
    except (OSError, ValueError):
        pass
    return v2, v1


def _quota_v2(d: str) -> int | None:
    """Read a cgroup v2 CPU quota, or ``None`` if uncapped/unavailable."""
    try:
        with open(os.path.join(d, "cpu.max")) as fh:
            quota, period = fh.read().split()
        return None if quota == "max" else max(1, int(int(quota) / int(period)))
    except (OSError, ValueError, ZeroDivisionError):
        return None


def _quota_v1(d: str) -> int | None:
    """Read a cgroup v1 CPU quota, or ``None`` if uncapped/unavailable."""
    try:
        with open(os.path.join(d, "cpu.cfs_quota_us")) as fh:
            quota = int(fh.read())
        with open(os.path.join(d, "cpu.cfs_period_us")) as fh:
            period = int(fh.read())
    except (OSError, ValueError):
        return None
    return max(1, int(quota / period)) if quota > 0 and period > 0 else None


def _tightest_quota(root: str, rel: str, read) -> int | None:
    """Return the tightest quota between the process cgroup and mount root."""
    best = None
    parts = [p for p in rel.split("/") if p]
    while True:
        n = read(os.path.join(root, *parts))
        if n is not None:
            best = n if best is None else min(best, n)
        if not parts:
            return best
        parts.pop()


def _cgroup_cpu_quota() -> int | None:
    """Return the CPU quota imposed by cgroups, or ``None`` if uncapped."""
    v2, v1 = _cgroup_rel_paths()
    for root, rel, read in (
        ("/sys/fs/cgroup", v2, _quota_v2),
        ("/sys/fs/cgroup/cpu", v1, _quota_v1),
    ):
        n = _tightest_quota(root, rel, read)
        if n is not None:
            return n
    return None


def usable_cpus() -> int:
    """Return the CPUs this process can actually use.

    Uses the tightest applicable limit from machine CPU count, scheduler affinity, and cgroup CPU
    quota.
    """
    limits = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        limits.append(len(os.sched_getaffinity(0)))
    quota = _cgroup_cpu_quota()
    if quota is not None:
        limits.append(quota)
    return max(1, min(limits))


def worker_count(num_workers: int | None = None) -> int:
    """Resolve ``num_workers`` to a worker count; ``-1`` means all usable CPUs."""
    n = _resolve_num_workers(num_workers)
    if n == -1:
        return usable_cpus()
    return max(1, n)


def parallel_map(fn, items, *, num_workers: int | None = None) -> list:
    """Map ``fn`` over ``items`` serially or with a process pool.

    ``num_workers <= 1`` runs serially; ``-1`` uses all usable CPUs. Input order is preserved.
    ``fn`` and ``items`` must be picklable.
    """
    n = worker_count(num_workers)
    items = list(items)
    if n <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    import concurrent.futures as cf

    with cf.ProcessPoolExecutor(max_workers=min(n, len(items))) as pool:
        return list(pool.map(fn, items))


# ---- tensor transfer (torch backends) -----------------------------------------------------------


def as_tensor(data: np.ndarray, dev: str):
    """A torch tensor of ``data`` on ``dev`` -- float32 on ``mps``, else float64."""
    t = torch()
    dtype = t.float32 if dev == "mps" else t.float64
    return t.as_tensor(data, dtype=dtype, device=dev)


def to_numpy(tensor) -> np.ndarray:
    """A ``float64`` numpy array from a torch tensor, whatever the compute precision."""
    return tensor.detach().cpu().numpy().astype(np.float64, copy=False)


# ---- introspection ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Backend:
    """A snapshot of the backend + worker settings and what is available on this machine."""

    backend: str
    num_workers: int
    numba: bool
    torch: bool
    cuda: bool
    mps: bool

    def __repr__(self) -> str:  # a compact, human line
        available = ", ".join(
            "cpu (torch)" if name == "cpu" else name
            for name, ok in self.available_backends().items()
            if ok
        )
        return (
            f"Backend(backend={self.backend!r}, num_workers={self.num_workers}, "
            f"available=[{available}])"
        )

    def available_backends(self) -> dict[str, bool]:
        """Availability keyed by the ``ANDREY_BACKEND`` values: ``cpu`` is torch on the CPU."""
        return {
            "numpy": True,
            "numba": self.numba,
            "cpu": self.torch,
            "cuda": self.cuda,
            "mps": self.mps,
        }

    def as_dict(self) -> dict:
        """The machine-readable snapshot. ``andrey config --json`` reports its ``backend`` and
        ``num_workers`` under ``configuration`` and its ``available_backends`` unchanged."""
        return {
            "backend": self.backend,
            "num_workers": self.num_workers,
            "available_backends": self.available_backends(),
        }


def _installed(name: str) -> bool:
    """Whether module ``name`` is imported or can be found, without importing it."""
    if sys.modules.get(name) is not None:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _torch_gpu_build() -> bool:
    """Whether the installed torch was built for CUDA or ROCm, read from ``torch/version.py``.

    Runs that one generated file, not the ``torch`` package. A missing or unexpected file counts as
    a GPU build.
    """
    spec = importlib.util.find_spec("torch")
    locations = spec.submodule_search_locations if spec is not None else None
    if not locations:
        return True
    version = importlib.util.spec_from_file_location(
        "_andrey_torch_version", os.path.join(locations[0], "version.py")
    )
    if version is None or version.loader is None:
        return True
    module = importlib.util.module_from_spec(version)
    try:
        version.loader.exec_module(module)
    except Exception:  # noqa: BLE001 - an unknown build is checked by importing torch
        return True
    return getattr(module, "cuda", True) is not None or getattr(module, "hip", True) is not None


def describe() -> Backend:
    """Report the backend and worker settings in effect, and the backends this machine can run.

    Each operation picks its own implementation, so the report gives the preference, not the
    backend a given call will use. Checking for numba and torch does not import them; torch is
    imported only to look for a GPU, on a CUDA or ROCm build of torch or on macOS.

    Returns
    -------
    Backend
        A read-only snapshot with the fields ``backend`` (the preference, such as ``"auto"``),
        ``num_workers`` (``-1`` means every usable core), and ``numba``, ``torch``, ``cuda``, and
        ``mps`` (``True`` when usable here). ``available_backends()`` maps each ``backend`` value
        to its availability, and ``as_dict()`` returns the values as a JSON-ready dict.

    Raises
    ------
    ValueError
        If ``ANDREY_BACKEND``, ``ANDREY_DEVICE``, or ``ANDREY_NUM_WORKERS`` holds an invalid value.
    """
    has_torch = _installed("torch")
    cuda_build = has_torch and ("torch" in sys.modules or _torch_gpu_build())
    return Backend(
        backend=_resolve_backend(None),
        num_workers=_resolve_num_workers(None),
        numba=_installed("numba"),
        torch=has_torch,
        cuda=cuda_build and _is_available("cuda"),
        mps=has_torch and sys.platform == "darwin" and _is_available("mps"),
    )
