"""Backend, worker, and tie-break selection.

``backend`` sets a preference for NumPy or PyTorch and the device it runs on; ``n_jobs``
controls parallel execution; ``tie_break`` sets how the searches order tied choices.
``docs/docs/configuration.md`` describes every value. Each operation
supports a subset of the backends, including the always-available ``numpy`` reference. A backend
the operation does not support falls back to automatic selection without a warning
(:func:`dispatch_stats` records which one ran). A supported backend unavailable on the machine
warns once and uses automatic selection. PyTorch, provided by the ``torch`` extra, is imported
only when needed.

A setting resolves from the call's argument, then ``with andrey.config(...)``, then the
:data:`config` object, then ``ANDREY_BACKEND`` / ``ANDREY_N_JOBS`` / ``ANDREY_TIE_BREAK``
(read live), then the default. ``with`` uses context variables, so its overrides do not reach
threads started inside the block.
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

# What to install when a pinned-but-unavailable backend falls back (surfaced in the warning).
_UNAVAILABLE_HINT = {
    "cpu": " (torch-on-CPU; install the [torch] extra)",
    "cuda": " (needs a CUDA GPU and the [torch] extra)",
    "mps": " (needs Apple MPS and the [torch] extra)",
}

# Scoped overrides set by the ``andrey.config(...)`` context manager (thread- / task-safe).
_backend_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "andrey_backend", default=None
)
_n_jobs_var: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "andrey_n_jobs", default=None
)
_tie_break_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "andrey_tie_break", default=None
)


def _warn_once(message: str) -> None:
    warning_policy.warn_once(message, warning_policy.BackendFallbackWarning)


def _warn_former(old: str, new: str) -> None:
    """Warn once that the setting ``old`` is now ``new``."""
    warning_policy.warn_once(
        f"{old} is now {new}; the former name stops working in the next release",
        warning_policy.DeprecatedArgumentWarning,
    )


# ---- optional accelerators ----------------------------------------------------------------------


@lru_cache(maxsize=1)
def torch():
    """The ``torch`` module, or ``None`` if the ``[torch]`` extra is not installed."""
    try:
        import torch as _torch
    except ImportError:
        return None
    return _torch


def _is_available(b: str) -> bool:
    """True iff backend ``b`` can run on this machine right now."""
    if b == "numpy":
        return True
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
    """Process-wide backend, worker, and tie-break settings, also usable as a ``with`` block.

    Assigning ``andrey.config.backend``, ``andrey.config.n_jobs``, or
    ``andrey.config.tie_break`` changes the default for later calls; assigning ``None`` returns to
    ``ANDREY_BACKEND``, ``ANDREY_N_JOBS``, and ``ANDREY_TIE_BREAK``.
    ``with andrey.config(backend=..., n_jobs=..., tie_break=...):`` changes them inside the
    block only, and does not reach threads started inside it. The order, most specific first:
    ``with`` block, these attributes, the environment variables, the defaults.

    ``backend="auto"`` (the default) considers ``"cuda"`` when the operation meets its size and
    runtime-startup requirements and a compatible GPU is available; otherwise it uses ``"numpy"``.
    ``"numpy"`` uses NumPy on the CPU in float64. The ``torch`` extra provides ``"cpu"`` for
    PyTorch on the CPU, ``"cuda"`` for a CUDA-compatible GPU, and ``"mps"`` for an Apple GPU in
    unvalidated float32. ``"none"`` means ``"cpu"`` and ``"metal"`` means ``"mps"``. An explicit
    choice bypasses the size and startup requirements. An operation without the requested backend
    uses ``"auto"``; a supported but unavailable backend also warns once with
    ``BackendFallbackWarning`` before using ``"auto"``. Unknown names raise ``ValueError``.

    ``n_jobs`` counts worker processes (threads for ``granger``): ``-1`` uses every usable
    core, ``0`` or ``1`` runs serially (the default is ``1``), and larger values request that many
    workers.

    ``tie_break`` sets how a method chooses between candidates that tie, such as two moves with
    equal rounded score changes. ``"rank"`` (the default) orders them by a rank computed from each
    column's own data (:func:`andrey.search.hc.tie_ranks`), so reordering the columns relabels the
    result. ``"position"`` orders them by column position instead, which skips computing the ranks,
    and the result can then depend on the column order. ``pc`` and ``cdnod`` take their colliders'
    ``collider_order``, and ``fci`` and ``gfci`` their ``possible_dsep``, from it when the call
    gives none: ``"p_value"`` and ``"order_free"`` under ``"rank"``, ``"column"`` and ``"fast"``
    under ``"position"``. ``pc``, ``cdnod``, ``fci``, and ``gfci`` also fire their orientation
    rules one at a time in column order under ``"position"``, where ``"rank"`` runs them in
    order-free rounds. The speed difference is small, except for ``fci`` and ``gfci`` on large
    dense graphs. Each fit reads the setting once, when it starts. The
    :doc:`column-order guide </docs/guides/column-order>` gives the numbers and lists the methods
    it reaches.
    """

    _backend: str | None = None
    _n_jobs: int | None = None
    _tie_break: str | None = None

    @property
    def backend(self) -> str:
        """The backend preference in effect; assigning an unknown name raises ``ValueError``."""
        return _resolve_backend()

    @backend.setter
    def backend(self, value: str | None) -> None:
        self._backend = None if value is None else env.parse_backend(value, "config.backend")

    @property
    def n_jobs(self) -> int:
        """The worker count in effect; assigning a value below ``-1`` raises ``ValueError``."""
        return _resolve_n_jobs(None)

    @n_jobs.setter
    def n_jobs(self, value: int | None) -> None:
        self._n_jobs = None if value is None else env.parse_n_jobs(value, "config.n_jobs")

    @property
    def num_workers(self) -> int:
        """The former name of :attr:`n_jobs`, kept for one release with a warning."""
        _warn_former("andrey.config.num_workers", "andrey.config.n_jobs")
        return self.n_jobs

    @num_workers.setter
    def num_workers(self, value: int | None) -> None:
        _warn_former("andrey.config.num_workers", "andrey.config.n_jobs")
        self.n_jobs = value

    @property
    def tie_break(self) -> str:
        """The tie-break rule in effect; assigning an unknown name raises ``ValueError``."""
        return _resolve_tie_break()

    @tie_break.setter
    def tie_break(self, value: str | None) -> None:
        self._tie_break = None if value is None else env.parse_tie_break(value, "config.tie_break")

    def __call__(
        self,
        *,
        backend: str | None = None,
        n_jobs: int | None = None,
        tie_break: str | None = None,
        num_workers: int | None = None,
    ):
        """Return a context manager that sets the given settings for a ``with`` block.

        Parameters
        ----------
        backend : str or None, default=None
            Backend preference for the block; ``None`` keeps the current one.
        n_jobs : int or None, default=None
            Worker count for the block, ``-1`` or more; ``None`` keeps the current one.
        tie_break : {"rank", "position"} or None, default=None
            Tie-break rule for the block; ``None`` keeps the current one.
        num_workers : int or None, default=None
            The former name of ``n_jobs``, kept for one release with a
            :class:`~andrey.DeprecatedArgumentWarning`.

        Returns
        -------
        contextlib.AbstractContextManager
            The context manager for the ``with`` statement.

        Raises
        ------
        ValueError
            On entering the block, if ``backend`` or ``tie_break`` is not a known name or
            ``n_jobs`` is not an integer of at least ``-1``; the message lists the accepted
            values. At once, if the call passes both ``n_jobs`` and ``num_workers``.
        """
        if num_workers is not None:
            if n_jobs is not None:
                raise ValueError("num_workers is the former name of n_jobs; pass only n_jobs")
            _warn_former("andrey.config(num_workers=...)", "andrey.config(n_jobs=...)")
            n_jobs = num_workers
        return _scope(backend, n_jobs, tie_break)


config = _Config()


@contextlib.contextmanager
def _scope(backend: str | None, n_jobs: int | None, tie_break: str | None):
    b = None if backend is None else env.parse_backend(backend, "backend")
    w = None if n_jobs is None else env.parse_n_jobs(n_jobs, "n_jobs")
    r = None if tie_break is None else env.parse_tie_break(tie_break, "tie_break")
    bt = _backend_var.set(b) if b is not None else None
    wt = _n_jobs_var.set(w) if w is not None else None
    rt = _tie_break_var.set(r) if r is not None else None
    try:
        yield
    finally:
        if bt is not None:
            _backend_var.reset(bt)
        if wt is not None:
            _n_jobs_var.reset(wt)
        if rt is not None:
            _tie_break_var.reset(rt)


# ---- resolution ---------------------------------------------------------------------------------


def _resolve_backend() -> str:
    """Resolve the backend preference by precedence: context > config > env > ``auto``."""
    scoped = _backend_var.get()
    if scoped is not None:
        return scoped
    if config._backend is not None:
        return config._backend
    # Parse both, so a bad alias raises even when ANDREY_BACKEND takes precedence.
    backend, device = env.BACKEND.read(), env.DEVICE.read()
    return backend or device or "auto"


def _resolve_n_jobs(kwarg: int | None) -> int:
    """Resolve n_jobs by precedence: kwarg > context > config > env > ``1`` (serial)."""
    if kwarg is not None:
        return env.parse_n_jobs(kwarg, "n_jobs")
    scoped = _n_jobs_var.get()
    if scoped is not None:
        return scoped
    if config._n_jobs is not None:
        return config._n_jobs
    n = env.N_JOBS.read()
    if n is None:
        n = env.NUM_WORKERS.read()
        if n is not None:
            _warn_former("ANDREY_NUM_WORKERS", "ANDREY_N_JOBS")
    return 1 if n is None else n


def _resolve_tie_break() -> str:
    """Resolve the tie-break rule by precedence: context > config > env > ``rank``."""
    scoped = _tie_break_var.get()
    if scoped is not None:
        return scoped
    if config._tie_break is not None:
        return config._tie_break
    return env.TIE_BREAK.read() or "rank"


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


def select(
    pinnable: tuple[str, ...],
    auto_order: tuple[str, ...],
    *,
    metric: int | None = None,
    thresholds: dict[str, int | None] | None = None,
    amortized: bool = True,
    op: str | None = None,
) -> str:
    """Pick the backend an op should run on, from the ``pinnable`` set it supports.

    A pinned preference the op supports and is available wins (a device pin overrides the size
    threshold *and* the bootstrap rule below). A preference the op does not support falls back to
    ``auto`` silently; a supported-but-unavailable one warns once and falls back.

    ``auto`` picks the first device in ``auto_order`` that clears *its own* ``thresholds`` entry
    (when ``metric`` is given; a ``None`` cutoff means the device never auto-offloads), else the
    CPU ``numpy`` path. The cutoff is per-device because the numpy->device crossover depends on the
    device. ``amortized`` states whether the op recurs enough per fit to amortize a runtime
    bootstrap: an amortized op may be the call that imports torch and creates a CUDA context; a
    one-shot op (``amortized=False``) only takes a GPU that is already live in-process
    (:func:`_is_live`), so a single millisecond-scale call can never pay the multi-second
    torch/CUDA start-up inside a fit. ``op`` labels the decision in :func:`dispatch_stats`.
    """
    pref = _resolve_backend()
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


# ---- the executor axis: n_jobs over a process pool -----------------------------------------


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


def worker_count(n_jobs: int | None = None) -> int:
    """Resolve ``n_jobs`` to a worker count; ``-1`` means all usable CPUs."""
    n = _resolve_n_jobs(n_jobs)
    if n == -1:
        return usable_cpus()
    return max(1, n)


@lru_cache(maxsize=1)
def _thread_pools():
    from threadpoolctl import ThreadpoolController

    return ThreadpoolController()


def one_blas_thread() -> contextlib.AbstractContextManager:
    """Limit the loaded BLAS and OpenMP thread pools to one thread inside a ``with`` block.

    The libraries are found once per process; one loaded after the first call is not limited.
    """
    return _thread_pools().limit(limits=1)


# ---- tensor transfer (torch backends) -----------------------------------------------------------


def as_tensor(data: np.ndarray, dev: str):
    """A torch tensor of ``data`` on ``dev`` -- float32 on ``mps``, else float64."""
    t = torch()
    dtype = t.float32 if dev == "mps" else t.float64
    if any(stride < 0 for stride in data.strides):  # torch rejects negative strides
        data = np.ascontiguousarray(data)
    return t.as_tensor(data, dtype=dtype, device=dev)


def to_numpy(tensor) -> np.ndarray:
    """A ``float64`` numpy array from a torch tensor, whatever the compute precision."""
    return tensor.detach().cpu().numpy().astype(np.float64, copy=False)


# ---- introspection ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Backend:
    """A snapshot of the backend + worker settings and what is available on this machine."""

    backend: str
    n_jobs: int
    torch: bool
    cuda: bool
    mps: bool

    def __repr__(self) -> str:  # a compact, human line
        available = ", ".join(
            "cpu (torch)" if name == "cpu" else name
            for name, ok in self.available_backends().items()
            if ok
        )
        return f"Backend(backend={self.backend!r}, n_jobs={self.n_jobs}, available=[{available}])"

    def available_backends(self) -> dict[str, bool]:
        """Availability for each explicit ``ANDREY_BACKEND`` choice; ``cpu`` uses PyTorch."""
        return {
            "numpy": True,
            "cpu": self.torch,
            "cuda": self.cuda,
            "mps": self.mps,
        }

    def as_dict(self) -> dict:
        """The machine-readable snapshot. ``andrey config --json`` reports its ``backend`` and
        ``n_jobs`` under ``configuration`` and its ``available_backends`` unchanged."""
        return {
            "backend": self.backend,
            "n_jobs": self.n_jobs,
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
    """The backend and worker settings in effect, with package and device availability.

    Each operation picks its own implementation, so the report gives the preference, not the
    backend a given call will use. Checking whether the ``torch`` module is installed does not
    import it. GPU checks may import it on macOS or when the build supports CUDA or ROCm; an
    unrecognized build is also checked by importing it.

    Returns
    -------
    Backend
        A read-only snapshot with the fields ``backend`` (the preference, such as ``"auto"``),
        ``n_jobs`` (``-1`` means every usable core), ``torch`` (whether the module is installed
        or already imported), and ``cuda`` and ``mps`` (whether PyTorch reports a usable device).
        ``available_backends()`` maps ``numpy``, ``cpu``, ``cuda``, and ``mps`` to availability;
        ``cpu`` uses the ``torch`` field. ``as_dict()`` returns ``backend``, ``n_jobs``, and
        this availability mapping as a JSON-ready dict.

    Raises
    ------
    ValueError
        If a setting read from ``ANDREY_BACKEND``, ``ANDREY_DEVICE``, or ``ANDREY_N_JOBS``
        holds an invalid value; a context or config override takes precedence over those variables.
    """
    has_torch = _installed("torch")
    cuda_build = has_torch and ("torch" in sys.modules or _torch_gpu_build())
    return Backend(
        backend=_resolve_backend(),
        n_jobs=_resolve_n_jobs(None),
        torch=has_torch,
        cuda=cuda_build and _is_available("cuda"),
        mps=has_torch and sys.platform == "darwin" and _is_available("mps"),
    )
