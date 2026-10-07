"""Run one benchmark repeat in a child process and write its result to disk."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import pickle
import resource
import sys
import threading
import time

# Write the start marker before importing numpy: the parent reads its existence to tell a fit
# that ran out of time from a step that never started, and a cold-filesystem numpy import can
# outlast a short cap.
_STARTED_PATH = sys.argv[2] if len(sys.argv) > 2 else None
if _STARTED_PATH:
    try:
        open(_STARTED_PATH, "wb").close()
    except OSError:
        pass

import numpy as np  # noqa: E402 - write the start marker first

#: Bytes per MiB used by the memory metrics.
_MIB = 1024.0 * 1024.0

#: Clock ticks per second for converting ``/proc/<pid>/stat`` values.
_SC_CLK_TCK = os.sysconf("SC_CLK_TCK")


def _descendants(pid: int) -> list[int]:
    """Return all descendant PIDs of ``pid``."""
    seen: list[int] = []
    visited = {pid}
    frontier = [pid]
    children_iface_worked = False
    while frontier:
        cur = frontier.pop()
        try:
            tids = os.listdir(f"/proc/{cur}/task")
        except OSError:
            continue
        for tid in tids:
            try:
                with open(f"/proc/{cur}/task/{tid}/children") as fh:
                    kids = [int(x) for x in fh.read().split()]
                children_iface_worked = True
            except OSError:
                continue
            for k in kids:
                if k not in visited:
                    visited.add(k)
                    seen.append(k)
                    frontier.append(k)
    if seen or children_iface_worked:
        return seen
    return _descendants_via_ppid_scan(pid)


def _descendants_via_ppid_scan(root: int) -> list[int]:
    """Return descendants of ``root`` by scanning parent PIDs in ``/proc/*/stat``."""
    try:
        pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return []
    children_of: dict[int, list[int]] = {}
    for p in pids:
        try:
            with open(f"/proc/{p}/stat") as fh:
                # The comm field may contain spaces and parentheses; after it, index 1 is
                # field 4, ppid.
                fields = fh.read().rsplit(") ", 1)[1].split()
            ppid = int(fields[1])
        except (OSError, IndexError, ValueError):
            continue
        children_of.setdefault(ppid, []).append(p)
    out: list[int] = []
    seen = {root}
    frontier = [root]
    while frontier:
        cur = frontier.pop()
        for k in children_of.get(cur, ()):
            if k not in seen:
                seen.add(k)
                out.append(k)
                frontier.append(k)
    return out


def _proc_cpu_ticks(pid: int) -> int:
    """Return process and child CPU ticks from ``/proc/<pid>/stat``."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            # The tail starts at field 3, so CPU fields 14-17 are at indices 11-14.
            fields = fh.read().rsplit(") ", 1)[1].split()
        # cutime/cstime fold in the pool workers a forkserver has already reaped.
        return int(fields[11]) + int(fields[12]) + int(fields[13]) + int(fields[14])
    except (OSError, IndexError, ValueError):
        return 0


def _child_cpu_snapshot() -> float:
    """Return CPU seconds used by reaped children and live descendants."""
    # Two snapshots around a fit subtract to the CPU the pool burned during it; a warm-up pool
    # taken before the first cancels out.
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    total = ru.ru_utime + ru.ru_stime
    for pid in _descendants(os.getpid()):
        total += _proc_cpu_ticks(pid) / _SC_CLK_TCK
    return total


def _vmrss_kb(pid: int) -> int:
    """Resident set size (KiB) of ``pid`` from ``/proc/<pid>/status``; 0 if the process is gone."""
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, IndexError, ValueError):
        return 0
    return 0


class _TreeRSSSampler(threading.Thread):
    """Sample peak RSS for the worker and its descendants during a fit."""

    def __init__(self, interval: float = 0.2) -> None:
        super().__init__(daemon=True)
        self._interval = interval
        # Named ``_stop_evt`` because ``Thread._stop`` is a method ``join`` calls.
        self._stop_evt = threading.Event()
        self.max_total_kb = 0
        self.max_children_kb = 0

    def _sample(self) -> None:
        self_pid = os.getpid()
        children_kb = sum(_vmrss_kb(p) for p in _descendants(self_pid))
        total_kb = _vmrss_kb(self_pid) + children_kb
        if children_kb > self.max_children_kb:
            self.max_children_kb = children_kb
        if total_kb > self.max_total_kb:
            self.max_total_kb = total_kb

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self._sample()
            self._stop_evt.wait(self._interval)

    def stop(self) -> None:
        """Stop the loop after one final sample."""
        self._stop_evt.set()
        self._sample()
        self.join(timeout=1.0)


#: Device selected by each explicit benchmark backend.
_BACKEND_DEVICE = {
    "torch-cuda": "cuda",
    "torch-cpu": "cpu",
    "numpy": "cpu",
    "numba": "cpu",
    "native": "cpu",
}


def _wants_cuda(backend: str) -> bool:
    """Return whether ``backend`` names CUDA, falling back to ``ANDREY_DEVICE`` if it is unknown."""
    device = _BACKEND_DEVICE.get(backend)
    if device is not None:
        return device == "cuda"
    return os.environ.get("ANDREY_DEVICE", "").lower() == "cuda"


def _cuda_reset_peak_memory(backend: str) -> None:
    """Reset PyTorch's peak memory counter when CUDA is available."""
    if not _wants_cuda(backend):
        return
    try:  # pragma: no cover - no CUDA in the local test environment
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _cuda_peak_alloc_mb(backend: str) -> float | None:
    """Return peak CUDA tensor memory in MiB since the last reset."""
    if not _wants_cuda(backend):
        return None
    try:  # pragma: no cover - no CUDA in the local test environment
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / _MIB
    except Exception:
        pass
    return None


def _cuda_synchronize(backend: str) -> None:
    """Wait for pending CUDA work when CUDA is available."""
    if not _wants_cuda(backend):
        return
    try:  # pragma: no cover - no CUDA in the local test environment
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _int_env(name: str) -> int | None:
    """Return an environment variable as an integer when possible."""
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return None


def _observed_device(backend: str) -> str:
    """Return the device ``backend`` names, resolved to the card when that device is CUDA."""
    if not _wants_cuda(backend):
        return "cpu"
    try:  # pragma: no cover - no CUDA in the local test environment
        import torch

        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            return f"cuda:{index}:{torch.cuda.get_device_name(index)}"
    except Exception:
        pass
    return "cpu"


def _mark(**fields: float) -> None:
    """Record completed stages in the child's start marker.

    Each stage writes its duration on completion so the parent can read it after killing the child.
    An empty marker means only that the child started.
    """
    if not _STARTED_PATH:
        return
    try:
        pathlib.Path(_STARTED_PATH).write_text(json.dumps(fields))
    except OSError:
        pass


def _run(adapter, data: np.ndarray, params, warmup: int) -> dict:
    """Run setup, warmups, and one timed fit, then return its metrics."""
    backend_name = getattr(adapter, "backend", "") or ""

    # Initialize interpreters and libraries that the adapter cannot unpickle. Setup precedes
    # warm-up fits and stays outside fit timing at `--warmup 0`, the reach-probe setting.
    setup = getattr(adapter, "setup", None)
    setup_s = None
    if setup is not None:
        s0 = time.perf_counter()
        setup()
        setup_s = time.perf_counter() - s0
        _mark(setup_s=setup_s)

    # Warm up JIT compilation, CUDA, and caches outside the fit timer.
    warmup_s = setup_s
    if warmup and warmup > 0:
        w0 = time.perf_counter()
        for _ in range(warmup):
            adapter.fit(data, params)
        _cuda_synchronize(backend_name)
        warmup_s = (setup_s or 0.0) + (time.perf_counter() - w0)
    if warmup_s is not None:
        _mark(setup_s=setup_s, warmup_s=warmup_s)

    # Sample tree RSS and child CPU around the timed fit.
    sampler = _TreeRSSSampler()
    sampler.start()

    _cuda_reset_peak_memory(backend_name)
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    c0 = _child_cpu_snapshot()
    t0 = time.perf_counter()
    adj = adapter.fit(data, params)
    _cuda_synchronize(backend_name)
    t1 = time.perf_counter()
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    c1 = _child_cpu_snapshot()  # Includes worker-pool CPU reaped after the fit.

    sampler.stop()

    # Pair fit-only wall time with CPU used by this process and its children.
    wall_s = t1 - t0
    self_cpu_s = (ru1.ru_utime + ru1.ru_stime) - (ru0.ru_utime + ru0.ru_stime)
    child_cpu_s = max(0.0, c1 - c0)
    cpu_s = self_cpu_s + child_cpu_s

    # Use sampled tree RSS for child processes and ``ru_maxrss`` for the worker.
    child_peak_rss_mb = sampler.max_children_kb / 1024.0
    peak_rss_mb = max(ru1.ru_maxrss / 1024.0, sampler.max_total_kb / 1024.0)

    adj = np.ascontiguousarray(adj)
    output_hash = hashlib.sha1(adj.tobytes()).hexdigest()

    return {
        "status": "ok",
        "wall_s": wall_s,
        "cpu_s": cpu_s,
        "child_cpu_s": child_cpu_s,
        "warmup_s": warmup_s,
        "setup_s": setup_s,
        "peak_rss_mb": peak_rss_mb,
        "child_peak_rss_mb": child_peak_rss_mb,
        "output_hash": output_hash,
        # Record the resource settings seen by the child.
        "threads": _int_env("OMP_NUM_THREADS"),
        "num_workers": _int_env("ANDREY_NUM_WORKERS"),
        "device_id": _observed_device(backend_name),
        "gpu_peak_alloc_mb": _cuda_peak_alloc_mb(backend_name),
        "adj": adj,
    }


def main(argv: list[str]) -> int:
    with open(argv[1], "rb") as fh:
        job = pickle.load(fh)

    with np.load(job["data_path"]) as npz:
        data = npz["data"]

    # Cast before timing. ``copy=False`` reuses data that already has the requested dtype.
    data = data.astype(job["dtype"], copy=False)

    result = _run(job["adapter"], data, job["params"], int(job.get("warmup", 1)))

    with open(job["result_path"], "wb") as fh:
        pickle.dump(result, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
