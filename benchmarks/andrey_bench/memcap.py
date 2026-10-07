"""Enforce memory caps on the worker's process tree.

* :data:`POLL`: sample tree RSS and kill the group on a breach, recording an OOM.
* :data:`SLURM`: use ``srun --mem=`` with ``ConstrainRAMSpace=yes`` for kernel enforcement;
  :func:`stderr_reports_oom` checks the step's reported reason.
"""

from __future__ import annotations

import math
import os
import re
import signal
import threading
from typing import Mapping

import psutil

#: Route names. ``NONE`` is what an uncapped run records.
POLL = "poll"
SLURM = "slurm"
NONE = "none"
ROUTES: tuple[str, ...] = (POLL, SLURM, NONE)

#: Sampling period for the polling route.
DEFAULT_POLL_INTERVAL_S = 0.05

_MB = 1024.0 * 1024.0


def job_mem_mb(env: Mapping[str, str] | None = None) -> float | None:
    """Return the Slurm job's memory allocation in MB, or ``None`` if unavailable."""
    e = os.environ if env is None else env
    if per_node := _float_or_none(e.get("SLURM_MEM_PER_NODE")):
        return per_node
    per_cpu = _float_or_none(e.get("SLURM_MEM_PER_CPU"))
    cpus = _float_or_none(e.get("SLURM_CPUS_ON_NODE"))
    return per_cpu * cpus if per_cpu and cpus else None


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def resolve_route(
    cap_mem_mb: float | None,
    requested: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Return ``NONE`` without a cap; otherwise default to ``POLL``.

    Reject a cap paired with ``NONE``.
    """
    if requested is not None and requested not in ROUTES:
        raise ValueError(f"unknown memory-cap route {requested!r}; expected one of {ROUTES}")
    if cap_mem_mb is None:
        return NONE
    if requested == NONE:
        raise ValueError(
            f"cap_mem_mb={cap_mem_mb} with route {NONE!r} records a cap nothing enforces; "
            f"drop the cap, or route it through {POLL!r} or {SLURM!r}"
        )
    route = requested or POLL
    if route == SLURM:
        # Each fit is an srun step asking for --mem=<cap>. A step wanting more than the job owns is
        # not schedulable, and Slurm queues it rather than refusing it: the run holds the node with
        # no step, no record, and no error until the wall runs out. Only raise when the job's memory
        # is actually readable — off-scheduler this is silent, not blocked.
        owned = job_mem_mb(env)
        if owned is not None and cap_mem_mb > owned:
            raise ValueError(
                f"cap_mem_mb={cap_mem_mb:.0f} exceeds the {owned:.0f} MB this job owns, and route "
                f"{SLURM!r} runs each fit as 'srun --mem={int(math.ceil(cap_mem_mb))}M'. Slurm "
                "would block that step instead of failing it. Raise the job's --mem above the cap, "
                f"or route the cap through {POLL!r}."
            )
    return route


def srun_prefix(cap_mem_mb: float) -> list[str]:
    """Return the ``srun`` prefix for a Slurm memory cap."""
    return ["srun", "--ntasks=1", f"--mem={int(math.ceil(cap_mem_mb))}M"]


#: Slurm has worded the OOM message several ways across versions, so match the stable fragments
#: rather than one release's sentence.
_OOM_STDERR = re.compile(
    r"oom[-_ ]?kill|out[- ]of[- ]memory|exceeded (?:step|job) memory limit",
    re.IGNORECASE,
)


def stderr_reports_oom(text: str | None) -> bool:
    """Return whether Slurm stderr indicates an OOM kill."""
    return bool(text) and bool(_OOM_STDERR.search(text))


def tree_rss_mb(pid: int) -> float:
    """Return summed RSS in MB for a process and its descendants."""
    try:
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0
    total = 0
    for proc in procs:
        try:
            total += proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    return total / _MB


def _kill_tree(pid: int) -> None:
    """Kill the process group if it still exists.

    The child must have its own session (``setsid``) to avoid killing the caller's group.
    """
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


class TreeMemoryPoller(threading.Thread):
    """Kill the process group when sampled tree RSS exceeds the cap.

    ``breached`` records a cap violation; ``peak_mb`` may underestimate the true peak.
    """

    def __init__(
        self,
        pid: int,
        cap_mem_mb: float,
        *,
        interval: float = DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        super().__init__(daemon=True)
        self.pid = pid
        self.cap_mem_mb = float(cap_mem_mb)
        self.peak_mb = 0.0
        self.breached = False
        self._interval = interval
        # Not ``_stop`` — that name shadows ``threading.Thread._stop``, which ``join`` calls.
        self._stop_evt = threading.Event()

    def sample(self) -> float:
        """Sample tree RSS and enforce the cap."""
        total = tree_rss_mb(self.pid)
        if total > self.peak_mb:
            self.peak_mb = total
        if not self.breached and total > self.cap_mem_mb:
            self.breached = True
            _kill_tree(self.pid)
        return total

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self.sample()
            self._stop_evt.wait(self._interval)

    def stop(self) -> None:
        """Stop polling and join a running thread, including after an interrupted start."""
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=2.0)


__all__ = [
    "POLL",
    "SLURM",
    "NONE",
    "ROUTES",
    "DEFAULT_POLL_INTERVAL_S",
    "TreeMemoryPoller",
    "resolve_route",
    "srun_prefix",
    "stderr_reports_oom",
    "tree_rss_mb",
]
