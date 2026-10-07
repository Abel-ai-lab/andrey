"""Run benchmark repeats in fresh interpreters and return raw measurements.

Each repeat runs :mod:`andrey_bench._worker`, isolating BLAS, JIT, CUDA, and allocator state and
resetting ``ru_maxrss``. A supervisor cleans up the process group when the driver or worker exits.
Adapters may select a separate interpreter for incompatible dependencies. Graph conversion and
scoring run in the parent, outside fit timing.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from andrey_bench import memcap
from andrey_bench.contracts import SolutionAdapter, Status, thread_env

# --------------------------------------------------------------------------------------------------
# Locations. BENCH_ROOT is the directory that *contains* the ``andrey_bench`` package, so putting it
# on the child ``PYTHONPATH`` is what makes ``python -m andrey_bench._worker`` resolve.
# --------------------------------------------------------------------------------------------------
BENCH_ROOT = Path(__file__).resolve().parent.parent

#: The shared venv python a ``fit`` is spawned in unless its solution names another.
BENCH_PYTHON = BENCH_ROOT / ".venv-bench" / "bin" / "python"

#: Every fit is launched under this, so the fit's whole group dies with the driver. Run by path,
#: not ``-m``: it is standard library only, so any solution's interpreter can run it as a script.
SUPERVISOR = BENCH_ROOT / "andrey_bench" / "_supervisor.py"


def available_cores() -> int:
    """Return the affinity CPU count, falling back to the machine count or 1.

    Affinity respects Slurm and ``taskset`` limits, avoiding oversubscription. Kept independent of
    Andrey so alternate benchmark builds need not provide its CPU helper.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux — no affinity mask to read
        return os.cpu_count() or 1


def split_cores(cores: int, mode: str) -> tuple[int, int]:
    """Return ``(threads, num_workers)`` with ``threads * num_workers == cores``.

    Parallel mode uses ``(1, cores)`` to avoid nested BLAS oversubscription; all other modes use
    ``(cores, 1)``.
    """
    if mode == "parallel":
        return (1, cores)
    return (cores, 1)


def _interpreter(var: str, raw: str) -> Path:
    """Validate ``raw`` as an executable file, naming ``var`` in errors.

    Make the path absolute before the child changes directory. Preserve symlinks so a venv's
    ``bin/python`` retains its environment.
    """
    python = Path(raw).absolute()
    if not python.exists():
        raise FileNotFoundError(f"{var}={raw!r} does not point at an existing interpreter")
    # A directory carries the execute bit as *search* permission, so ``os.access(X_OK)`` alone
    # accepts the venv itself - the most natural thing to point this at by mistake. Every fit would
    # then fail deep in a subprocess launch instead of here, where the variable naming it is still
    # in hand.
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"{var}={raw!r} is not an executable file")
    return python


def _bench_python(adapter: SolutionAdapter | None = None) -> Path:
    """Resolve the interpreter for ``adapter``'s fits.

    ``adapter.python_env`` names an environment variable and takes precedence over
    ``ANDREY_BENCH_PYTHON`` and the default bench-env. Missing interpreters raise before launch.
    """
    var = getattr(adapter, "python_env", None)
    if var:
        raw = os.environ.get(var, "").strip()
        if not raw:
            name = getattr(adapter, "name", type(adapter).__name__)
            raise FileNotFoundError(
                f"${var} is unset: {name} runs its fits in an environment of its own. Build it and "
                f"export ${var}."
            )
        return _interpreter(var, raw)
    shared = os.environ.get("ANDREY_BENCH_PYTHON", "").strip()
    if shared:
        return _interpreter("ANDREY_BENCH_PYTHON", shared)
    if not BENCH_PYTHON.exists():
        raise FileNotFoundError(f"{BENCH_PYTHON} not found — run benchmarks/start.sh")
    return BENCH_PYTHON


def _child_env(
    threads: int,
    adapter: SolutionAdapter,
    *,
    num_workers: int = 1,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the child environment with thread limits and import paths for the worker and adapter.

    Remove inherited ``ANDREY_NUM_WORKERS``, ``ANDREY_GES_PARALLEL_MIN_WORK``, and
    ``ANDREY_HC_PARALLEL_MIN_WORK``, then set ``ANDREY_NUM_WORKERS`` to ``num_workers``.
    Apply ``extra_env`` last.
    """
    env = dict(os.environ)
    env.update(thread_env(threads))  # BLAS/numeric pools pinned before the child imports numpy.

    # Remove inherited Andrey worker settings, including both algorithm-specific thresholds.
    # Set the worker count allocated to this task.
    env.pop("ANDREY_NUM_WORKERS", None)
    env.pop("ANDREY_GES_PARALLEL_MIN_WORK", None)
    env.pop("ANDREY_HC_PARALLEL_MIN_WORK", None)
    env["ANDREY_NUM_WORKERS"] = str(num_workers)

    paths = [str(BENCH_ROOT)]
    # The adapter's defining module must be importable in the child so pickle can rebuild it — for a
    # test stub that lives outside the package this is what puts it on the path.
    module = sys.modules.get(type(adapter).__module__)
    module_file = getattr(module, "__file__", None)
    if module_file:
        paths.append(str(Path(module_file).resolve().parent))
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)

    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    return env


def _int_or_none(value: str | None) -> int | None:
    """Return ``value`` as an int, or ``None`` if invalid."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _marker_stages(started_path: Path) -> dict[str, float | None]:
    """Read completed ``setup_s`` and ``warmup_s`` from the child's marker.

    Written before and after warmup fits, these remain available after the child is killed.
    """
    try:
        stages = json.loads(started_path.read_text())
    except (OSError, ValueError):
        return {"setup_s": None, "warmup_s": None}
    return {name: stages.get(name) for name in ("setup_s", "warmup_s")}


def resolved_env_fields(env: Mapping[str, str]) -> dict[str, Any]:
    """Read ``run_id`` environment fields from the child's environment."""
    return {
        "env_threads": _int_or_none(env.get("OMP_NUM_THREADS")),
        "env_num_workers": _int_or_none(env.get("ANDREY_NUM_WORKERS")),
        "env_device": (env.get("ANDREY_DEVICE") or "cpu").strip().lower(),
        "env_ges_min_work": _int_or_none(env.get("ANDREY_GES_PARALLEL_MIN_WORK")),
        "env_hc_min_work": _int_or_none(env.get("ANDREY_HC_PARALLEL_MIN_WORK")),
    }


def planned_env_fields(
    adapter: SolutionAdapter,
    *,
    threads: int,
    num_workers: int,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve the child's environment fields before launch, including ``extra_env`` overrides.

    Use these fields to name resume parts consistently with :func:`run_task`.
    """
    return resolved_env_fields(
        _child_env(threads, adapter, num_workers=num_workers, extra_env=extra_env)
    )


def check_interpreters(adapters: Sequence[SolutionAdapter]) -> dict[str, Path]:
    """Return ``{solution name: interpreter}``, raising if an environment is missing.

    Called before dataset materialization to fail the campaign before measuring any units.
    """
    return {adapter.name: _bench_python(adapter) for adapter in adapters}


@functools.lru_cache(maxsize=1)
def _scheduler() -> str:
    """Return cached scheduler JSON: job, node, partition and CPUs.

    A local UUID substitutes for the Slurm job ID and distinguishes local resumptions.
    """
    return json.dumps(
        {
            "job": os.environ.get("SLURM_JOB_ID") or f"local-{uuid.uuid4().hex[:12]}",
            "node": os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "cpus": os.environ.get("SLURM_CPUS_ON_NODE"),
        },
        sort_keys=True,
    )


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the child's whole process group; tolerate an already-dead child."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run_one_repeat(
    *,
    python: Path,
    env: Mapping[str, str],
    adapter: SolutionAdapter,
    data_path: Path,
    dtype: str,
    params: Mapping[str, Any],
    cap_wall_s: float,
    cap_mem_mb: float | None,
    mem_route: str | None,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    """Run one worker with wall and memory caps and return its measurements.

    The child loads ``data_path`` and casts to ``dtype``. Failure classification checks the wall
    timeout, poller breach, Slurm OOM report, then the exit-code fallback.
    """
    route = memcap.resolve_route(cap_mem_mb, mem_route)
    # Parent-declared, so every one of these is on a timeout / oom record too — and ``dtype`` among
    # them, because a ``run_id`` field only a surviving child could report would not be an identity.
    common = {
        "scheduler": _scheduler(),
        "repeat": repeat,
        "cap_wall_s": cap_wall_s,
        "cap_mem_mb": cap_mem_mb,
        "mem_route": route,
        "warmup": warmup,
        "dtype": dtype,
        **resolved_env_fields(env),
    }
    with tempfile.TemporaryDirectory(prefix="andrey_bench_") as tmp:
        tmpdir = Path(tmp)
        job_path = tmpdir / "job.pkl"
        result_path = tmpdir / "result.pkl"
        exit_path = tmpdir / "exit_code"
        # The child touches this first; its absence after a wall-cap kill means it never started.
        started_path = tmpdir / "started"

        job = {
            "adapter": adapter,
            "data_path": str(data_path),
            "dtype": dtype,
            "params": dict(params),
            "result_path": str(result_path),
            "started_path": str(started_path),
            "warmup": warmup,
        }
        with open(job_path, "wb") as fh:
            pickle.dump(job, fh)

        cmd = [str(python), "-m", "andrey_bench._worker", str(job_path), str(started_path)]
        if route == memcap.SLURM:
            cmd = memcap.srun_prefix(float(cap_mem_mb)) + cmd  # type: ignore[arg-type]

        start = time.perf_counter()
        # The driver holds the write end for as long as it lives, and the supervisor blocks on the
        # read end. However the driver dies - ``SIGKILL``, an OOM, a job wall - the kernel closes
        # its copy, the read reaches end of file, and the supervisor kills the group. Only the
        # supervisor gets a copy (``pass_fds`` closes every other fd), or that end never arrives.
        proc = None
        poller = None
        timed_out = False
        elapsed = None
        death_r, death_w = os.pipe()
        try:
            try:
                proc = subprocess.Popen(
                    [str(python), str(SUPERVISOR), str(death_r), str(exit_path), "--", *cmd],
                    env=dict(env),
                    cwd=str(BENCH_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    # The supervisor, fit, and pool share one group for both caps and cleanup.
                    preexec_fn=os.setsid,
                    pass_fds=(death_r,),
                )
            finally:
                os.close(death_r)

            if route == memcap.POLL:
                poller = memcap.TreeMemoryPoller(proc.pid, float(cap_mem_mb))  # type: ignore[arg-type]
                poller.start()
            try:
                _out, err = proc.communicate(timeout=cap_wall_s)
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - start
                timed_out = True
                _kill_group(proc)
                _out, err = proc.communicate()  # reap the killed child, drain its pipes
        except BaseException:
            # The driver is unwinding: Ctrl-C, ``SystemExit``, an error nobody planned for. The
            # supervisor waits on this process's death, which a caller that catches and carries on
            # never delivers, so take the group down here rather than leave it to a driver that may
            # yet survive.
            if proc is not None:
                _kill_group(proc)
                proc.wait()
            raise
        finally:
            # Close the death pipe even if poller startup failed or stopping it is interrupted.
            os.close(death_w)
            if poller is not None:
                poller.stop()
            if proc is not None:
                proc.stdout.close()
                proc.stderr.close()

        observed = {"tree_peak_mem_mb": poller.peak_mb if poller is not None else None}
        if timed_out:
            return {
                **common,
                **observed,
                # A step that never started says nothing about speed. Under ``--mem-route slurm`` a
                # step whose memory cannot be scheduled blocks until this same wall cap, and
                # recording that as a timeout censors a solution that would have finished -- and is
                # indistinguishable afterwards from a real ceiling. The marker is the child's first
                # act, so its absence says the child never ran.
                # Only the slurm route asks a scheduler for a step and so can block. Elsewhere a
                # missing marker means the child died during startup, which the cap covers.
                "status": (
                    Status.BLOCKED.value
                    if route == memcap.SLURM and not started_path.exists()
                    else Status.TIMEOUT.value
                ),
                "elapsed_at_kill_s": elapsed,
                # Durations are present only for completed stages. A warm-up is a complete fit, so
                # `warmup_s - setup_s` bounds fit time when `warmup >= 1`. At `warmup=0`, the
                # durations are equal and provide no fit-time bound.
                **_marker_stages(started_path),
                "error_text": (
                    None
                    if started_path.exists() or route != memcap.SLURM
                    else "the child never started: the step was never scheduled, so this is not a "
                    "timeout of the solution. Under --mem-route slurm give the job more memory "
                    "than --cap-mem-mb."
                ),
            }

        err_text = err.decode("utf-8", "replace") if err else ""

        # A breach outranks a clean exit: the child may have raced past the cap and still finished,
        # but a fit that exceeded its declared budget is not a measurement of that budget.
        if poller is not None and poller.breached:
            return {
                **common,
                **observed,
                # Preserve completed stages, including R loading, after a memory-cap kill, as after
                # a wall-cap kill; both discard the child's remaining results.
                **_marker_stages(started_path),
                "status": Status.OOM.value,
                "oom_source": "poller",
                "exit_code": proc.returncode,
                "stderr": err_text[-4000:],
            }

        # The supervisor kills its group on worker exit. Recover the worker's saved status only
        # after the wall-cap and poller checks, which take precedence over a worker's own exit.
        rc = proc.returncode
        with contextlib.suppress(OSError, ValueError):
            rc = int(exit_path.read_text())

        if rc == 0 and result_path.exists():
            with open(result_path, "rb") as fh:
                measures = pickle.load(fh)
            return {**common, **observed, **measures}

        return {
            **common,
            **observed,
            **_marker_stages(started_path),
            **_classify_death(rc, err_text, route=route, cap_mem_mb=cap_mem_mb),
            "exit_code": rc,
            "stderr": err_text[-4000:],
        }


def _classify_death(
    rc: int, stderr: str, *, route: str, cap_mem_mb: float | None
) -> dict[str, Any]:
    """Return ``{status[, oom_source]}`` for a death not caused by the parent.

    Slurm OOM reports use ``oom_source="cgroup"``. Under a declared cap, a bare ``SIGKILL`` uses
    ``"external"``: it may be an OOM between poller samples or an unrelated kill. Other deaths are
    errors.
    """
    if route == memcap.SLURM and memcap.stderr_reports_oom(stderr):
        return {"status": Status.OOM.value, "oom_source": "cgroup"}
    if cap_mem_mb is not None and rc == -signal.SIGKILL:
        return {"status": Status.OOM.value, "oom_source": "external"}
    return {"status": Status.ERROR.value}


def run_task(
    adapter: SolutionAdapter,
    data: np.ndarray | str | Path,
    params: Mapping[str, Any],
    *,
    cap_wall_s: float,
    cap_mem_mb: float | None = None,
    mem_route: str | None = None,
    repeats: int = 5,
    warmup: int = 1,
    threads: int = 1,
    num_workers: int = 1,
    dtype: str = "float64",
    extra_env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Run isolated subprocess repeats and return raw measurements.

    Each child runs untimed warmups followed by one timed ``fit``. Graph conversion and scoring
    are left to the caller. The parent enforces caps and classifies failures.

    Parameters
    ----------
    adapter, data, params
        Picklable :class:`~andrey_bench.contracts.SolutionAdapter`, input and fit parameters.
        An ``ndarray`` is serialized once to a task-scoped ``.npz`` shared by all repeats.
        An existing ``.npz`` path is passed through without reserialization.
    dtype
        Declared fit precision and ``run_id`` field. The child casts before timing; canonical
        datasets are stored as ``float64``, the default.
    cap_wall_s
        Per-repeat wall cap, including imports, warmup, and fit. Exceeding it kills the group and
        returns ``status="timeout"`` with ``elapsed_at_kill_s``.
    cap_mem_mb, mem_route
        Optional process-tree memory cap. ``"poll"`` (default with a cap) samples RSS and kills on
        a breach; ``"slurm"`` uses ``srun --mem=`` for cgroup enforcement. OOM records include
        ``oom_source``. With ``cap_mem_mb=None``, memory is uncapped and a bare ``SIGKILL`` is an
        error. Only polling reports ``tree_peak_mem_mb``; successful Slurm fits report their own
        ``peak_rss_mb`` because the step may run on another node.
    repeats, warmup, threads
        Timed repeats, untimed warmups per repeat, and BLAS/numeric threads per child.
    num_workers
        Pool size, exported as ``ANDREY_NUM_WORKERS`` after removing its inherited value.
        :func:`split_cores` sets ``threads * num_workers`` to the core budget.
    extra_env
        Optional child-environment overrides, applied last.

    Returns
    -------
    list[dict]
        One dict per repeat, including ``status``, caps, ``mem_route``, ``tree_peak_mem_mb``,
        ``warmup``, ``dtype``, and :func:`resolved_env_fields`, even if the child dies. Successful
        repeats include ``wall_s``, ``cpu_s``, ``warmup_s``, ``peak_rss_mb``, ``adj``,
        ``output_hash``, and child-reported ``threads``, ``num_workers``, and ``device_id``.
    """
    python = _bench_python(adapter)
    env = _child_env(threads, adapter, num_workers=num_workers, extra_env=extra_env)

    results: list[dict[str, Any]] = []
    with contextlib.ExitStack() as stack:
        if isinstance(data, (str, Path)):
            # Campaign path: an already-materialized file — pass it straight through, serialize nothing.
            resolved = Path(data).resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"data file for the task not found: {resolved}")
        else:
            # In-memory array: write the canonical bytes once into a task-scoped tempdir that
            # outlives every repeat (torn down when this ExitStack closes), never per repeat.
            tmp = stack.enter_context(tempfile.TemporaryDirectory(prefix="andrey_bench_data_"))
            resolved = Path(tmp) / "data.npz"
            np.savez(resolved, data=np.asarray(data))

        for repeat in range(repeats):
            results.append(
                _run_one_repeat(
                    python=python,
                    env=env,
                    adapter=adapter,
                    data_path=resolved,
                    dtype=dtype,
                    params=params,
                    cap_wall_s=cap_wall_s,
                    cap_mem_mb=cap_mem_mb,
                    mem_route=mem_route,
                    warmup=warmup,
                    repeat=repeat,
                )
            )
    return results
