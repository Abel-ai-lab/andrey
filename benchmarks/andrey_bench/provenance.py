"""Capture benchmark build and machine identity in ``provenance.json``.

Versions (``uv pip freeze``) and numpy BLAS/LAPACK linkage are read from each benchmark
environment's own interpreter, which may differ from the capturing process. :func:`capture`
records CPU, GPU, git revision, a caller-supplied timestamp, and any ``extra_venvs`` under
``environments``. Benchmark flags belong in ``run_meta.json``; generator calls in ``registry.json``.

Unavailable probes are recorded as errors or missing values so partial capture remains usable.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Mapping

# benchmarks/ root (…/bench) and the venv fits run in, resolved from this file's location so capture
# works regardless of the caller's cwd.
_BENCH_ROOT = Path(__file__).resolve().parents[1]
_BENCH_VENV = _BENCH_ROOT / ".venv-bench"


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 60) -> tuple[int, str, str]:
    """Return ``(returncode, stdout, stderr)`` without raising; a missing binary returns 127."""
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"{cmd[0]}: timeout after {timeout}s"
    except Exception as exc:  # pragma: no cover - defensive
        return 1, "", f"{cmd[0]}: {exc}"


def _freeze(venv: Path) -> dict:
    """Read ``venv``'s package versions with ``uv pip freeze``.

    Returns ``{"python": ver, "packages": {name: ver}, ...}``. Falls back to the environment's pip;
    records an ``error`` field if the environment is missing or both freeze commands fail.
    """
    py = venv / "bin" / "python"
    entry: dict = {"path": str(venv), "source": "uv pip freeze"}
    if not py.exists():
        entry["error"] = "venv python not found"
        return entry

    rc_v, out_v, _ = _run([str(py), "--version"])
    entry["python"] = out_v.strip() or None

    rc, out, err = _run(["uv", "pip", "freeze", "--python", str(py)])
    if rc != 0:
        # Fall back to the venv's own pip if uv is unavailable.
        rc, out, err = _run([str(py), "-m", "pip", "freeze"])
    if rc != 0:
        entry["error"] = (err or "freeze failed").strip()
        return entry

    packages: dict[str, str] = {}
    editables: list[str] = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Using "):
            continue
        if "==" in line:
            pkg, ver = line.split("==", 1)
            packages[pkg.strip().lower()] = ver.strip()
        else:
            # Keep editable installs and VCS/URL requirements verbatim.
            editables.append(line)
    entry["packages"] = packages
    if editables:
        entry["editable"] = editables
    return entry


# Probe run *inside* the interpreter being described, so the identity is that interpreter's own.
# Prints one JSON object on stdout; degrades to a ``note`` on a numpy without dict-mode show_config.
_BLAS_PROBE = """
import json
import numpy as np
out = {"numpy": np.__version__}
try:
    cfg = np.show_config(mode="dicts") or {}
except Exception:
    out["note"] = "show_config dict mode unavailable"
else:
    deps = cfg.get("Build Dependencies", {})
    for lib in ("blas", "lapack"):
        info = deps.get(lib, {})
        if info:
            out[lib] = {
                k: info.get(k)
                for k in ("name", "found", "version", "detection method", "openblas configuration")
                if k in info
            }
    simd = cfg.get("SIMD Extensions", {})
    if simd:
        out["simd"] = simd
print(json.dumps(out, default=str))
"""


def _blas_identity(venv: Path) -> dict:
    """Read numpy's BLAS/LAPACK identity by running :data:`_BLAS_PROBE` inside ``venv``.

    The capturing process may use different linkage. Returns an ``error`` field if the interpreter,
    numpy, or probe output is unusable.
    """
    py = venv / "bin" / "python"
    if not py.exists():
        return {"error": "venv python not found"}
    rc, out, err = _run([str(py), "-c", _BLAS_PROBE])
    if rc != 0:
        return {"error": (err or "blas probe failed").strip()}
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        return {"error": f"blas probe output unparseable: {exc}"}


def _cpu_identity() -> dict:
    """CPU model + logical core count (``platform`` / ``/proc/cpuinfo`` model name)."""
    info: dict = {
        "processor": platform.processor() or None,
        "machine": platform.machine(),
        "system": platform.system(),
        "logical_cores": os.cpu_count(),
    }
    model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            for line in cpuinfo.read_text().splitlines():
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
        except Exception:  # pragma: no cover - defensive
            pass
    info["model"] = model
    return info


def _gpu_identity() -> dict:
    """Capture ``nvidia-smi`` output; return ``{"available": False}`` if unavailable."""
    rc, out, err = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    if rc != 0:
        return {"available": False, "reason": (err or "nvidia-smi unavailable").strip()}
    gpus = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return {"available": bool(gpus), "gpus": gpus}


def _git_sha(worktree: Path) -> str | None:
    rc, out, _ = _run(["git", "rev-parse", "HEAD"], cwd=worktree)
    return out.strip() if rc == 0 else None


def capture(
    out_dir: str | Path,
    *,
    timestamp: str,
    extra_venvs: Mapping[str, str | Path] | None = None,
) -> dict:
    """Write ``out_dir/provenance.json`` and return the captured dict.

    Parameters
    ----------
    out_dir
        Output directory, created if absent.
    timestamp
        Caller-supplied ISO-8601 string; the capture never reads the wall clock.
    extra_venvs
        Additional fit environments as ``{label: venv_directory}``. Each is frozen and BLAS-probed
        under ``environments[label]``; top-level ``versions`` and ``blas`` describe the shared
        environment. Pass the environment directory, not its ``bin/python``.

    Returns
    -------
    dict
        The provenance object, with keys ``versions``, ``blas``, ``cpu``, ``gpu``, ``git_sha``,
        ``timestamp``, and ``environments``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    prov = {
        "timestamp": timestamp,
        "versions": _freeze(_BENCH_VENV),
        "blas": _blas_identity(_BENCH_VENV),
        "cpu": _cpu_identity(),
        "gpu": _gpu_identity(),
        "git_sha": _git_sha(_BENCH_ROOT),
        "environments": {
            label: {"versions": _freeze(Path(venv)), "blas": _blas_identity(Path(venv))}
            for label, venv in (extra_venvs or {}).items()
        },
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True))
    return prov


__all__ = ["capture"]
