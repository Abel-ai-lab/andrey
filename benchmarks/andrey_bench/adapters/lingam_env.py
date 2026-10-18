"""The environment lingam's fits run in, and its package versions.

lingam 1.13.0 pins scipy 1.13.1 or older, and pygam, which lingam imports, pins scipy below 1.17;
neither installs beside the bench environment's scipy on its Python.
``benchmarks/start_lingam.sh`` builds ``.venv-lingam`` on Python 3.12, and an adapter that declares
``python_env = PYTHON_ENV_VAR`` has its fits started there. Only ``fit`` runs in that environment;
``to_structure`` and scoring run in the parent, so such an adapter's module imports nothing but
numpy at import time and builds its marks without Andrey.
"""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path

#: Name of the environment variable holding the interpreter these fits run in.
PYTHON_ENV_VAR = "ANDREY_BENCH_LINGAM_PYTHON"


@functools.lru_cache(maxsize=None)
def interpreter_version(python: str, distribution: str) -> str:
    """`distribution`'s version, as the interpreter that will run the fits reports it."""
    try:
        out = subprocess.run(
            [python, "-c", f"import importlib.metadata as m; print(m.version({distribution!r}))"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def env_version(distribution: str) -> str:
    """`distribution`'s version in the environment :data:`PYTHON_ENV_VAR` names, if set."""
    python = os.environ.get(PYTHON_ENV_VAR, "").strip()
    return interpreter_version(python, distribution) if python else "unknown"


def lingam_venv() -> Path:
    """Return the venv :data:`PYTHON_ENV_VAR` names, for ``provenance.json``.

    Raises ``SystemExit`` when the variable is unset or does not name a ``<venv>/bin/python``: the
    environment recorded must be the one the fits run in.
    """
    raw = os.environ.get(PYTHON_ENV_VAR, "").strip()
    if not raw:
        raise SystemExit(
            f"${PYTHON_ENV_VAR} is unset. Build the environment with benchmarks/start_lingam.sh "
            "and export its bin/python."
        )
    # Not resolve(): a venv's python is a symlink to the base interpreter.
    python = Path(raw).absolute()
    venv = python.parent.parent
    if venv / "bin" / "python" != python:
        raise SystemExit(f"${PYTHON_ENV_VAR}={raw!r} is not a <venv>/bin/python")
    return venv
