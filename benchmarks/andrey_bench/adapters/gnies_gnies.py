"""Adapter for gnies 0.3.3, the GnIES package of the method's authors, in an environment of its own.

Its fits run in ``.venv-gnies`` (``benchmarks/start_gnies.sh``), whose interpreter
``ANDREY_BENCH_GNIES_PYTHON`` names: a separate environment keeps the pins exact (gnies 0.3.3, ges
1.1.1, NumPy 2.5.4 on Python 3.12) and the PyPI module ``ges``, which gnies imports, out of
bench-env. Each call
starts ``benchmarks/gnies_worker.py`` there with one ``.npz`` request and reads one JSON answer; the
reference's 0/1 matrix becomes Andrey's endpoint marks here, in the parent (:func:`marks`), so one
scorer serves every package.

The fit takes the environments as a list of row blocks, ``[X[env == k] for k in range(K)]``, codes
in sorted-label order, and no labels. A call past its cap is killed and answers
``{"status": "timeout"}``, never an empty graph; a call that raises answers ``{"status": "error"}``
with the end of its error output (the worker raises when a local score is NaN or the total is not
finite). BLAS runs one thread.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from andrey.core import ARROW, TAIL

#: The environment variable that holds the reference's interpreter.
PYTHON_ENV_VAR = "ANDREY_BENCH_GNIES_PYTHON"

#: The script the reference's interpreter runs.
WORKER = Path(__file__).resolve().parents[2] / "gnies_worker.py"

#: Thread variables pinned to 1 for every call.
ONE_THREAD = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}


def interpreter() -> str:
    """The reference's interpreter; raises when ``ANDREY_BENCH_GNIES_PYTHON`` is unset."""
    python = os.environ.get(PYTHON_ENV_VAR, "").strip()
    if not python:
        raise RuntimeError(f"set {PYTHON_ENV_VAR}; benchmarks/start_gnies.sh builds it")
    return python


@functools.lru_cache(maxsize=None)
def versions(python: str) -> dict[str, str]:
    """gnies, ges, and NumPy versions as the reference's interpreter reports them."""
    code = (
        "import importlib.metadata as m, json, numpy; "
        "print(json.dumps({'gnies': m.version('gnies'), 'ges': m.version('ges'), "
        "'numpy': numpy.__version__}))"
    )
    done = subprocess.run([python, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    return json.loads(done.stdout) if done.returncode == 0 else {}


def marks(icpdag: list[list[int]]) -> np.ndarray:
    """The reference's 0/1 matrix as Andrey's endpoint marks: ``A[i, j]`` alone is ``i -> j``,
    both directions an undirected edge."""
    A = np.asarray(icpdag) != 0
    out = np.zeros(A.shape, dtype=np.int8)
    out[A | A.T] = TAIL
    out[A.T & ~A] = ARROW
    return out


def request(
    X: np.ndarray, env: np.ndarray, mode: str, *, cap_s: float | None = None, **fields: Any
) -> dict[str, Any]:
    """Run one ``gnies_worker.py`` request and return its answer with a ``status``.

    ``mode`` and ``fields`` are as the worker's docstring gives them. ``status`` is ``"ok"``,
    ``"timeout"`` (past ``cap_s`` seconds, the worker killed), or ``"error"`` (with
    ``error_text``, the last lines of the worker's error output).
    """
    python = interpreter()
    with tempfile.TemporaryDirectory(prefix="andrey_bench_gnies_") as tmp:
        path = Path(tmp) / "request.npz"
        np.savez(path, X=np.asarray(X, dtype=np.float64), env=env, mode=mode, **fields)
        try:
            done = subprocess.run(
                [python, "-I", str(WORKER), str(path)],
                capture_output=True,
                text=True,
                timeout=cap_s,
                env={**os.environ, **ONE_THREAD},
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error_text": f"past the {cap_s:g} s cap"}
    if done.returncode != 0:
        tail = "\n".join(done.stderr.strip().splitlines()[-5:])
        return {"status": "error", "error_text": tail or f"exit code {done.returncode}"}
    return {"status": "ok", **json.loads(done.stdout)}
